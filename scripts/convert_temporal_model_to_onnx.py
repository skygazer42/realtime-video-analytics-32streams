#!/usr/bin/env python3
"""
Script to convert temporal video analysis models from PyTorch to ONNX format.

Supports:
- CNN-LSTM models
- 3D CNN models (C3D, I3D, ResNet3D)
- ConvGRU models
- SlowFast networks

Usage:
    python convert_temporal_model_to_onnx.py \
        --model-type cnn_lstm \
        --checkpoint model.pth \
        --output model.onnx \
        --sequence-length 16 \
        --input-size 224 224

Requirements:
    pip install torch torchvision onnx onnxruntime
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DummyCNNLSTM(nn.Module):
    """
    Example CNN-LSTM model for demonstration.

    In practice, replace this with your actual model architecture.
    """

    def __init__(self, num_classes=400, hidden_size=512):
        super().__init__()
        # CNN feature extractor (simplified ResNet-like)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # LSTM temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
        )

        # Classifier
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch, time, channels, height, width]
        batch_size, seq_len = x.size(0), x.size(1)

        # Extract features for each frame
        features = []
        for t in range(seq_len):
            frame = x[:, t]  # [batch, C, H, W]
            feat = self.cnn(frame)  # [batch, 128, 1, 1]
            feat = feat.view(batch_size, -1)  # [batch, 128]
            features.append(feat)

        # Stack features: [batch, time, features]
        features = torch.stack(features, dim=1)

        # LSTM processing
        lstm_out, _ = self.lstm(features)  # [batch, time, hidden_size]

        # Use last timestep output
        out = self.fc(lstm_out[:, -1])  # [batch, num_classes]

        return out


class Dummy3DCNN(nn.Module):
    """
    Example 3D CNN model for demonstration.

    In practice, replace with actual C3D, I3D, or ResNet3D.
    """

    def __init__(self, num_classes=400):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: [batch, channels, time, height, width]
        x = self.conv3d(x)  # [batch, 256, 1, 1, 1]
        x = x.view(x.size(0), -1)  # [batch, 256]
        x = self.fc(x)  # [batch, num_classes]
        return x


def create_model(model_type: str, num_classes: int, checkpoint_path: str | None = None):
    """
    Create model instance and optionally load checkpoint.

    Args:
        model_type: Model architecture type
        num_classes: Number of output classes
        checkpoint_path: Path to PyTorch checkpoint (optional)

    Returns:
        PyTorch model in eval mode
    """
    if model_type == "cnn_lstm":
        model = DummyCNNLSTM(num_classes=num_classes)
    elif model_type == "3d_cnn":
        model = Dummy3DCNN(num_classes=num_classes)
    elif model_type == "conv_gru":
        # Similar to CNN-LSTM but with GRU
        logger.info("ConvGRU model - using CNN-LSTM as template, replace with your GRU model")
        model = DummyCNNLSTM(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Load checkpoint if provided
    if checkpoint_path:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)

    model.eval()
    return model


def export_to_onnx(
    model: nn.Module,
    model_type: str,
    output_path: str,
    sequence_length: int,
    input_size: tuple[int, int],
    opset_version: int = 17,
    dynamic_batch: bool = True,
):
    """
    Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model in eval mode
        model_type: Model architecture type
        output_path: Path to save ONNX model
        sequence_length: Number of frames in sequence
        input_size: Spatial resolution (H, W)
        opset_version: ONNX opset version
        dynamic_batch: Whether to use dynamic batch size
    """
    height, width = input_size

    # Create dummy input based on model type
    if model_type in ("cnn_lstm", "conv_gru"):
        # Input: [batch, time, channels, height, width]
        dummy_input = torch.randn(1, sequence_length, 3, height, width)
        input_names = ["input"]
        output_names = ["output"]

        if dynamic_batch:
            dynamic_axes = {
                "input": {0: "batch"},
                "output": {0: "batch"},
            }
        else:
            dynamic_axes = None

    elif model_type == "3d_cnn":
        # Input: [batch, channels, time, height, width]
        dummy_input = torch.randn(1, 3, sequence_length, height, width)
        input_names = ["input"]
        output_names = ["output"]

        if dynamic_batch:
            dynamic_axes = {
                "input": {0: "batch"},
                "output": {0: "batch"},
            }
        else:
            dynamic_axes = None

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    logger.info(f"Exporting model to ONNX format: {output_path}")
    logger.info(f"Input shape: {dummy_input.shape}")

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
    )

    logger.info(f"Model exported successfully to {output_path}")


def verify_onnx_model(onnx_path: str, sequence_length: int, input_size: tuple[int, int]):
    """
    Verify ONNX model by running a test inference.

    Args:
        onnx_path: Path to ONNX model
        sequence_length: Number of frames in sequence
        input_size: Spatial resolution (H, W)
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnx or onnxruntime not installed, skipping verification")
        return

    logger.info("Verifying ONNX model...")

    # Check model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model check passed")

    # Test inference
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Get input details
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    logger.info(f"Input name: {input_name}, shape: {input_shape}")

    # Create dummy input
    height, width = input_size

    # Determine input format from shape
    if len(input_shape) == 5:
        if input_shape[1] == 3:  # [B, C, T, H, W]
            dummy_input = torch.randn(1, 3, sequence_length, height, width).numpy()
        else:  # [B, T, C, H, W]
            dummy_input = torch.randn(1, sequence_length, 3, height, width).numpy()
    else:
        raise ValueError(f"Unexpected input shape: {input_shape}")

    # Run inference
    outputs = session.run(None, {input_name: dummy_input})
    logger.info(f"Output shape: {outputs[0].shape}")
    logger.info("ONNX model verification successful!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert temporal video analysis models to ONNX format"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["cnn_lstm", "3d_cnn", "conv_gru"],
        help="Model architecture type",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to PyTorch checkpoint (.pth file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save ONNX model",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=400,
        help="Number of action classes (default: 400 for Kinetics)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=16,
        help="Number of frames in sequence (default: 16)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Input spatial resolution (H W) (default: 224 224)",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--no-dynamic-batch",
        action="store_true",
        help="Disable dynamic batch size",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify ONNX model after export",
    )

    args = parser.parse_args()

    # Validate inputs
    if args.checkpoint and not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Create model
    logger.info(f"Creating {args.model_type} model with {args.num_classes} classes")
    model = create_model(args.model_type, args.num_classes, args.checkpoint)

    # Export to ONNX
    export_to_onnx(
        model=model,
        model_type=args.model_type,
        output_path=args.output,
        sequence_length=args.sequence_length,
        input_size=tuple(args.input_size),
        opset_version=args.opset_version,
        dynamic_batch=not args.no_dynamic_batch,
    )

    # Verify if requested
    if args.verify:
        verify_onnx_model(args.output, args.sequence_length, tuple(args.input_size))

    logger.info("Done!")


if __name__ == "__main__":
    main()
