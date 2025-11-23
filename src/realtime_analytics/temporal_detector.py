"""
Temporal video analysis detectors supporting sequence-based models.

Supported temporal model types:
- CNN-LSTM: Convolutional Neural Network + Long Short-Term Memory for action recognition
- 3D CNN: 3D Convolutional Neural Network for spatiotemporal feature learning
- ConvGRU: Convolutional Gated Recurrent Unit for video analysis
- SlowFast: SlowFast networks for action recognition with dual pathways

Supported backends:
- ONNX Runtime: Cross-platform inference with GPU/CPU support
- OpenVINO: Intel hardware optimization

These detectors process sequences of video frames to recognize actions, events, or
temporal patterns that cannot be detected from single frames alone.
"""

from __future__ import annotations

import abc
import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import numpy as np

from .config import DetectorConfig
from .detector import Detection
from .video_stream import FramePacket

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TemporalDetection(Detection):
    """
    Temporal detection result from sequence-based model inference.

    Extends Detection with temporal information such as action labels,
    temporal score, and sequence metadata.
    """

    action_label: Optional[str] = None  # Human-readable action name
    temporal_score: float = 0.0  # Confidence score for temporal prediction
    sequence_start_frame: int = 0  # First frame in the analyzed sequence
    sequence_end_frame: int = 0  # Last frame in the analyzed sequence


class BaseTemporalDetector(abc.ABC):
    """
    Abstract base class for temporal video analysis detectors.

    Manages frame buffering, sequence extraction, and temporal inference.
    Subclasses implement specific temporal model architectures.
    """

    def __init__(self, config: DetectorConfig):
        self.config = config

        # Frame buffers for each stream (keyed by stream name)
        self._frame_buffers: Dict[str, Deque[FramePacket]] = {}

        # Calculate step size between sequences based on overlap
        # overlap = 0.5 means 50% overlap, so step = sequence_length * (1 - overlap)
        self.sequence_step = max(
            1, int(self.config.sequence_length * (1.0 - self.config.temporal_overlap))
        )

        LOGGER.info(
            "Temporal detector initialized: sequence_length=%d, stride=%d, step=%d, overlap=%.2f",
            self.config.sequence_length,
            self.config.sequence_stride,
            self.sequence_step,
            self.config.temporal_overlap,
        )

    def predict(self, packet: FramePacket) -> List[Detection]:
        """
        Process a frame packet and return detections when sequence is ready.

        This method manages frame buffering and calls _predict_sequence when
        a complete sequence is available.
        """
        stream_name = packet.stream.name

        # Initialize buffer for this stream if needed
        if stream_name not in self._frame_buffers:
            self._frame_buffers[stream_name] = deque(
                maxlen=self.config.sequence_length * self.config.sequence_stride
            )

        # Add frame to buffer
        buffer = self._frame_buffers[stream_name]
        buffer.append(packet)

        # Check if we have enough frames for a sequence
        required_frames = self.config.sequence_length * self.config.sequence_stride
        if len(buffer) < required_frames:
            # Not enough frames yet, return empty detections
            return []

        # Extract sequence with stride
        sequence = [buffer[i * self.config.sequence_stride] for i in range(self.config.sequence_length)]

        # Perform temporal inference
        detections = self._predict_sequence(sequence)

        # Clear buffer based on step size to prepare for next sequence
        # Keep (required_frames - step) frames for overlap
        frames_to_keep = max(0, required_frames - self.sequence_step)
        if frames_to_keep > 0:
            # Keep the last frames_to_keep frames
            new_buffer = deque(list(buffer)[-frames_to_keep:], maxlen=buffer.maxlen)
            self._frame_buffers[stream_name] = new_buffer
        else:
            # No overlap, clear buffer
            buffer.clear()

        return detections

    @abc.abstractmethod
    def _predict_sequence(self, sequence: List[FramePacket]) -> List[Detection]:
        """
        Perform inference on a sequence of frames.

        Args:
            sequence: List of FramePacket objects forming a temporal sequence

        Returns:
            List of Detection objects (may include TemporalDetection instances)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _preprocess_sequence(self, sequence: List[FramePacket]) -> np.ndarray:
        """
        Preprocess a sequence of frames into model input format.

        Args:
            sequence: List of FramePacket objects

        Returns:
            Preprocessed numpy array ready for model inference
            Shape depends on model type (e.g., [B, T, C, H, W] or [B, C, T, H, W])
        """
        raise NotImplementedError


class CNNLSTMDetector(BaseTemporalDetector):
    """
    CNN-LSTM detector for action recognition and video analysis.

    Architecture:
    - CNN: Extracts spatial features from each frame
    - LSTM: Models temporal dependencies across frames
    - Fully Connected: Produces action classification

    Input format: [batch, time, channels, height, width]
    Output format: [batch, num_classes] (action probabilities)
    """

    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        self._session = None
        self._load_model()

    def _load_model(self) -> None:
        """Load CNN-LSTM model using configured backend."""
        if self.config.backend in ("onnx", "onnxruntime"):
            self._load_onnx_model()
        elif self.config.backend == "openvino":
            self._load_openvino_model()
        else:
            raise ValueError(
                f"CNN-LSTM models not supported with backend '{self.config.backend}'"
            )

    def _load_onnx_model(self) -> None:
        """Load CNN-LSTM model using ONNX Runtime."""
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "ONNX Runtime backend selected but not installed. "
                "Install with `pip install onnxruntime` or `pip install onnxruntime-gpu`."
            ) from exc

        self.ort = ort

        # Setup providers
        providers = []
        provider_options = []

        if self.config.device == "cuda" or (
            isinstance(self.config.device, str) and self.config.device.startswith("cuda")
        ):
            if "CUDAExecutionProvider" in ort.get_available_providers():
                cuda_options = {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 4 * 1024 * 1024 * 1024,  # 4GB for temporal models
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                }
                providers.append("CUDAExecutionProvider")
                provider_options.append(cuda_options)
                LOGGER.info("Using CUDA Execution Provider for CNN-LSTM model")
            else:
                LOGGER.warning("CUDA requested but not available, falling back to CPU")
                providers.append("CPUExecutionProvider")
                provider_options.append({})
        else:
            providers.append("CPUExecutionProvider")
            provider_options.append({})

        LOGGER.info(
            "Loading CNN-LSTM ONNX model '%s' with providers: %s",
            self.config.model_path,
            providers,
        )

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.enable_mem_pattern = True
        session_options.enable_cpu_mem_arena = True
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self._session = ort.InferenceSession(
            self.config.model_path,
            sess_options=session_options,
            providers=providers,
            provider_options=provider_options,
        )

        self.input_name = self._session.get_inputs()[0].name
        self.output_name = self._session.get_outputs()[0].name

        # Determine input shape
        input_shape = self._session.get_inputs()[0].shape
        if self.config.input_size:
            self.input_hw = (int(self.config.input_size[0]), int(self.config.input_size[1]))
        elif len(input_shape) >= 4:
            # Common formats: [B, T, C, H, W] or [B, T, H, W, C]
            # Assume BTCHW format for CNN-LSTM
            self.input_hw = (int(input_shape[-2]), int(input_shape[-1]))
        else:
            LOGGER.warning("Could not determine input size from model, defaulting to 224x224")
            self.input_hw = (224, 224)

        # Warmup
        if self.config.warmup:
            LOGGER.debug("Running CNN-LSTM warmup")
            dummy = self._create_dummy_sequence()
            self._session.run([self.output_name], {self.input_name: dummy})

        LOGGER.info("CNN-LSTM ONNX detector initialized")

    def _load_openvino_model(self) -> None:
        """Load CNN-LSTM model using OpenVINO."""
        try:
            from openvino.runtime import Core  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "OpenVINO backend selected but not installed. "
                "Install with `pip install openvino`."
            ) from exc

        self.ov_core = Core()

        device_map = {
            "cpu": "CPU",
            "gpu": "GPU",
            "cuda": "GPU",
            "auto": "AUTO",
            "npu": "NPU",
        }
        ov_device = device_map.get(str(self.config.device).lower(), "CPU")

        LOGGER.info(
            "Loading CNN-LSTM OpenVINO model '%s' on device '%s'",
            self.config.model_path,
            ov_device,
        )

        model = self.ov_core.read_model(model=self.config.model_path)

        compile_config = {}
        if ov_device == "CPU":
            compile_config["PERFORMANCE_HINT"] = "LATENCY"
            compile_config["CPU_THREADS_NUM"] = "0"
        elif ov_device in ("GPU", "AUTO"):
            compile_config["PERFORMANCE_HINT"] = "THROUGHPUT"

        self._compiled_model = self.ov_core.compile_model(
            model=model, device_name=ov_device, config=compile_config
        )

        self.input_layer = self._compiled_model.input(0)
        self.output_layer = self._compiled_model.output(0)

        # Determine input shape
        input_shape = self.input_layer.shape
        if self.config.input_size:
            self.input_hw = (int(self.config.input_size[0]), int(self.config.input_size[1]))
        elif len(input_shape) >= 4:
            self.input_hw = (int(input_shape[-2]), int(input_shape[-1]))
        else:
            self.input_hw = (224, 224)

        self._infer_request = self._compiled_model.create_infer_request()

        # Warmup
        if self.config.warmup:
            LOGGER.debug("Running CNN-LSTM OpenVINO warmup")
            dummy = self._create_dummy_sequence()
            self._infer_request.infer({0: dummy})

        LOGGER.info("CNN-LSTM OpenVINO detector initialized")

    def _create_dummy_sequence(self) -> np.ndarray:
        """Create a dummy input sequence for warmup."""
        dtype = np.float16 if self.config.half else np.float32
        # Shape: [B, T, C, H, W]
        return np.zeros(
            (1, self.config.sequence_length, 3, *self.input_hw),
            dtype=dtype,
        )

    def _preprocess_sequence(self, sequence: List[FramePacket]) -> np.ndarray:
        """
        Preprocess sequence of frames for CNN-LSTM.

        Output shape: [1, T, C, H, W] where T is sequence_length
        """
        import cv2

        preprocessed_frames = []

        for packet in sequence:
            frame = packet.frame

            # Resize frame
            resized = cv2.resize(frame, (self.input_hw[1], self.input_hw[0]))

            # Convert BGR to RGB
            image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # Normalize (ImageNet mean and std)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

            image = image.astype(np.float32) / 255.0
            image = (image - mean) / std

            # Transpose to CHW format
            tensor = np.transpose(image, (2, 0, 1))

            preprocessed_frames.append(tensor)

        # Stack frames into sequence: [T, C, H, W]
        sequence_tensor = np.stack(preprocessed_frames, axis=0)

        # Add batch dimension: [1, T, C, H, W]
        sequence_tensor = np.expand_dims(sequence_tensor, axis=0)

        # Convert to appropriate dtype
        if self.config.half:
            sequence_tensor = sequence_tensor.astype(np.float16)
        else:
            sequence_tensor = sequence_tensor.astype(np.float32)

        return np.ascontiguousarray(sequence_tensor)

    def _predict_sequence(self, sequence: List[FramePacket]) -> List[Detection]:
        """Perform CNN-LSTM inference on a frame sequence."""
        if not sequence:
            return []

        # Preprocess sequence
        input_tensor = self._preprocess_sequence(sequence)

        # Run inference
        if self.config.backend in ("onnx", "onnxruntime"):
            output = self._session.run([self.output_name], {self.input_name: input_tensor})[0]
        elif self.config.backend == "openvino":
            output = self._infer_request.infer({0: input_tensor})[self.output_layer]
        else:
            raise RuntimeError(f"Unsupported backend: {self.config.backend}")

        # Postprocess predictions
        if output.ndim > 1:
            output = output.flatten()

        # Get top predictions
        top_k = min(5, len(output))
        top_k_indices = np.argsort(output)[-top_k:][::-1]
        top_k_probs = output[top_k_indices]

        # Create temporal detections
        detections: List[Detection] = []
        first_packet = sequence[0]
        last_packet = sequence[-1]
        h, w = first_packet.frame.shape[:2]

        for class_id, confidence in zip(top_k_indices, top_k_probs):
            if confidence >= self.config.confidence_threshold:
                # Get action label if available
                action_label = None
                if self.config.action_classes and class_id < len(self.config.action_classes):
                    action_label = self.config.action_classes[class_id]

                detection = TemporalDetection(
                    stream_name=first_packet.stream.name,
                    frame_id=last_packet.frame_id,  # Use last frame ID
                    class_id=int(class_id),
                    confidence=float(confidence),
                    bbox_xyxy=(0.0, 0.0, float(w), float(h)),  # Full frame
                    action_label=action_label,
                    temporal_score=float(confidence),
                    sequence_start_frame=first_packet.frame_id,
                    sequence_end_frame=last_packet.frame_id,
                )
                detections.append(detection)

        return detections


class CNN3DDetector(BaseTemporalDetector):
    """
    3D CNN detector for spatiotemporal video analysis.

    Architecture:
    - 3D Convolutions: Process both spatial and temporal dimensions simultaneously
    - Pooling: 3D max/avg pooling for feature reduction
    - Fully Connected: Action classification

    Input format: [batch, channels, time, height, width] (NCTHW)
    Output format: [batch, num_classes] (action probabilities)
    """

    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        self._session = None
        self._load_model()

    def _load_model(self) -> None:
        """Load 3D CNN model using configured backend."""
        if self.config.backend in ("onnx", "onnxruntime"):
            self._load_onnx_model()
        elif self.config.backend == "openvino":
            self._load_openvino_model()
        else:
            raise ValueError(f"3D CNN models not supported with backend '{self.config.backend}'")

    def _load_onnx_model(self) -> None:
        """Load 3D CNN model using ONNX Runtime."""
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "ONNX Runtime not installed. Install with `pip install onnxruntime`."
            ) from exc

        self.ort = ort
        providers = []
        provider_options = []

        if self.config.device == "cuda" or (
            isinstance(self.config.device, str) and self.config.device.startswith("cuda")
        ):
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.append("CUDAExecutionProvider")
                provider_options.append({
                    "device_id": 0,
                    "gpu_mem_limit": 4 * 1024 * 1024 * 1024,
                })
            else:
                providers.append("CPUExecutionProvider")
                provider_options.append({})
        else:
            providers.append("CPUExecutionProvider")
            provider_options.append({})

        LOGGER.info("Loading 3D CNN ONNX model '%s'", self.config.model_path)

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            self.config.model_path,
            sess_options=session_options,
            providers=providers,
            provider_options=provider_options,
        )

        self.input_name = self._session.get_inputs()[0].name
        self.output_name = self._session.get_outputs()[0].name

        # Determine input shape
        input_shape = self._session.get_inputs()[0].shape
        if self.config.input_size:
            self.input_hw = (int(self.config.input_size[0]), int(self.config.input_size[1]))
        elif len(input_shape) >= 4:
            # 3D CNN format: [B, C, T, H, W]
            self.input_hw = (int(input_shape[-2]), int(input_shape[-1]))
        else:
            self.input_hw = (112, 112)  # Common 3D CNN size

        if self.config.warmup:
            dummy = np.zeros(
                (1, 3, self.config.sequence_length, *self.input_hw),
                dtype=np.float32 if not self.config.half else np.float16,
            )
            self._session.run([self.output_name], {self.input_name: dummy})

        LOGGER.info("3D CNN ONNX detector initialized")

    def _load_openvino_model(self) -> None:
        """Load 3D CNN model using OpenVINO."""
        try:
            from openvino.runtime import Core  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("OpenVINO not installed. Install with `pip install openvino`.") from exc

        self.ov_core = Core()
        device_map = {"cpu": "CPU", "gpu": "GPU", "cuda": "GPU", "auto": "AUTO"}
        ov_device = device_map.get(str(self.config.device).lower(), "CPU")

        LOGGER.info("Loading 3D CNN OpenVINO model '%s' on '%s'", self.config.model_path, ov_device)

        model = self.ov_core.read_model(model=self.config.model_path)
        self._compiled_model = self.ov_core.compile_model(model=model, device_name=ov_device)

        self.input_layer = self._compiled_model.input(0)
        self.output_layer = self._compiled_model.output(0)

        input_shape = self.input_layer.shape
        if self.config.input_size:
            self.input_hw = (int(self.config.input_size[0]), int(self.config.input_size[1]))
        elif len(input_shape) >= 4:
            self.input_hw = (int(input_shape[-2]), int(input_shape[-1]))
        else:
            self.input_hw = (112, 112)

        self._infer_request = self._compiled_model.create_infer_request()

        if self.config.warmup:
            dummy = np.zeros((1, 3, self.config.sequence_length, *self.input_hw), dtype=np.float32)
            self._infer_request.infer({0: dummy})

        LOGGER.info("3D CNN OpenVINO detector initialized")

    def _preprocess_sequence(self, sequence: List[FramePacket]) -> np.ndarray:
        """
        Preprocess sequence for 3D CNN.

        Output shape: [1, C, T, H, W] (NCTHW format)
        """
        import cv2

        preprocessed_frames = []

        for packet in sequence:
            frame = packet.frame
            resized = cv2.resize(frame, (self.input_hw[1], self.input_hw[0]))
            image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # Normalize
            image = image.astype(np.float32) / 255.0
            mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
            std = np.array([0.225, 0.225, 0.225], dtype=np.float32)
            image = (image - mean) / std

            # CHW format
            tensor = np.transpose(image, (2, 0, 1))
            preprocessed_frames.append(tensor)

        # Stack: [C, T, H, W] -> transpose to [T, C, H, W] -> stack channels properly
        # Actually we want [C, T, H, W] so we stack on time axis
        # preprocessed_frames is list of [C, H, W], stack on new axis to get [T, C, H, W]
        sequence_tensor = np.stack(preprocessed_frames, axis=0)  # [T, C, H, W]

        # Transpose to [C, T, H, W]
        sequence_tensor = np.transpose(sequence_tensor, (1, 0, 2, 3))

        # Add batch dimension: [1, C, T, H, W]
        sequence_tensor = np.expand_dims(sequence_tensor, axis=0)

        if self.config.half:
            sequence_tensor = sequence_tensor.astype(np.float16)

        return np.ascontiguousarray(sequence_tensor)

    def _predict_sequence(self, sequence: List[FramePacket]) -> List[Detection]:
        """Perform 3D CNN inference on frame sequence."""
        if not sequence:
            return []

        input_tensor = self._preprocess_sequence(sequence)

        # Run inference
        if self.config.backend in ("onnx", "onnxruntime"):
            output = self._session.run([self.output_name], {self.input_name: input_tensor})[0]
        elif self.config.backend == "openvino":
            output = self._infer_request.infer({0: input_tensor})[self.output_layer]
        else:
            raise RuntimeError(f"Unsupported backend: {self.config.backend}")

        if output.ndim > 1:
            output = output.flatten()

        top_k = min(5, len(output))
        top_k_indices = np.argsort(output)[-top_k:][::-1]
        top_k_probs = output[top_k_indices]

        detections: List[Detection] = []
        first_packet = sequence[0]
        last_packet = sequence[-1]
        h, w = first_packet.frame.shape[:2]

        for class_id, confidence in zip(top_k_indices, top_k_probs):
            if confidence >= self.config.confidence_threshold:
                action_label = None
                if self.config.action_classes and class_id < len(self.config.action_classes):
                    action_label = self.config.action_classes[class_id]

                detection = TemporalDetection(
                    stream_name=first_packet.stream.name,
                    frame_id=last_packet.frame_id,
                    class_id=int(class_id),
                    confidence=float(confidence),
                    bbox_xyxy=(0.0, 0.0, float(w), float(h)),
                    action_label=action_label,
                    temporal_score=float(confidence),
                    sequence_start_frame=first_packet.frame_id,
                    sequence_end_frame=last_packet.frame_id,
                )
                detections.append(detection)

        return detections


class ConvGRUDetector(BaseTemporalDetector):
    """
    Convolutional GRU detector for efficient video analysis.

    Architecture:
    - Convolutional layers: Extract spatial features
    - GRU (Gated Recurrent Unit): Model temporal dependencies (more efficient than LSTM)
    - Fully Connected: Action classification

    ConvGRU is similar to CNN-LSTM but uses GRU which has fewer parameters and
    often trains faster while maintaining competitive performance.
    """

    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        self._session = None
        self._load_model()

    def _load_model(self) -> None:
        """Load ConvGRU model using configured backend."""
        if self.config.backend in ("onnx", "onnxruntime"):
            self._load_onnx_model()
        elif self.config.backend == "openvino":
            self._load_openvino_model()
        else:
            raise ValueError(f"ConvGRU models not supported with backend '{self.config.backend}'")

    def _load_onnx_model(self) -> None:
        """Load ConvGRU model using ONNX Runtime."""
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("ONNX Runtime not installed.") from exc

        self.ort = ort
        providers = ["CUDAExecutionProvider"] if "cuda" in str(self.config.device).lower() else []
        providers.append("CPUExecutionProvider")

        LOGGER.info("Loading ConvGRU ONNX model '%s'", self.config.model_path)

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            self.config.model_path, sess_options=session_options, providers=providers
        )

        self.input_name = self._session.get_inputs()[0].name
        self.output_name = self._session.get_outputs()[0].name

        input_shape = self._session.get_inputs()[0].shape
        if self.config.input_size:
            self.input_hw = (int(self.config.input_size[0]), int(self.config.input_size[1]))
        elif len(input_shape) >= 4:
            self.input_hw = (int(input_shape[-2]), int(input_shape[-1]))
        else:
            self.input_hw = (224, 224)

        LOGGER.info("ConvGRU ONNX detector initialized")

    def _load_openvino_model(self) -> None:
        """Load ConvGRU model using OpenVINO."""
        try:
            from openvino.runtime import Core  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("OpenVINO not installed.") from exc

        self.ov_core = Core()
        ov_device = "GPU" if "cuda" in str(self.config.device).lower() else "CPU"

        LOGGER.info("Loading ConvGRU OpenVINO model '%s'", self.config.model_path)

        model = self.ov_core.read_model(model=self.config.model_path)
        self._compiled_model = self.ov_core.compile_model(model=model, device_name=ov_device)

        self.input_layer = self._compiled_model.input(0)
        self.output_layer = self._compiled_model.output(0)

        input_shape = self.input_layer.shape
        if self.config.input_size:
            self.input_hw = (int(self.config.input_size[0]), int(self.config.input_size[1]))
        elif len(input_shape) >= 4:
            self.input_hw = (int(input_shape[-2]), int(input_shape[-1]))
        else:
            self.input_hw = (224, 224)

        self._infer_request = self._compiled_model.create_infer_request()
        LOGGER.info("ConvGRU OpenVINO detector initialized")

    def _preprocess_sequence(self, sequence: List[FramePacket]) -> np.ndarray:
        """Preprocess sequence for ConvGRU (same format as CNN-LSTM)."""
        import cv2

        preprocessed_frames = []
        for packet in sequence:
            resized = cv2.resize(packet.frame, (self.input_hw[1], self.input_hw[0]))
            image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std
            tensor = np.transpose(image, (2, 0, 1))
            preprocessed_frames.append(tensor)

        sequence_tensor = np.stack(preprocessed_frames, axis=0)
        sequence_tensor = np.expand_dims(sequence_tensor, axis=0)

        if self.config.half:
            sequence_tensor = sequence_tensor.astype(np.float16)

        return np.ascontiguousarray(sequence_tensor)

    def _predict_sequence(self, sequence: List[FramePacket]) -> List[Detection]:
        """Perform ConvGRU inference on frame sequence."""
        if not sequence:
            return []

        input_tensor = self._preprocess_sequence(sequence)

        if self.config.backend in ("onnx", "onnxruntime"):
            output = self._session.run([self.output_name], {self.input_name: input_tensor})[0]
        elif self.config.backend == "openvino":
            output = self._infer_request.infer({0: input_tensor})[self.output_layer]
        else:
            raise RuntimeError(f"Unsupported backend: {self.config.backend}")

        if output.ndim > 1:
            output = output.flatten()

        top_k = min(5, len(output))
        top_k_indices = np.argsort(output)[-top_k:][::-1]
        top_k_probs = output[top_k_indices]

        detections: List[Detection] = []
        first_packet, last_packet = sequence[0], sequence[-1]
        h, w = first_packet.frame.shape[:2]

        for class_id, confidence in zip(top_k_indices, top_k_probs):
            if confidence >= self.config.confidence_threshold:
                action_label = None
                if self.config.action_classes and class_id < len(self.config.action_classes):
                    action_label = self.config.action_classes[class_id]

                detection = TemporalDetection(
                    stream_name=first_packet.stream.name,
                    frame_id=last_packet.frame_id,
                    class_id=int(class_id),
                    confidence=float(confidence),
                    bbox_xyxy=(0.0, 0.0, float(w), float(h)),
                    action_label=action_label,
                    temporal_score=float(confidence),
                    sequence_start_frame=first_packet.frame_id,
                    sequence_end_frame=last_packet.frame_id,
                )
                detections.append(detection)

        return detections
