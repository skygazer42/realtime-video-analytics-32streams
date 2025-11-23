#!/usr/bin/env python3
"""
Test script for temporal video analysis detectors.

This script helps validate temporal model integration by:
1. Loading a temporal detector configuration
2. Processing a test video or image sequence
3. Outputting detection results and performance metrics

Usage:
    # Test with video file
    python test_temporal_detector.py \
        --model-type cnn_lstm \
        --model-path model.onnx \
        --backend onnxruntime \
        --video test_video.mp4

    # Test with image sequence
    python test_temporal_detector.py \
        --model-type 3d_cnn \
        --model-path model.onnx \
        --backend openvino \
        --image-dir frames/ \
        --image-pattern "frame_%04d.jpg"
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path to import realtime_analytics
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from realtime_analytics.config import DetectorConfig
from realtime_analytics.detector import create_detector
from realtime_analytics.video_stream import FramePacket, StreamConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DummyStream:
    """Dummy stream object for testing."""

    def __init__(self, name="test_stream"):
        self.name = name


def load_video_frames(video_path: str, max_frames: int | None = None) -> list[np.ndarray]:
    """
    Load frames from a video file.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to load (None for all)

    Returns:
        List of BGR frames
    """
    logger.info(f"Loading video from {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        frame_count += 1

        if max_frames and frame_count >= max_frames:
            break

    cap.release()
    logger.info(f"Loaded {len(frames)} frames from video")
    return frames


def load_image_sequence(
    image_dir: str, pattern: str, max_images: int | None = None
) -> list[np.ndarray]:
    """
    Load frames from an image sequence.

    Args:
        image_dir: Directory containing images
        pattern: Pattern for image filenames (e.g., "frame_%04d.jpg")
        max_images: Maximum number of images to load

    Returns:
        List of BGR frames
    """
    logger.info(f"Loading image sequence from {image_dir} with pattern {pattern}")
    frames = []
    idx = 0

    while True:
        if max_images and idx >= max_images:
            break

        image_path = Path(image_dir) / (pattern % idx)
        if not image_path.exists():
            if idx == 0:
                raise FileNotFoundError(f"No images found matching pattern: {pattern}")
            break

        frame = cv2.imread(str(image_path))
        if frame is None:
            logger.warning(f"Failed to load image: {image_path}")
            break

        frames.append(frame)
        idx += 1

    logger.info(f"Loaded {len(frames)} frames from image sequence")
    return frames


def create_frame_packets(frames: list[np.ndarray], stream_name: str = "test") -> list[FramePacket]:
    """
    Convert frames to FramePacket objects.

    Args:
        frames: List of BGR frames
        stream_name: Stream name for packets

    Returns:
        List of FramePacket objects
    """
    stream = DummyStream(stream_name)
    packets = []

    for i, frame in enumerate(frames):
        packet = FramePacket(
            stream=stream,
            frame_id=i,
            frame=frame,
            capture_time=time.time(),
        )
        packets.append(packet)

    return packets


def test_detector(
    detector_config: DetectorConfig,
    frames: list[np.ndarray],
    warmup_runs: int = 3,
) -> None:
    """
    Test temporal detector with frame sequence.

    Args:
        detector_config: Detector configuration
        frames: List of BGR frames
        warmup_runs: Number of warmup runs before timing
    """
    logger.info("Creating temporal detector...")
    detector = create_detector(detector_config)

    # Convert frames to packets
    packets = create_frame_packets(frames)

    logger.info(f"Testing detector with {len(packets)} frames")
    logger.info(
        f"Sequence length: {detector_config.sequence_length}, "
        f"stride: {detector_config.sequence_stride}"
    )

    # Calculate expected number of detections
    required_frames = detector_config.sequence_length * detector_config.sequence_stride
    logger.info(f"Frames required for sequence: {required_frames}")

    if len(packets) < required_frames:
        logger.warning(
            f"Not enough frames for a complete sequence! "
            f"Need {required_frames}, have {len(packets)}"
        )

    # Warmup
    if warmup_runs > 0 and len(packets) >= required_frames:
        logger.info(f"Performing {warmup_runs} warmup runs...")
        for i in range(warmup_runs):
            for packet in packets[:required_frames]:
                _ = detector.predict(packet)

    # Test inference
    all_detections = []
    inference_times = []

    logger.info("Running inference...")
    for i, packet in enumerate(packets):
        start_time = time.time()
        detections = detector.predict(packet)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        if detections:
            all_detections.extend(detections)
            inference_times.append(inference_time)
            logger.info(
                f"Frame {i}: {len(detections)} detection(s) in {inference_time:.2f}ms"
            )

            # Print detection details
            for det in detections:
                logger.info(
                    f"  - Class {det.class_id}, confidence: {det.confidence:.3f}"
                )
                if hasattr(det, "action_label") and det.action_label:
                    logger.info(f"    Action: {det.action_label}")
                if hasattr(det, "temporal_score"):
                    logger.info(f"    Temporal score: {det.temporal_score:.3f}")
                if hasattr(det, "sequence_start_frame"):
                    logger.info(
                        f"    Sequence: frames {det.sequence_start_frame}-{det.sequence_end_frame}"
                    )

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total frames processed: {len(packets)}")
    logger.info(f"Total detections: {len(all_detections)}")

    if inference_times:
        avg_time = np.mean(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        fps = 1000.0 / avg_time if avg_time > 0 else 0

        logger.info(f"Inference time (avg): {avg_time:.2f}ms")
        logger.info(f"Inference time (min): {min_time:.2f}ms")
        logger.info(f"Inference time (max): {max_time:.2f}ms")
        logger.info(f"Effective FPS: {fps:.2f}")
    else:
        logger.info("No detections generated (may need more frames)")

    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test temporal video analysis detectors")
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["cnn_lstm", "3d_cnn", "conv_gru", "slow_fast"],
        help="Model type",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model file (.onnx, .xml, etc.)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="onnxruntime",
        choices=["onnxruntime", "openvino"],
        help="Inference backend",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu, cuda, gpu, auto)",
    )

    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", type=str, help="Path to video file")
    input_group.add_argument("--image-dir", type=str, help="Directory with image sequence")

    parser.add_argument(
        "--image-pattern",
        type=str,
        default="frame_%04d.jpg",
        help="Pattern for image filenames (requires --image-dir)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process",
    )

    # Model configuration
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=16,
        help="Number of frames in sequence",
    )
    parser.add_argument(
        "--sequence-stride",
        type=int,
        default=1,
        help="Stride between frames",
    )
    parser.add_argument(
        "--temporal-overlap",
        type=float,
        default=0.5,
        help="Overlap between sequences (0-1)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=None,
        help="Input spatial resolution (H W)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=400,
        help="Number of action classes",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.3,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 precision",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup runs",
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    # Load frames
    if args.video:
        frames = load_video_frames(args.video, args.max_frames)
    else:
        frames = load_image_sequence(args.image_dir, args.image_pattern, args.max_frames)

    if not frames:
        raise ValueError("No frames loaded!")

    # Create detector config
    config = DetectorConfig(
        model_type=args.model_type,
        model_path=args.model_path,
        backend=args.backend,
        device=args.device,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
        temporal_overlap=args.temporal_overlap,
        num_action_classes=args.num_classes,
        confidence_threshold=args.confidence_threshold,
        half=args.half,
        warmup=True,
    )

    if args.input_size:
        config.input_size = args.input_size

    # Test detector
    test_detector(config, frames, warmup_runs=args.warmup)


if __name__ == "__main__":
    main()
