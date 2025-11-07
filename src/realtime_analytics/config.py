"""
Configuration models and loading utilities for the realtime analytics stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml


class ConfigError(RuntimeError):
    """Raised when the supplied configuration is invalid."""


@dataclass(slots=True)
class FFmpegSimulatorConfig:
    """Configuration for spawning an ffmpeg process to emulate a camera stream."""

    enabled: bool = False
    input: str = ""
    loop: bool = True
    listen_host: Optional[str] = None
    log_level: str = "warning"
    video_codec: str = "libx264"
    audio_enabled: bool = False
    audio_codec: str = "aac"
    extra_args: List[str] = field(default_factory=list)

    def validate(self, stream: "StreamConfig" | None = None) -> None:
        if not self.enabled:
            return
        if not self.input:
            raise ConfigError("ffmpeg_simulator.input must not be empty when enabled")
        if stream is not None:
            if not stream.url:
                raise ConfigError(
                    f"Stream '{stream.name}' must define url when ffmpeg_simulator is enabled"
                )
            scheme = stream.url.split(":", 1)[0].lower()
            if scheme != "rtsp":
                raise ConfigError(
                    f"Stream '{stream.name}' uses scheme '{scheme}', "
                    "ffmpeg_simulator currently supports only RTSP outputs"
                )
        if self.video_codec and not isinstance(self.video_codec, str):
            raise ConfigError("ffmpeg_simulator.video_codec must be a string or empty")
        if self.audio_enabled and not self.audio_codec:
            raise ConfigError("ffmpeg_simulator.audio_codec must be set when audio_enabled is true")


@dataclass(slots=True)
class StreamConfig:
    """Configuration for a single RTSP/RTMP stream."""

    name: str
    url: str
    enabled: bool = True
    target_fps: Optional[float] = None
    batch_size: int = 1
    warmup_seconds: float = 2.0
    reconnect_backoff: float = 5.0
    max_retries: Optional[int] = None
    detector_id: Optional[str] = None
    roi_polygons: Optional[List[List[Tuple[int, int]]]] = None  # list of polygons
    motion_filter: bool = False
    motion_threshold: float = 0.02
    downsample_ratio: float = 1.0
    adaptive_fps: bool = False
    min_target_fps: float = 5.0
    idle_frame_tolerance: int = 60
    ffmpeg_simulator: Optional[FFmpegSimulatorConfig] = None

    def __post_init__(self) -> None:
        if isinstance(self.ffmpeg_simulator, dict):
            self.ffmpeg_simulator = FFmpegSimulatorConfig(**self.ffmpeg_simulator)

    def validate(self) -> None:
        if not self.name:
            raise ConfigError("Stream name must not be empty")
        if not self.url:
            raise ConfigError(f"Stream '{self.name}' must define a non-empty url")
        if self.batch_size < 1:
            raise ConfigError(f"Stream '{self.name}' batch_size must be >= 1")
        if self.target_fps is not None and self.target_fps <= 0:
            raise ConfigError(f"Stream '{self.name}' target_fps must be > 0 if provided")
        if self.warmup_seconds < 0:
            raise ConfigError(f"Stream '{self.name}' warmup_seconds must be >= 0")
        if self.reconnect_backoff < 0:
            raise ConfigError(f"Stream '{self.name}' reconnect_backoff must be >= 0")
        if self.max_retries is not None and self.max_retries < 0:
            raise ConfigError(f"Stream '{self.name}' max_retries must be >= 0")
        if self.motion_threshold < 0:
            raise ConfigError(f"Stream '{self.name}' motion_threshold must be >= 0")
        if not (0.1 <= self.downsample_ratio <= 1.0):
            raise ConfigError(f"Stream '{self.name}' downsample_ratio must be between 0.1 and 1.0")
        if self.adaptive_fps and (self.min_target_fps <= 0 or self.min_target_fps > (self.target_fps or 30)):
            raise ConfigError(
                f"Stream '{self.name}' min_target_fps must be > 0 and <= target_fps when adaptive_fps is enabled"
            )
        if self.ffmpeg_simulator and self.ffmpeg_simulator.enabled:
            self.ffmpeg_simulator.validate(self)


@dataclass(slots=True)
class DetectorConfig:
    """
    Detector configuration supporting multiple model types and backends.

    Supported model types:
    - yolov5: YOLOv5 object detection
    - yolov8: YOLOv8 object detection
    - resnet: ResNet classification
    - cnn_lstm: CNN-LSTM video sequence analysis
    - 3d_cnn: 3D CNN video analysis
    - conv_gru: Convolutional GRU video analysis
    - slow_fast: SlowFast networks for action recognition

    Supported backends:
    - ultralytics: Ultralytics YOLO (YOLOv5, YOLOv8)
    - tensorrt: NVIDIA TensorRT (YOLOv5, YOLOv8)
    - onnx/onnxruntime: ONNX Runtime (YOLOv5, YOLOv8, ResNet, temporal models)
    - openvino: Intel OpenVINO (YOLOv5, YOLOv8, ResNet, temporal models)
    - rknn: Rockchip RK3588 (YOLOv5, YOLOv8)
    """

    model_path: str = "yolov8n.pt"
    device: str = "auto"
    backend: str = "ultralytics"
    model_type: str = "yolov8"  # yolov5 | yolov8 | resnet | cnn_lstm | 3d_cnn | conv_gru | slow_fast
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    classes: Optional[List[int]] = None
    half: bool = False
    warmup: bool = True
    input_size: Optional[List[int]] = None  # H,W
    tensorrt_max_workspace_size: int = 1 << 30  # 1 GiB
    tensorrt_use_fp16: bool = False
    resnet_num_classes: int = 1000  # For ResNet classification
    resnet_top_k: int = 5  # Return top-K predictions for ResNet

    # Temporal model parameters
    sequence_length: int = 16  # Number of frames in sequence for temporal models
    sequence_stride: int = 1  # Stride between frames in sequence
    temporal_overlap: float = 0.5  # Overlap between consecutive sequences (0-1)
    temporal_pooling: str = "avg"  # Pooling method: avg | max | last
    action_classes: Optional[List[str]] = None  # Action class names for temporal models
    num_action_classes: int = 400  # Number of action classes (e.g., Kinetics-400)

    def validate(self) -> None:
        if not self.model_path:
            raise ConfigError("Detector model_path must not be empty")
        valid_backends = {"ultralytics", "tensorrt", "onnx", "onnxruntime", "openvino", "rknn", "rk3588"}
        if self.backend not in valid_backends:
            raise ConfigError(f"Detector backend must be one of {valid_backends}")
        valid_model_types = {"yolov5", "yolov8", "resnet", "cnn_lstm", "3d_cnn", "conv_gru", "slow_fast"}
        if self.model_type not in valid_model_types:
            raise ConfigError(f"Model type must be one of {valid_model_types}")
        if not (0.0 < self.confidence_threshold <= 1.0):
            raise ConfigError("confidence_threshold must be in (0, 1]")
        if not (0.0 < self.iou_threshold <= 1.0):
            raise ConfigError("iou_threshold must be in (0, 1]")
        if self.input_size and len(self.input_size) != 2:
            raise ConfigError("input_size must be [height, width]")
        if self.tensorrt_max_workspace_size <= 0:
            raise ConfigError("tensorrt_max_workspace_size must be > 0")
        if self.model_type == "resnet":
            if self.backend not in {"openvino", "onnx", "onnxruntime"}:
                raise ConfigError("ResNet models currently only supported with OpenVINO or ONNX Runtime backends")
            if self.resnet_num_classes <= 0:
                raise ConfigError("resnet_num_classes must be > 0")
            if self.resnet_top_k <= 0:
                raise ConfigError("resnet_top_k must be > 0")

        # Temporal model validation
        temporal_models = {"cnn_lstm", "3d_cnn", "conv_gru", "slow_fast"}
        if self.model_type in temporal_models:
            if self.backend not in {"onnx", "onnxruntime", "openvino"}:
                raise ConfigError(f"Temporal models currently only supported with ONNX Runtime or OpenVINO backends")
            if self.sequence_length <= 0:
                raise ConfigError("sequence_length must be > 0 for temporal models")
            if self.sequence_stride <= 0:
                raise ConfigError("sequence_stride must be > 0 for temporal models")
            if not (0.0 <= self.temporal_overlap < 1.0):
                raise ConfigError("temporal_overlap must be in [0, 1) for temporal models")
            if self.temporal_pooling not in {"avg", "max", "last"}:
                raise ConfigError("temporal_pooling must be one of: avg, max, last")
            if self.num_action_classes <= 0:
                raise ConfigError("num_action_classes must be > 0 for temporal models")


@dataclass(slots=True)
class TrackerConfig:
    """Multi-object tracker configuration (SORT/ByteTrack style)."""

    type: str = "byte_track"
    max_age: int = 30
    max_iou_distance: float = 0.7
    min_hits: int = 3

    def validate(self) -> None:
        if self.max_age < 1:
            raise ConfigError("Tracker max_age must be >= 1")
        if self.max_iou_distance <= 0:
            raise ConfigError("Tracker max_iou_distance must be > 0")
        if self.min_hits < 0:
            raise ConfigError("Tracker min_hits must be >= 0")


@dataclass(slots=True)
class KafkaSinkConfig:
    """Kafka publisher configuration."""

    enabled: bool = False
    bootstrap_servers: str = "localhost:9092"
    topic: str = "analytics"
    linger_ms: int = 10
    max_batch_size: int = 16384
    include_frames: bool = False
    frame_quality: int = 75

    def validate(self) -> None:
        if self.enabled and not self.topic:
            raise ConfigError("Kafka sink topic must not be empty when enabled")
        if self.linger_ms < 0:
            raise ConfigError("Kafka sink linger_ms must be >= 0")
        if self.max_batch_size <= 0:
            raise ConfigError("Kafka sink max_batch_size must be > 0")
        if not (1 <= self.frame_quality <= 100):
            raise ConfigError("Kafka sink frame_quality must be between 1 and 100")


@dataclass(slots=True)
class PrometheusConfig:
    """Prometheus endpoint configuration."""

    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 9000
    interval_seconds: float = 5.0

    def validate(self) -> None:
        if not (0 < self.port < 65536):
            raise ConfigError("Prometheus port must be between 1 and 65535")
        if self.interval_seconds <= 0:
            raise ConfigError("Prometheus interval_seconds must be > 0")


@dataclass(slots=True)
class PipelineConfig:
    """Top level configuration for the realtime analytics pipeline."""

    streams: List[StreamConfig] = field(default_factory=list)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    detectors: Dict[str, DetectorConfig] = field(default_factory=dict)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    kafka: KafkaSinkConfig = field(default_factory=KafkaSinkConfig)
    prometheus: PrometheusConfig = field(default_factory=PrometheusConfig)
    max_concurrent_streams: int = 32
    stats_interval_seconds: float = 15.0

    def validate(self) -> None:
        if not self.streams:
            raise ConfigError("At least one stream must be configured")
        if self.max_concurrent_streams < 1:
            raise ConfigError("max_concurrent_streams must be >= 1")
        if len(self.streams) > self.max_concurrent_streams:
            raise ConfigError(
                f"Configured {len(self.streams)} streams but max_concurrent_streams={self.max_concurrent_streams}"
            )
        if self.stats_interval_seconds <= 0:
            raise ConfigError("stats_interval_seconds must be > 0")
        available_detectors = set(self.detectors.keys())
        if not available_detectors:
            available_detectors.add("__default__")
        for stream in self.streams:
            if stream.detector_id and stream.detector_id not in self.detectors:
                raise ConfigError(
                    f"Stream '{stream.name}' references unknown detector_id='{stream.detector_id}'"
                )
        _validate_all(
            self.streams,
            self.detector,
            list(self.detectors.values()),
            self.tracker,
            self.kafka,
            self.prometheus,
        )


def _validate_all(*items: Iterable[object]) -> None:
    for item in items:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            for sub in item:
                _validate_all(sub)
        else:
            validator = getattr(item, "validate", None)
            if callable(validator):
                validator()


def _object_from_dict(cls, data: dict):
    allowed_keys = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs = {key: value for key, value in data.items() if key in allowed_keys}
    return cls(**kwargs)


def load_config(path: Path | str) -> PipelineConfig:
    """Load a pipeline configuration from a YAML file."""

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("Top level configuration must be a mapping/dictionary")

    stream_dicts = raw.get("streams")
    if not isinstance(stream_dicts, list):
        raise ConfigError("'streams' must be a list in the configuration")

    streams = [
        _object_from_dict(StreamConfig, stream_dict) for stream_dict in stream_dicts
    ]

    detector = _object_from_dict(DetectorConfig, raw.get("detector", {}))
    detectors_raw = raw.get("detectors", {}) or {}
    if not isinstance(detectors_raw, dict):
        raise ConfigError("'detectors' section must be a mapping of id -> config")
    detectors = {
        key: _object_from_dict(DetectorConfig, value or {})
        for key, value in detectors_raw.items()
    }
    tracker = _object_from_dict(TrackerConfig, raw.get("tracker", {}))
    kafka = _object_from_dict(KafkaSinkConfig, raw.get("kafka", {}))
    prometheus = _object_from_dict(PrometheusConfig, raw.get("prometheus", {}))
    pipeline = PipelineConfig(
        streams=streams,
        detector=detector,
        detectors=detectors,
        tracker=tracker,
        kafka=kafka,
        prometheus=prometheus,
        max_concurrent_streams=raw.get("max_concurrent_streams", 32),
        stats_interval_seconds=raw.get("stats_interval_seconds", 15.0),
    )
    pipeline.validate()
    return pipeline
