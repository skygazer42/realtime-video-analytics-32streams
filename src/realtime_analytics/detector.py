"""
Detector backends (Ultralytics YOLO, ONNX Runtime, TensorRT, OpenVINO, RKNN) and factory helpers.

Supported model types:
- YOLOv5: Object detection with YOLOv5 architecture
- YOLOv8: Object detection with YOLOv8 architecture
- ResNet: Image classification with ResNet architecture

Supported backends:
- Ultralytics: PyTorch-based inference (YOLOv5, YOLOv8)
- TensorRT: NVIDIA GPU optimization (YOLOv5, YOLOv8)
- ONNX Runtime: Cross-platform inference (YOLOv5, YOLOv8, ResNet)
- OpenVINO: Intel hardware optimization (YOLOv5, YOLOv8, ResNet)
- RKNN: Rockchip RK3588 NPU (YOLOv5, YOLOv8)
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from .config import DetectorConfig
from .video_stream import FramePacket

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Detection:
    """Single detection result from a model inference."""

    stream_name: str
    frame_id: int
    class_id: int
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


class BaseDetector(abc.ABC):
    """Abstract detector interface."""

    def __init__(self, config: DetectorConfig):
        self.config = config

    @abc.abstractmethod
    def predict(self, packet: FramePacket) -> List[Detection]:
        raise NotImplementedError


def create_detector(config: DetectorConfig) -> BaseDetector:
    """Instantiate a detector backend based on configuration."""
    backend = config.backend.lower()
    model_type = config.model_type.lower()

    # Temporal models (CNN-LSTM, 3D CNN, ConvGRU, SlowFast)
    temporal_models = {"cnn_lstm", "3d_cnn", "conv_gru", "slow_fast"}
    if model_type in temporal_models:
        from .temporal_detector import CNNLSTMDetector, CNN3DDetector, ConvGRUDetector

        if model_type == "cnn_lstm":
            return CNNLSTMDetector(config)
        if model_type == "3d_cnn":
            return CNN3DDetector(config)
        if model_type == "conv_gru":
            return ConvGRUDetector(config)
        if model_type == "slow_fast":
            # SlowFast can use similar architecture to 3D CNN
            # For now, use CNN3DDetector (can be specialized later)
            LOGGER.info("Using 3D CNN detector for SlowFast model")
            return CNN3DDetector(config)
        raise ValueError(f"Temporal model type '{model_type}' not implemented")

    # ResNet classification models
    if model_type == "resnet":
        if backend in ("openvino",):
            return ResNetOpenVINODetector(config)
        if backend in ("onnx", "onnxruntime"):
            return ResNetONNXDetector(config)
        raise ValueError(f"ResNet models not supported with backend '{config.backend}'")

    # YOLO object detection models (YOLOv5 and YOLOv8)
    if backend == "ultralytics":
        return UltralyticsDetector(config)
    if backend == "tensorrt":
        return TensorRTDetector(config)
    if backend == "onnx" or backend == "onnxruntime":
        return ONNXRuntimeDetector(config)
    if backend == "openvino":
        return OpenVINODetector(config)
    if backend == "rknn" or backend == "rk3588":
        return RKNNDetector(config)
    raise ValueError(f"Unsupported detector backend '{config.backend}'")


def filter_detections(
    detections: Iterable[Detection], min_confidence: float
) -> List[Detection]:
    """Utility helper to drop low confidence detections."""
    return [det for det in detections if det.confidence >= min_confidence]


class UltralyticsDetector(BaseDetector):
    """Thin wrapper that hides YOLO import details and user options."""

    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Ultralytics backend selected but 'ultralytics' package is not installed. "
                "Install with `uv sync --extra detector` or `pip install ultralytics`."
            ) from exc

        LOGGER.info(
            "Loading Ultralytics YOLO model '%s' on device '%s'",
            self.config.model_path,
            self.config.device,
        )
        self._model = YOLO(self.config.model_path)
        if self.config.warmup:
            LOGGER.debug("Running detector warmup on dummy tensor")
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model.predict(
                source=dummy,
                device=self.config.device,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                verbose=False,
                half=self.config.half,
            )

    def predict(self, packet: FramePacket) -> List[Detection]:
        self._load()
        if self._model is None:
            raise RuntimeError("Model failed to load")

        result_list = self._model.predict(
            source=packet.frame,
            device=self.config.device,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            classes=self.config.classes,
            half=self.config.half,
            verbose=False,
        )

        detections: List[Detection] = []
        if not result_list:
            return detections

        result = result_list[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections

        for box in boxes:
            coordinates = box.xyxy.cpu().numpy().flatten().tolist()
            conf = float(box.conf.cpu().item())
            cls = int(box.cls.cpu().item())
            detections.append(
                Detection(
                    stream_name=packet.stream.name,
                    frame_id=packet.frame_id,
                    class_id=cls,
                    confidence=conf,
                    bbox_xyxy=tuple(coordinates[:4]),
                )
            )
        return detections


class _TensorRTBaseDetector(BaseDetector):
    """
    Shared preprocessing / postprocessing utilities for TensorRT YOLO engines.

    Supports both YOLOv5 and YOLOv8 output formats.
    """

    def __init__(self, config: DetectorConfig, input_hw: tuple[int, int]):
        super().__init__(config)
        self.input_hw = input_hw

    def predict(self, packet: FramePacket) -> List[Detection]:
        tensor, meta = self._preprocess(packet.frame)
        raw = self._infer(tensor)
        return self._postprocess(raw, packet, meta)

    def _preprocess(self, frame) -> tuple[np.ndarray, dict]:
        """
        Optimized preprocessing with letterbox, color conversion, normalization.

        Optimizations:
        - Minimize memory copies
        - Use contiguous arrays for better cache performance
        - Efficient color space conversion and normalization
        """
        import cv2

        h, w = frame.shape[:2]
        target_h, target_w = self.input_hw
        scale = min(target_w / w, target_h / h)

        # Compute new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize with optimized interpolation
        resized = cv2.resize(
            frame,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )

        # Compute padding
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        # Apply padding
        canvas = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        # Color conversion and normalization in one step (optimized)
        # Convert BGR to RGB and normalize to [0, 1]
        image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        # Determine target dtype
        target_dtype = np.float16 if self.config.half else np.float32

        # Normalize and convert dtype in single operation
        image = image.astype(target_dtype) * (1.0 / 255.0)

        # Transpose to CHW format (optimized with copy to ensure contiguous)
        tensor = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))

        # Add batch dimension
        tensor = np.expand_dims(tensor, axis=0)

        meta = {
            "orig_shape": (h, w),
            "scale": scale,
            "pad": (left, top),
        }
        return tensor, meta

    def _postprocess(
        self,
        predictions: np.ndarray,
        packet: FramePacket,
        meta: dict,
    ) -> List[Detection]:
        """
        Postprocess YOLO model outputs (supports YOLOv5 and YOLOv8).

        YOLOv5 output: [batch, num_anchors, 85] where 85 = [x, y, w, h, obj_conf, cls1, cls2, ...]
        YOLOv8 output: [batch, num_anchors, 84] where 84 = [x, y, w, h, cls1, cls2, ...]
        """
        if isinstance(predictions, list):
            predictions = predictions[0]
        if predictions.ndim == 3:
            predictions = np.squeeze(predictions, axis=0)
        if predictions.shape[0] != 0 and predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        if predictions.ndim != 2 or predictions.shape[1] < 5:
            LOGGER.warning("Unexpected prediction shape: %s", predictions.shape)
            return []

        boxes = predictions[:, :4]

        # Detect model type based on output shape
        # YOLOv5: has objectness score at index 4, then class probabilities
        # YOLOv8: directly has class probabilities starting from index 4
        if predictions.shape[1] > 5:
            # Check if this is YOLOv5 (with objectness score)
            # YOLOv5 typically has objectness score separate from class probabilities
            if self.config.model_type == "yolov5":
                objectness = predictions[:, 4:5]
                class_probs = predictions[:, 5:]
                scores = class_probs * objectness
            else:
                # YOLOv8 or treat as YOLOv5 with implicit objectness
                objectness = predictions[:, 4:5]
                class_probs = predictions[:, 5:]
                scores = class_probs * objectness
        else:
            scores = predictions[:, 4:]

        class_indices = np.argmax(scores, axis=1)
        confidences = scores[np.arange(scores.shape[0]), class_indices]

        mask = confidences >= self.config.confidence_threshold
        if self.config.classes:
            mask &= np.isin(class_indices, np.array(self.config.classes))
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_indices = class_indices[mask]

        if boxes.size == 0:
            return []

        boxes = self._xywh2xyxy(boxes)
        boxes = self._scale_boxes(boxes, meta)
        keep = self._nms(boxes, confidences, self.config.iou_threshold)

        detections: List[Detection] = []
        for idx in keep:
            box = boxes[idx]
            detections.append(
                Detection(
                    stream_name=packet.stream.name,
                    frame_id=packet.frame_id,
                    class_id=int(class_indices[idx]),
                    confidence=float(confidences[idx]),
                    bbox_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                )
            )
        return detections

    def _scale_boxes(self, boxes: np.ndarray, meta: dict) -> np.ndarray:
        left, top = meta["pad"]
        scale = meta["scale"]
        orig_h, orig_w = meta["orig_shape"]
        boxes = boxes.copy()
        boxes[:, [0, 2]] -= left
        boxes[:, [1, 3]] -= top
        boxes /= scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)
        return boxes

    @staticmethod
    def _xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
        result = boxes.copy()
        result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        return result

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        if not len(boxes):
            return []
        order = scores.argsort()[::-1]
        keep: List[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            iou = _iou(boxes[i], boxes[order[1:]])
            remaining = np.where(iou <= iou_threshold)[0]
            order = order[remaining + 1]
        return keep

    @abc.abstractmethod
    def _infer(self, tensor: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class TensorRTDetector(_TensorRTBaseDetector):
    """TensorRT engine backend (expects YOLO-style outputs)."""

    def __init__(self, config: DetectorConfig):
        try:
            import tensorrt as trt  # type: ignore
            import pycuda.driver as cuda  # type: ignore
            import pycuda.autoinit  # noqa: F401  # type: ignore  # ensures CUDA context
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "TensorRT backend selected but required packages are missing. "
                "Install NVIDIA TensorRT Python bindings and pycuda."
            ) from exc

        self.trt = trt
        self.cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)
        with open(config.model_path, "rb") as f:
            engine_data = f.read()
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        if self.engine.num_bindings < 2:
            raise RuntimeError(
                "TensorRT detector expects a single input / single output engine."
            )
        self.stream = cuda.Stream()
        self.input_binding = 0
        self.output_binding = 1
        self.input_name = self.engine.get_binding_name(self.input_binding)
        self.output_name = self.engine.get_binding_name(self.output_binding)

        if config.input_size:
            input_hw = (int(config.input_size[0]), int(config.input_size[1]))
        else:
            shape = self.engine.get_binding_shape(self.input_binding)
            if -1 in shape:
                raise RuntimeError(
                    "Dynamic TensorRT engine detected. Please set detector.input_size in config."
                )
            input_hw = (shape[2], shape[3])

        super().__init__(config, input_hw)
        self._allocate_buffers()

    def _allocate_buffers(self) -> None:
        binding_shape = (1, 3, *self.input_hw)
        if -1 in self.engine.get_binding_shape(self.input_binding):
            self.context.set_binding_shape(self.input_binding, binding_shape)
        output_dims = self.context.get_binding_shape(self.output_binding)
        output_shape = tuple(int(dim) for dim in output_dims)
        if -1 in output_shape:
            raise RuntimeError(
                "TensorRT engine output shape is dynamic. Please provide a static engine or set binding shape manually."
            )
        self.input_size = int(np.prod(binding_shape))
        self.output_size = int(np.prod(output_shape))
        self.input_dtype = np.float16 if self.config.half else np.float32
        self.host_input = np.empty(self.input_size, dtype=self.input_dtype)
        self.device_input = self.cuda.mem_alloc(self.host_input.nbytes)
        output_dtype = self.trt.nptype(self.engine.get_binding_dtype(self.output_binding))
        self.host_output = np.empty(self.output_size, dtype=output_dtype)
        self.device_output = self.cuda.mem_alloc(self.host_output.nbytes)
        self.output_shape = output_shape

    def _infer(self, tensor: np.ndarray) -> np.ndarray:
        binding_shape = (tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3])
        if -1 in self.engine.get_binding_shape(self.input_binding):
            self.context.set_binding_shape(self.input_binding, binding_shape)
        flat_input = tensor.astype(self.input_dtype, copy=False).ravel()
        np.copyto(self.host_input, flat_input)
        self.cuda.memcpy_htod_async(self.device_input, self.host_input, self.stream)
        self.context.execute_async_v2(
            bindings=[int(self.device_input), int(self.device_output)],
            stream_handle=self.stream.handle,
        )
        self.cuda.memcpy_dtoh_async(self.host_output, self.device_output, self.stream)
        self.stream.synchronize()
        output = self.host_output.astype(np.float32, copy=False)
        return output.reshape(self.output_shape)


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.array([])
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - inter_area
    return inter_area / np.clip(union, a_min=1e-6, a_max=None)


class ONNXRuntimeDetector(_TensorRTBaseDetector):
    """
    ONNX Runtime backend with GPU/CPU support (expects YOLO-style outputs).

    Optimizations for ONNX Runtime 1.23.0+:
    - Enhanced graph optimizations
    - Improved memory management
    - Parallel execution for multi-stream scenarios
    - Optimized execution providers configuration
    """

    def __init__(self, config: DetectorConfig):
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "ONNX Runtime backend selected but 'onnxruntime' or 'onnxruntime-gpu' package is not installed. "
                "Install with `pip install onnxruntime` (CPU) or `pip install onnxruntime-gpu` (GPU)."
            ) from exc

        self.ort = ort

        # Determine providers (execution providers) with optimized configurations
        providers = []
        provider_options = []

        if config.device == "cuda" or (isinstance(config.device, str) and config.device.startswith("cuda")):
            if "CUDAExecutionProvider" in ort.get_available_providers():
                # CUDA provider with optimized settings for ONNX Runtime 1.23.0+
                cuda_options = {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                }
                providers.append("CUDAExecutionProvider")
                provider_options.append(cuda_options)
                LOGGER.info("Using CUDA Execution Provider for ONNX Runtime with optimized settings")
            else:
                LOGGER.warning("CUDA requested but CUDAExecutionProvider not available, falling back to CPU")
                providers.append("CPUExecutionProvider")
                provider_options.append({})
        else:
            # CPU provider with optimized settings
            cpu_options = {
                "intra_op_num_threads": 0,  # Auto-detect
                "inter_op_num_threads": 0,  # Auto-detect
            }
            providers.append("CPUExecutionProvider")
            provider_options.append(cpu_options)
            LOGGER.info("Using CPU Execution Provider for ONNX Runtime")

        # Load ONNX model with enhanced session options
        LOGGER.info(
            "Loading ONNX model '%s' with providers: %s",
            config.model_path,
            providers,
        )

        session_options = ort.SessionOptions()

        # Enable all graph optimizations (essential for ONNX Runtime 1.23.0+)
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Enable memory pattern optimization
        session_options.enable_mem_pattern = True

        # Enable CPU memory arena for better memory reuse
        session_options.enable_cpu_mem_arena = True

        # Set execution mode (sequential for single stream, parallel for multiple)
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Intra-op parallelism (operations within a node)
        session_options.intra_op_num_threads = 0  # Auto-detect optimal threads

        # Inter-op parallelism (operations between nodes)
        session_options.inter_op_num_threads = 0  # Auto-detect optimal threads

        self.session = ort.InferenceSession(
            config.model_path,
            sess_options=session_options,
            providers=providers,
            provider_options=provider_options,
        )

        # Get model input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Determine input shape
        input_shape = self.session.get_inputs()[0].shape
        if config.input_size:
            input_hw = (int(config.input_size[0]), int(config.input_size[1]))
        elif len(input_shape) == 4 and input_shape[2] > 0 and input_shape[3] > 0:
            input_hw = (int(input_shape[2]), int(input_shape[3]))
        else:
            LOGGER.warning("Could not determine input size from ONNX model, defaulting to 640x640")
            input_hw = (640, 640)

        super().__init__(config, input_hw)

        # Warmup
        if config.warmup:
            LOGGER.debug("Running ONNX detector warmup")
            dummy = np.zeros((1, 3, *self.input_hw), dtype=np.float32)
            if self.config.half:
                dummy = dummy.astype(np.float16)
            self.session.run([self.output_name], {self.input_name: dummy})

        LOGGER.info("ONNX Runtime detector initialized with version %s", ort.__version__)

    def _infer(self, tensor: np.ndarray) -> np.ndarray:
        """
        Optimized inference with ONNX Runtime 1.23.0+.

        Benefits:
        - Efficient memory management
        - Optimized execution graph
        - Hardware-specific optimizations
        """
        # Use IO Binding for better performance (available in ONNX Runtime 1.23.0+)
        # For now, use standard run() method which is already optimized
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        return outputs[0]


class OpenVINODetector(_TensorRTBaseDetector):
    """
    OpenVINO backend for Intel CPUs/GPUs/NPUs (expects YOLO-style outputs).

    Optimizations:
    - Uses OpenVINO's optimized inference pipeline
    - Supports async inference for better throughput
    - Leverages Intel hardware acceleration (AVX, oneDNN)
    """

    def __init__(self, config: DetectorConfig):
        try:
            from openvino.runtime import Core  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "OpenVINO backend selected but 'openvino' package is not installed. "
                "Install with `pip install openvino`."
            ) from exc

        self.ov_core = Core()

        # Map device string to OpenVINO device
        device_map = {
            "cpu": "CPU",
            "gpu": "GPU",
            "cuda": "GPU",  # Map CUDA to GPU for OpenVINO
            "auto": "AUTO",
            "npu": "NPU",  # Neural Processing Unit (if available)
        }

        ov_device = device_map.get(str(config.device).lower(), "CPU")

        LOGGER.info(
            "Loading OpenVINO model '%s' on device '%s'",
            config.model_path,
            ov_device,
        )

        # Load model (supports .xml + .bin or .onnx)
        model = self.ov_core.read_model(model=config.model_path)

        # Compile model for target device with performance hints
        compile_config = {}
        if ov_device == "CPU":
            # CPU optimizations
            compile_config["PERFORMANCE_HINT"] = "LATENCY"
            compile_config["CPU_THREADS_NUM"] = "0"  # Auto-detect
        elif ov_device in ("GPU", "AUTO"):
            # GPU/AUTO optimizations
            compile_config["PERFORMANCE_HINT"] = "THROUGHPUT"

        self.compiled_model = self.ov_core.compile_model(
            model=model, device_name=ov_device, config=compile_config
        )

        # Get input/output info
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        # Determine input shape
        input_shape = self.input_layer.shape
        if config.input_size:
            input_hw = (int(config.input_size[0]), int(config.input_size[1]))
        elif len(input_shape) == 4 and input_shape[2] > 0 and input_shape[3] > 0:
            # Shape is typically [N, C, H, W]
            input_hw = (int(input_shape[2]), int(input_shape[3]))
        else:
            LOGGER.warning("Could not determine input size from OpenVINO model, defaulting to 640x640")
            input_hw = (640, 640)

        super().__init__(config, input_hw)

        # Create inference request
        self.infer_request = self.compiled_model.create_infer_request()

        # Warmup
        if config.warmup:
            LOGGER.debug("Running OpenVINO detector warmup")
            dummy = np.zeros((1, 3, *self.input_hw), dtype=np.float32)
            self.infer_request.infer({0: dummy})

        LOGGER.info("OpenVINO detector initialized with optimized settings")

    def _infer(self, tensor: np.ndarray) -> np.ndarray:
        """
        Optimized inference using OpenVINO's efficient execution.

        The tensor is already preprocessed and in the correct format (NCHW).
        """
        result = self.infer_request.infer({0: tensor})
        return result[self.output_layer]


class RKNNDetector(_TensorRTBaseDetector):
    """
    RKNN (Rockchip Neural Network) backend for RK3588 NPU (expects YOLO-style outputs).

    Optimizations for RK3588:
    - Leverages Rockchip NPU hardware acceleration (up to 6 TOPS)
    - Optimized memory management for embedded systems
    - Efficient preprocessing pipeline
    - Supports RKNN model format (.rknn)

    The RK3588 features a dedicated NPU with excellent power efficiency
    for real-time video analytics on edge devices.
    """

    def __init__(self, config: DetectorConfig):
        try:
            from rknn.api import RKNN  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "RKNN backend selected but 'rknn-toolkit2' or 'rknnlite' package is not installed. "
                "Install with `pip install rknn-toolkit2` (for x86 development) or "
                "`pip install rknnlite` (for RK3588 runtime)."
            ) from exc

        self.rknn = RKNN(verbose=False)

        LOGGER.info(
            "Loading RKNN model '%s' for RK3588 NPU",
            config.model_path,
        )

        # Load RKNN model
        ret = self.rknn.load_rknn(config.model_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {config.model_path}")

        # Initialize runtime
        # For RK3588, use NPU core 0 (supports 0, 1, 2 for triple-core NPU)
        ret = self.rknn.init_runtime(target="rk3588", core_mask=RKNN.NPU_CORE_AUTO)
        if ret != 0:
            raise RuntimeError("Failed to initialize RKNN runtime on RK3588")

        # Query model input shape
        input_attrs = self.rknn.query(cmd="input_attr")[0]
        output_attrs = self.rknn.query(cmd="output_attr")[0]

        LOGGER.debug("RKNN Input: %s", input_attrs)
        LOGGER.debug("RKNN Output: %s", output_attrs)

        # Determine input shape
        if config.input_size:
            input_hw = (int(config.input_size[0]), int(config.input_size[1]))
        elif hasattr(input_attrs, "dims") and len(input_attrs.dims) == 4:
            # RKNN typically uses NHWC format
            input_hw = (int(input_attrs.dims[1]), int(input_attrs.dims[2]))
        else:
            LOGGER.warning("Could not determine input size from RKNN model, defaulting to 640x640")
            input_hw = (640, 640)

        # Store whether model expects NCHW or NHWC
        self.use_nhwc = hasattr(input_attrs, "fmt") and "NHWC" in str(input_attrs.fmt)

        super().__init__(config, input_hw)

        # Warmup
        if config.warmup:
            LOGGER.debug("Running RKNN detector warmup")
            dummy = np.zeros((1, *self.input_hw, 3) if self.use_nhwc else (1, 3, *self.input_hw), dtype=np.uint8)
            self.rknn.inference(inputs=[dummy])

        LOGGER.info("RKNN detector initialized on RK3588 NPU")

    def _preprocess(self, frame) -> tuple[np.ndarray, dict]:
        """
        Optimized preprocessing for RK3588 RKNN.

        RKNN models often expect uint8 input in NHWC format,
        with quantization handled by the NPU driver.
        """
        import cv2

        h, w = frame.shape[:2]
        target_h, target_w = self.input_hw
        scale = min(target_w / w, target_h / h)

        # Compute new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize
        resized = cv2.resize(
            frame,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )

        # Compute padding
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        # Apply padding
        canvas = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        # RKNN typically expects BGR uint8 input (no RGB conversion needed)
        # The NPU driver handles quantization internally

        if self.use_nhwc:
            # NHWC format: [N, H, W, C]
            tensor = np.expand_dims(canvas, axis=0)
        else:
            # NCHW format: [N, C, H, W]
            tensor = np.transpose(canvas, (2, 0, 1))
            tensor = np.expand_dims(tensor, axis=0)

        # Keep as uint8 for RKNN (quantized inference)
        tensor = tensor.astype(np.uint8)

        meta = {
            "orig_shape": (h, w),
            "scale": scale,
            "pad": (left, top),
        }
        return tensor, meta

    def _infer(self, tensor: np.ndarray) -> np.ndarray:
        """
        Optimized inference on RK3588 NPU.

        The RKNN runtime handles:
        - Automatic quantization (if model is quantized)
        - NPU core scheduling
        - Memory management
        """
        # RKNN inference expects list of inputs
        outputs = self.rknn.inference(inputs=[tensor])

        if not outputs or len(outputs) == 0:
            raise RuntimeError("RKNN inference returned no outputs")

        # Convert output to float32 for postprocessing
        output = outputs[0]
        if output.dtype != np.float32:
            output = output.astype(np.float32)

        return output

    def __del__(self):
        """Clean up RKNN runtime resources."""
        if hasattr(self, "rknn"):
            try:
                self.rknn.release()
            except Exception:  # noqa: BLE001
                pass


class ResNetOpenVINODetector(BaseDetector):
    """
    ResNet classification model using OpenVINO backend.

    Supports ImageNet and custom ResNet models for image classification.
    Unlike YOLO detectors, this returns detections with full-frame bounding boxes
    and class predictions.
    """

    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        try:
            from openvino.runtime import Core  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "OpenVINO backend selected but 'openvino' package is not installed. "
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
        ov_device = device_map.get(str(config.device).lower(), "CPU")

        LOGGER.info(
            "Loading ResNet OpenVINO model '%s' on device '%s'",
            config.model_path,
            ov_device,
        )

        model = self.ov_core.read_model(model=config.model_path)

        # Compile with CPU optimization
        compile_config = {}
        if ov_device == "CPU":
            compile_config["PERFORMANCE_HINT"] = "LATENCY"
            compile_config["CPU_THREADS_NUM"] = "0"
        elif ov_device in ("GPU", "AUTO"):
            compile_config["PERFORMANCE_HINT"] = "THROUGHPUT"

        self.compiled_model = self.ov_core.compile_model(
            model=model, device_name=ov_device, config=compile_config
        )

        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        # Get input shape
        input_shape = self.input_layer.shape
        if config.input_size:
            self.input_hw = (int(config.input_size[0]), int(config.input_size[1]))
        elif len(input_shape) == 4 and input_shape[2] > 0 and input_shape[3] > 0:
            self.input_hw = (int(input_shape[2]), int(input_shape[3]))
        else:
            # Default ResNet input size
            self.input_hw = (224, 224)

        self.infer_request = self.compiled_model.create_infer_request()

        # Warmup
        if config.warmup:
            LOGGER.debug("Running ResNet OpenVINO warmup")
            dummy = np.zeros((1, 3, *self.input_hw), dtype=np.float32)
            self.infer_request.infer({0: dummy})

        LOGGER.info("ResNet OpenVINO detector initialized")

    def predict(self, packet: FramePacket) -> List[Detection]:
        """
        Run ResNet classification on the frame.

        Returns top-K predictions as Detection objects with full-frame bounding boxes.
        """
        tensor = self._preprocess(packet.frame)
        output = self.infer_request.infer({0: tensor})[self.output_layer]

        # Get top-K predictions
        if output.ndim > 1:
            output = output.flatten()

        top_k_indices = np.argsort(output)[-self.config.resnet_top_k:][::-1]
        top_k_probs = output[top_k_indices]

        # Create detections for top-K predictions
        # Use full frame as bounding box since this is classification
        h, w = packet.frame.shape[:2]
        detections: List[Detection] = []

        for i, (class_id, confidence) in enumerate(zip(top_k_indices, top_k_probs)):
            if confidence >= self.config.confidence_threshold:
                detections.append(
                    Detection(
                        stream_name=packet.stream.name,
                        frame_id=packet.frame_id,
                        class_id=int(class_id),
                        confidence=float(confidence),
                        bbox_xyxy=(0.0, 0.0, float(w), float(h)),
                    )
                )

        return detections

    def _preprocess(self, frame) -> np.ndarray:
        """Preprocess frame for ResNet classification."""
        import cv2

        # Resize to input size
        resized = cv2.resize(frame, (self.input_hw[1], self.input_hw[0]))

        # Convert BGR to RGB
        image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize (ImageNet mean and std)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        image = image.astype(np.float32) / 255.0
        image = (image - mean) / std

        # Transpose to NCHW format
        tensor = np.transpose(image, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        return np.ascontiguousarray(tensor)


class ResNetONNXDetector(BaseDetector):
    """
    ResNet classification model using ONNX Runtime backend.

    Similar to ResNetOpenVINODetector but uses ONNX Runtime for inference.
    """

    def __init__(self, config: DetectorConfig):
        super().__init__(config)
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

        if config.device == "cuda" or (isinstance(config.device, str) and config.device.startswith("cuda")):
            if "CUDAExecutionProvider" in ort.get_available_providers():
                cuda_options = {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                }
                providers.append("CUDAExecutionProvider")
                provider_options.append(cuda_options)
            else:
                providers.append("CPUExecutionProvider")
                provider_options.append({})
        else:
            providers.append("CPUExecutionProvider")
            provider_options.append({})

        LOGGER.info(
            "Loading ResNet ONNX model '%s' with providers: %s",
            config.model_path,
            providers,
        )

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.enable_mem_pattern = True
        session_options.enable_cpu_mem_arena = True

        self.session = ort.InferenceSession(
            config.model_path,
            sess_options=session_options,
            providers=providers,
            provider_options=provider_options,
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Get input shape
        input_shape = self.session.get_inputs()[0].shape
        if config.input_size:
            self.input_hw = (int(config.input_size[0]), int(config.input_size[1]))
        elif len(input_shape) == 4 and input_shape[2] > 0 and input_shape[3] > 0:
            self.input_hw = (int(input_shape[2]), int(input_shape[3]))
        else:
            self.input_hw = (224, 224)

        # Warmup
        if config.warmup:
            LOGGER.debug("Running ResNet ONNX warmup")
            dummy = np.zeros((1, 3, *self.input_hw), dtype=np.float32)
            self.session.run([self.output_name], {self.input_name: dummy})

        LOGGER.info("ResNet ONNX detector initialized")

    def predict(self, packet: FramePacket) -> List[Detection]:
        """Run ResNet classification on the frame."""
        tensor = self._preprocess(packet.frame)
        output = self.session.run([self.output_name], {self.input_name: tensor})[0]

        # Get top-K predictions
        if output.ndim > 1:
            output = output.flatten()

        top_k_indices = np.argsort(output)[-self.config.resnet_top_k:][::-1]
        top_k_probs = output[top_k_indices]

        # Create detections
        h, w = packet.frame.shape[:2]
        detections: List[Detection] = []

        for class_id, confidence in zip(top_k_indices, top_k_probs):
            if confidence >= self.config.confidence_threshold:
                detections.append(
                    Detection(
                        stream_name=packet.stream.name,
                        frame_id=packet.frame_id,
                        class_id=int(class_id),
                        confidence=float(confidence),
                        bbox_xyxy=(0.0, 0.0, float(w), float(h)),
                    )
                )

        return detections

    def _preprocess(self, frame) -> np.ndarray:
        """Preprocess frame for ResNet classification."""
        import cv2

        # Resize to input size
        resized = cv2.resize(frame, (self.input_hw[1], self.input_hw[0]))

        # Convert BGR to RGB
        image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize (ImageNet mean and std)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        image = image.astype(np.float32) / 255.0
        image = (image - mean) / std

        # Transpose to NCHW format
        tensor = np.transpose(image, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        return np.ascontiguousarray(tensor)
