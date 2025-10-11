"""
Detector backends (Ultralytics YOLO, ONNX Runtime, TensorRT) and factory helpers.
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
    if backend == "ultralytics":
        return UltralyticsDetector(config)
    if backend == "tensorrt":
        return TensorRTDetector(config)
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
            dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
            self._model.predict(
                source=dummy,
                device=self.config.device,
                conf=self.config.conf_threshold,
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
            conf=self.config.conf_threshold,
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
    """Shared preprocessing / postprocessing utilities for TensorRT YOLO engines."""

    def __init__(self, config: DetectorConfig, input_hw: tuple[int, int]):
        super().__init__(config)
        self.input_hw = input_hw

    def predict(self, packet: FramePacket) -> List[Detection]:
        tensor, meta = self._preprocess(packet.frame)
        raw = self._infer(tensor)
        return self._postprocess(raw, packet, meta)

    def _preprocess(self, frame) -> tuple[np.ndarray, dict]:
        import cv2

        h, w = frame.shape[:2]
        target_h, target_w = self.input_hw
        scale = min(target_w / w, target_h / h)
        resized = cv2.resize(
            frame,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_LINEAR,
        )
        pad_w = target_w - resized.shape[1]
        pad_h = target_h - resized.shape[0]
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        canvas = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        tensor = np.transpose(image, (2, 0, 1))
        if self.config.half:
            tensor = tensor.astype(np.float16)
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
        if isinstance(predictions, list):
            predictions = predictions[0]
        if predictions.ndim == 3:
            predictions = np.squeeze(predictions, axis=0)
        if predictions.shape[0] != 0 and predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        if predictions.ndim != 2 or predictions.shape[1] < 6:
            LOGGER.warning("Unexpected prediction shape: %s", predictions.shape)
            return []

        boxes = predictions[:, :4]
        if predictions.shape[1] > 5:
            objectness = predictions[:, 4:5]
            class_probs = predictions[:, 5:]
            scores = class_probs * objectness
        else:
            scores = predictions[:, 4:]
        class_indices = np.argmax(scores, axis=1)
        confidences = scores[np.arange(scores.shape[0]), class_indices]

        mask = confidences >= self.config.conf_threshold
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
