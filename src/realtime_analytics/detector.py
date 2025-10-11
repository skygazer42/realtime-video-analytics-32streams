"""
Object detection interface built around the Ultralytics YOLO models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List

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


class Detector:
    """Thin wrapper that hides YOLO import details and user options."""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self._model = None

    def load(self) -> None:
        """Load the model lazily to speed up CLI startup."""
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "The ultralytics package is required for YOLO detection. "
                "Install it with `pip install ultralytics`."
            ) from exc

        LOGGER.info(
            "Loading YOLO model '%s' on device '%s'",
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
        """Run inference on a frame packet and return detection results."""
        self.load()
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


def filter_detections(
    detections: Iterable[Detection], min_confidence: float
) -> List[Detection]:
    """Utility helper to drop low confidence detections."""
    return [det for det in detections if det.confidence >= min_confidence]
