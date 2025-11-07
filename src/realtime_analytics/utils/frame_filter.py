"""Lightweight frame filtering strategies (motion detection, ROI masks, downsampling)."""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class MotionFilterConfig:
    enable: bool = False
    history: int = 5
    threshold: float = 0.02  # percentage of pixels changed
    blur_kernel: Tuple[int, int] = (5, 5)


class MotionFilter:
    def __init__(self, config: MotionFilterConfig, frame_shape: Tuple[int, int, int]):
        self.config = config
        self.previous_gray: np.ndarray | None = None
        self.alpha = 1.0 / max(1, config.history)
        self.accumulator: np.ndarray | None = None

    def should_process(self, frame: np.ndarray) -> bool:
        if not self.config.enable:
            return True

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.config.blur_kernel, 0)
        if self.previous_gray is None:
            self.previous_gray = gray
            return True

        diff = cv2.absdiff(gray, self.previous_gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion_ratio = float(np.count_nonzero(thresh)) / float(thresh.size)
        self.previous_gray = gray
        return motion_ratio >= self.config.threshold


def apply_roi(frame: np.ndarray, polygons: List[List[Tuple[int, int]]]) -> np.ndarray:
    if not polygons:
        return frame
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for polygon in polygons:
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return cv2.bitwise_and(frame, frame, mask=mask)


def downsample(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 0.999:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
