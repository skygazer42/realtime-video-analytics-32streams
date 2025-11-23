"""
Lightweight tracker implementation (IOU-based) suitable as a placeholder.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from .config import TrackerConfig
from .detector import Detection

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Track:
    """Track state that we propagate across frames."""

    track_id: int
    class_id: int
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    age: int = 0
    hits: int = 0

    # Temporal detection fields (optional)
    action_label: str | None = None
    temporal_score: float | None = None
    sequence_start_frame: int | None = None
    sequence_end_frame: int | None = None


class IouTracker:
    """
    Very small IOU based tracker, keeps the public API open for future upgrades.

    For production we expect to swap this out with ByteTrack/DeepSORT bindings, yet
    this tracker allows the rest of the pipeline to remain functional without
    heavy dependencies.
    """

    def __init__(self, config: TrackerConfig):
        self.config = config
        self._next_track_id = itertools.count(1)
        self._tracks: Dict[str, Dict[int, Track]] = {}

    def update(self, stream_name: str, detections: Iterable[Detection]) -> List[Track]:
        detection_list = list(detections)
        tracks = self._tracks.setdefault(stream_name, {})
        matched_tracks: Dict[int, Track] = {}

        for detection in detection_list:
            match_id = self._match_detection(tracks, detection)

            # Extract temporal fields if this is a TemporalDetection
            temporal_fields = {}
            if hasattr(detection, "action_label"):
                temporal_fields["action_label"] = getattr(detection, "action_label", None)
            if hasattr(detection, "temporal_score"):
                temporal_fields["temporal_score"] = getattr(detection, "temporal_score", None)
            if hasattr(detection, "sequence_start_frame"):
                temporal_fields["sequence_start_frame"] = getattr(detection, "sequence_start_frame", None)
            if hasattr(detection, "sequence_end_frame"):
                temporal_fields["sequence_end_frame"] = getattr(detection, "sequence_end_frame", None)

            if match_id is None:
                track = Track(
                    track_id=next(self._next_track_id),
                    class_id=detection.class_id,
                    confidence=detection.confidence,
                    bbox_xyxy=detection.bbox_xyxy,
                    age=0,
                    hits=1,
                    **temporal_fields,
                )
                tracks[track.track_id] = track
                matched_tracks[track.track_id] = track
            else:
                track = tracks[match_id]
                track.bbox_xyxy = detection.bbox_xyxy
                track.confidence = detection.confidence
                track.hits += 1
                track.age = 0

                # Update temporal fields if present
                for key, value in temporal_fields.items():
                    setattr(track, key, value)

                matched_tracks[track.track_id] = track

        self._prune_tracks(stream_name, matched_tracks.keys())
        return list(tracks.values())

    def _match_detection(
        self, tracks: Dict[int, Track], detection: Detection
    ) -> int | None:
        best_iou = 0.0
        best_track_id: int | None = None
        for track_id, track in tracks.items():
            if track.class_id != detection.class_id:
                continue
            iou = _iou(track.bbox_xyxy, detection.bbox_xyxy)
            if iou >= self.config.max_iou_distance and iou > best_iou:
                best_iou = iou
                best_track_id = track_id
        return best_track_id

    def _prune_tracks(self, stream_name: str, active_ids: Iterable[int]) -> None:
        tracks = self._tracks[stream_name]
        active_set = set(active_ids)
        for track_id, track in list(tracks.items()):
            if track_id in active_set:
                continue
            track.age += 1
            if track.age > self.config.max_age or track.hits < self.config.min_hits:
                LOGGER.debug(
                    "Dropping track %d on stream '%s' (age=%d hits=%d)",
                    track_id,
                    stream_name,
                    track.age,
                    track.hits,
                )
                tracks.pop(track_id, None)


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area
