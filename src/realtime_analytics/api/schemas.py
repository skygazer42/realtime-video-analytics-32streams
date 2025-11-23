"""
Pydantic models shared by the dashboard API and websocket clients.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field


class TrackPayload(BaseModel):
    track_id: int
    class_id: int
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: List[float] = Field(min_items=4, max_items=4)

    # Temporal detection fields (optional)
    action_label: Optional[str] = None  # Human-readable action name
    temporal_score: Optional[float] = None  # Temporal confidence score
    sequence_start_frame: Optional[int] = None  # First frame in sequence
    sequence_end_frame: Optional[int] = None  # Last frame in sequence


class DetectionEvent(BaseModel):
    stream: str
    frame_id: int
    tracks: List[TrackPayload] = Field(default_factory=list)
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    frame_jpeg: Optional[str] = None  # 可选的 Base64 图像数据

    # Temporal detection metadata
    is_temporal: bool = False  # Whether this event contains temporal detections
    sequence_info: Optional[dict] = None  # Additional sequence metadata

    @property
    def track_count(self) -> int:
        return len(self.tracks)


class DashboardSnapshot(BaseModel):
    streams: List[DetectionEvent] = Field(default_factory=list)


class WsEnvelope(BaseModel):
    type: str
    payload: DetectionEvent | DashboardSnapshot
