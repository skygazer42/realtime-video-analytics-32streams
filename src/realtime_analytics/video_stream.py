"""
Asynchronous helpers for working with RTSP/RTMP video streams via OpenCV.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import cv2

from .config import StreamConfig

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class FramePacket:
    """Container for a video frame and associated metadata."""

    stream: StreamConfig
    frame: "cv2.typing.MatLike"
    frame_id: int
    timestamp: float


class VideoStream:
    """Wrapper around OpenCV capture supporting async frame retrieval."""

    def __init__(self, stream_config: StreamConfig):
        self.config = stream_config
        self._capture: Optional[cv2.VideoCapture] = None
        self._frame_id: int = 0

    async def __aenter__(self) -> "VideoStream":
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def open(self) -> None:
        if self._capture and self._capture.isOpened():
            return
        LOGGER.info("Opening stream '%s' (%s)", self.config.name, self.config.url)
        capture = cv2.VideoCapture(self.config.url, cv2.CAP_FFMPEG)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open stream {self.config.name}")
        if self.config.target_fps:
            capture.set(cv2.CAP_PROP_FPS, self.config.target_fps)
        self._capture = capture
        self._frame_id = 0

        if self.config.warmup_seconds > 0:
            LOGGER.debug(
                "Warming up stream '%s' for %.2fs",
                self.config.name,
                self.config.warmup_seconds,
            )
            await asyncio.sleep(self.config.warmup_seconds)

    async def close(self) -> None:
        if self._capture is not None:
            LOGGER.info("Closing stream '%s'", self.config.name)
            await asyncio.to_thread(self._capture.release)
            self._capture = None

    async def frames(self) -> AsyncGenerator[FramePacket, None]:
        if not self._capture:
            await self.open()

        retry_count = 0
        while True:
            if not self._capture:
                await asyncio.sleep(self.config.reconnect_backoff)
                continue
            success, frame = await asyncio.to_thread(self._capture.read)
            if not success or frame is None:
                retry_count += 1
                LOGGER.warning(
                    "Failed to read frame from '%s' (retry=%d)",
                    self.config.name,
                    retry_count,
                )
                if (
                    self.config.max_retries is not None
                    and retry_count >= self.config.max_retries
                ):
                    LOGGER.error(
                        "Giving up on stream '%s' after %d retries",
                        self.config.name,
                        retry_count,
                    )
                    break
                await asyncio.sleep(self.config.reconnect_backoff)
                continue

            retry_count = 0
            timestamp = time.time()
            packet = FramePacket(
                stream=self.config,
                frame=frame,
                frame_id=self._frame_id,
                timestamp=timestamp,
            )
            self._frame_id += 1
            yield packet

            if self.config.target_fps:
                await asyncio.sleep(max(0.0, 1.0 / self.config.target_fps))
