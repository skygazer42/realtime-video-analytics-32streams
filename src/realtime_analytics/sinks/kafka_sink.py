"""
Kafka sink for streaming detection + tracking results.

Enhancements:
- Adaptive frame quality based on content
- Multiple codec support (JPEG, WebP)
- Frame downscaling for bandwidth optimization
- Selective frame sending based on detections
- Progressive JPEG encoding
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

from ..config import KafkaSinkConfig
from ..tracker import Track

LOGGER = logging.getLogger(__name__)


class KafkaSink:
    """
    Publish events to Kafka using aiokafka if available.

    Enhancements:
    - Adaptive quality based on detection count
    - Frame rate limiting to reduce bandwidth
    - Multiple codec support
    - Downscaling for bandwidth optimization
    """

    def __init__(self, config: KafkaSinkConfig):
        self.config = config  # Kafka 配置，包括是否启用、延迟、帧质量等
        self._producer = None  # 延迟创建的 Kafka 生产者实例
        self._lock = asyncio.Lock()  # 串行化发送，避免多协程同时写入

        # Frame rate limiting: send at most 1 frame per stream per interval
        self._last_frame_time: dict[str, float] = {}  # stream_name -> timestamp
        self._frame_send_interval = 0.1  # Send at most 10 FPS per stream

        # Adaptive quality settings
        self._use_adaptive_quality = True
        self._base_quality = self.config.frame_quality
        self._webp_available = self._check_webp_support()

    def _check_webp_support(self) -> bool:
        """Check if WebP encoding is available in OpenCV."""
        try:
            # Try encoding a small test image
            test_img = np.zeros((10, 10, 3), dtype=np.uint8)
            success, _ = cv2.imencode(".webp", test_img, [cv2.IMWRITE_WEBP_QUALITY, 75])
            return success
        except Exception:
            return False

    async def connect(self) -> None:
        if not self.config.enabled:
            return
        if self._producer:
            return  # 已创建生产者则跳过

        try:
            from aiokafka import AIOKafkaProducer  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Kafka sink enabled but aiokafka is not installed. "
                "Install it with `pip install aiokafka`."
            ) from exc

        LOGGER.info(
            "Connecting Kafka producer to %s (topic=%s)",
            self.config.bootstrap_servers,
            self.config.topic,
        )
        producer = AIOKafkaProducer(
            bootstrap_servers=self.config.bootstrap_servers,
            linger_ms=self.config.linger_ms,
            max_batch_size=self.config.max_batch_size,
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        )  # 使用 JSON 序列化消息
        await producer.start()
        self._producer = producer

    async def send_tracks(
        self,
        stream_name: str,
        frame_id: int,
        tracks: Iterable[Track],
        frame: Optional["cv2.typing.MatLike"] = None,
    ) -> None:
        if not self.config.enabled or not self._producer:
            return  # 未启用或生产者未连接时直接返回

        track_list = []
        has_temporal = False

        for track in tracks:
            track_dict = {
                "track_id": track.track_id,
                "class_id": track.class_id,
                "confidence": track.confidence,
                "bbox_xyxy": track.bbox_xyxy,
            }

            # Add temporal fields if present
            if track.action_label is not None:
                track_dict["action_label"] = track.action_label
                has_temporal = True
            if track.temporal_score is not None:
                track_dict["temporal_score"] = track.temporal_score
            if track.sequence_start_frame is not None:
                track_dict["sequence_start_frame"] = track.sequence_start_frame
            if track.sequence_end_frame is not None:
                track_dict["sequence_end_frame"] = track.sequence_end_frame

            track_list.append(track_dict)

        payload = {
            "stream": stream_name,
            "frame_id": frame_id,
            "tracks": track_list,
            "is_temporal": has_temporal,  # Indicate if this contains temporal detections
        }  # Kafka 消息主体

        # Frame rate limiting: only send frames at a controlled rate
        should_send_frame = self._should_send_frame(stream_name)

        if self.config.include_frames and frame is not None and should_send_frame:
            try:
                # Adaptive quality based on number of detections
                quality = self._calculate_adaptive_quality(len(track_list))

                payload["frame_jpeg"] = await asyncio.to_thread(
                    self._render_frame, frame, track_list, quality
                )  # 异步线程中完成编码，避免阻塞事件循环
            except Exception:  # noqa: BLE001
                LOGGER.exception("Failed to encode frame preview for stream '%s'", stream_name)

        async with self._lock:
            await self._producer.send_and_wait(self.config.topic, payload)  # 发送并等待确认

    def _should_send_frame(self, stream_name: str) -> bool:
        """
        Check if enough time has passed to send another frame for this stream.

        Implements frame rate limiting to reduce bandwidth.
        """
        now = time.time()
        last_time = self._last_frame_time.get(stream_name, 0.0)

        if now - last_time >= self._frame_send_interval:
            self._last_frame_time[stream_name] = now
            return True
        return False

    def _calculate_adaptive_quality(self, detection_count: int) -> int:
        """
        Calculate adaptive quality based on frame content.

        More detections = higher quality to preserve detail.
        Fewer detections = lower quality to save bandwidth.
        """
        if not self._use_adaptive_quality:
            return self._base_quality

        # Scale quality based on detection count
        # 0 detections: base quality - 10
        # 1-3 detections: base quality
        # 4-10 detections: base quality + 5
        # 10+ detections: base quality + 10

        if detection_count == 0:
            quality_boost = -10
        elif detection_count <= 3:
            quality_boost = 0
        elif detection_count <= 10:
            quality_boost = 5
        else:
            quality_boost = 10

        # Clamp quality between 50 and 95
        quality = max(50, min(95, self._base_quality + quality_boost))
        return quality

    async def close(self) -> None:
        if self._producer is None:
            return
        await self._producer.stop()
        LOGGER.info("Kafka producer closed")

    def _render_frame(
        self,
        frame: "cv2.typing.MatLike",
        track_list: Sequence[dict],
        quality: Optional[int] = None,
    ) -> str:
        """
        Render frame with detection boxes and encode.

        Enhancements:
        - Adaptive quality parameter
        - Progressive JPEG for better streaming
        - Optional downscaling for bandwidth optimization
        - WebP support if available
        """
        if quality is None:
            quality = self._base_quality

        image = frame.copy()  # 避免修改原始帧

        # Optional downscaling to reduce bandwidth
        # Downscale if frame is very large (> 1920x1080)
        h, w = image.shape[:2]
        scale_factor = 1.0
        if w > 1920 or h > 1080:
            scale_factor = min(1920 / w, 1080 / h)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Draw detection boxes and labels
        for track in track_list:
            x1, y1, x2, y2 = [int(v * scale_factor) for v in track["bbox_xyxy"]]
            color = self._color_for(track["class_id"])

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Draw label with background for better visibility
            label = f'ID {track["track_id"]}'
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )

            # Draw label background
            cv2.rectangle(
                image,
                (x1, max(0, y1 - label_h - baseline - 4)),
                (x1 + label_w, max(0, y1)),
                color,
                -1,
            )

            # Draw label text
            cv2.putText(
                image,
                label,
                (x1, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White text
                2,
                cv2.LINE_AA,
            )

        # Choose encoding format
        use_webp = self._webp_available and quality >= 80  # Use WebP for high quality
        if use_webp:
            success, buffer = cv2.imencode(
                ".webp",
                image,
                [cv2.IMWRITE_WEBP_QUALITY, quality],
            )
            mime_type = "image/webp"
        else:
            # Use progressive JPEG for better streaming
            success, buffer = cv2.imencode(
                ".jpg",
                image,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    quality,
                    cv2.IMWRITE_JPEG_PROGRESSIVE,
                    1,  # Enable progressive encoding
                    cv2.IMWRITE_JPEG_OPTIMIZE,
                    1,  # Enable optimization
                ],
            )
            mime_type = "image/jpeg"

        if not success:
            raise RuntimeError("cv2.imencode failed")

        encoded = base64.b64encode(buffer).decode("ascii")  # 转为 Base64 供前端直接展示
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _color_for(class_id: int) -> tuple[int, int, int]:
        seed = (hash(class_id) & 0xFFFFFF) or 0xFFAA33  # 基于类别 ID 生成稳定的颜色
        b = seed & 0xFF
        g = (seed >> 8) & 0xFF
        r = (seed >> 16) & 0xFF
        return int(b), int(g), int(r)
