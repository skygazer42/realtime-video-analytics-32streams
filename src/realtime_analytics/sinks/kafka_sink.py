"""
Kafka sink for streaming detection + tracking results.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Iterable, Optional, Sequence

import cv2

from ..config import KafkaSinkConfig
from ..tracker import Track

LOGGER = logging.getLogger(__name__)


class KafkaSink:
    """Publish events to Kafka using aiokafka if available."""

    def __init__(self, config: KafkaSinkConfig):
        self.config = config  # Kafka 配置，包括是否启用、延迟、帧质量等
        self._producer = None  # 延迟创建的 Kafka 生产者实例
        self._lock = asyncio.Lock()  # 串行化发送，避免多协程同时写入

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
        track_list = [
            {
                "track_id": track.track_id,
                "class_id": track.class_id,
                "confidence": track.confidence,
                "bbox_xyxy": track.bbox_xyxy,
            }
            for track in tracks
        ]  # 将 Track 对象转换为可序列化的字典
        payload = {
            "stream": stream_name,
            "frame_id": frame_id,
            "tracks": track_list,
        }  # Kafka 消息主体
        if self.config.include_frames and frame is not None:
            try:
                payload["frame_jpeg"] = await asyncio.to_thread(
                    self._render_frame, frame, track_list
                )  # 异步线程中完成编码，避免阻塞事件循环
            except Exception:  # noqa: BLE001
                LOGGER.exception("Failed to encode frame preview for stream '%s'", stream_name)

        async with self._lock:
            await self._producer.send_and_wait(self.config.topic, payload)  # 发送并等待确认

    async def close(self) -> None:
        if self._producer is None:
            return
        await self._producer.stop()
        LOGGER.info("Kafka producer closed")

    def _render_frame(
        self,
        frame: "cv2.typing.MatLike",
        track_list: Sequence[dict],
    ) -> str:
        image = frame.copy()  # 避免修改原始帧
        for track in track_list:
            x1, y1, x2, y2 = [int(v) for v in track["bbox_xyxy"]]
            color = self._color_for(track["class_id"])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)  # 绘制检测框
            label = f'ID {track["track_id"]}'
            cv2.putText(
                image,
                label,
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )  # 标注轨迹 ID

        success, buffer = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.config.frame_quality)],
        )  # 根据配置压缩为 JPEG
        if not success:
            raise RuntimeError("cv2.imencode failed")
        encoded = base64.b64encode(buffer).decode("ascii")  # 转为 Base64 供前端直接展示
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _color_for(class_id: int) -> tuple[int, int, int]:
        seed = (hash(class_id) & 0xFFFFFF) or 0xFFAA33  # 基于类别 ID 生成稳定的颜色
        b = seed & 0xFF
        g = (seed >> 8) & 0xFF
        r = (seed >> 16) & 0xFF
        return int(b), int(g), int(r)
