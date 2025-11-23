"""
Async Kafka consumer feeding dashboard state updates.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from .schemas import DetectionEvent, TrackPayload
from .state import ConnectionManager, DashboardState

LOGGER = logging.getLogger(__name__)


class DetectionConsumer:
    """
    Consumes Kafka topics to push detection events to the dashboard.

    Designed to be optional: the server still runs without Kafka dependencies,
    allowing the frontend to connect but remain idle.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        state: DashboardState,
        connections: ConnectionManager,
        group_id: str = "realtime-analytics-dashboard",
        poll_interval: float = 0.5,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.poll_interval = poll_interval
        self.state = state
        self.connections = connections
        self._consumer = None
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        if self._task:
            return
        try:
            from aiokafka import AIOKafkaConsumer  # type: ignore
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Kafka consumer unavailable: install aiokafka to enable dashboard streaming (%s)",
                exc,
            )
            return

        LOGGER.info(
            "Starting Kafka consumer for topic '%s' (%s)",
            self.topic,
            self.bootstrap_servers,
        )
        consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            enable_auto_commit=True,
            group_id=self.group_id,
            value_deserializer=lambda value: value.decode("utf-8"),
        )
        await consumer.start()
        self._consumer = consumer
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(), name="dashboard-kafka-consumer")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._consumer:
            await self._consumer.stop()
            self._consumer = None

    async def _run(self) -> None:
        assert self._consumer is not None
        consumer = self._consumer
        try:
            while not self._stop_event.is_set():
                try:
                    message = await asyncio.wait_for(
                        consumer.getone(), timeout=self.poll_interval
                    )
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    LOGGER.exception("Kafka consumer error")
                    await asyncio.sleep(self.poll_interval)
                    continue
                try:
                    detection = self._parse_message(message.value)
                except Exception:
                    LOGGER.exception("Failed to parse detection payload")
                    continue
                await self.state.update(detection)
                await self.connections.broadcast_event(detection)
        except asyncio.CancelledError:
            LOGGER.debug("Kafka consumer cancelled")
            raise

    def _parse_message(self, raw: str) -> DetectionEvent:
        payload = json.loads(raw)  # 解析 Kafka JSON 消息
        tracks = [
            TrackPayload(
                track_id=track["track_id"],
                class_id=track["class_id"],
                confidence=track["confidence"],
                bbox_xyxy=list(track["bbox_xyxy"]),
            )
            for track in payload.get("tracks", [])
        ]
        return DetectionEvent(
            stream=payload.get("stream", "unknown"),
            frame_id=payload.get("frame_id", 0),
            tracks=tracks,
            frame_jpeg=payload.get("frame_jpeg"),  # 可能包含的图像数据
        )
