"""
Kafka sink for streaming detection + tracking results.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Iterable, Optional

from ..config import KafkaSinkConfig
from ..tracker import Track

LOGGER = logging.getLogger(__name__)


class KafkaSink:
    """Publish events to Kafka using aiokafka if available."""

    def __init__(self, config: KafkaSinkConfig):
        self.config = config
        self._producer = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        if not self.config.enabled:
            return
        if self._producer:
            return

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
        )
        await producer.start()
        self._producer = producer

    async def send_tracks(
        self,
        stream_name: str,
        frame_id: int,
        tracks: Iterable[Track],
    ) -> None:
        if not self.config.enabled or not self._producer:
            return
        payload = {
            "stream": stream_name,
            "frame_id": frame_id,
            "tracks": [
                {
                    "track_id": track.track_id,
                    "class_id": track.class_id,
                    "confidence": track.confidence,
                    "bbox_xyxy": track.bbox_xyxy,
                }
                for track in tracks
            ],
        }
        async with self._lock:
            await self._producer.send_and_wait(self.config.topic, payload)

    async def close(self) -> None:
        if self._producer is None:
            return
        await self._producer.stop()
        LOGGER.info("Kafka producer closed")
