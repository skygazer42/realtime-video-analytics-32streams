"""
High level orchestration of the realtime analytics pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from dataclasses import dataclass
from typing import List

from .config import PipelineConfig, StreamConfig
from .detector import Detector, filter_detections
from .sinks import KafkaSink
from .telemetry import MetricsPublisher
from .tracker import IOUTracker
from .video_stream import FramePacket, VideoStream

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class StreamWorkerContext:
    stream: StreamConfig
    detector: Detector
    tracker: IOUTracker
    kafka: KafkaSink
    metrics: MetricsPublisher


class StreamWorker:
    """Runs detection/tracking for a single video stream."""

    def __init__(self, context: StreamWorkerContext):
        self.ctx = context

    async def run(self) -> None:
        stream = self.ctx.stream
        LOGGER.info("Starting worker for stream '%s'", stream.name)
        frame_counter = 0
        while True:
            try:
                async with VideoStream(stream) as video_stream:
                    async for packet in video_stream.frames():
                        frame_counter += 1
                        await self._process_packet(packet)
            except asyncio.CancelledError:
                LOGGER.info("Stream worker '%s' cancelled", stream.name)
                raise
            except Exception:
                LOGGER.exception("Unhandled exception in stream worker '%s'", stream.name)
                await asyncio.sleep(stream.reconnect_backoff)
                continue
            else:
                break
        LOGGER.info(
            "Stream worker '%s' shutting down after %d frames",
            stream.name,
            frame_counter,
        )

    async def _process_packet(self, packet: FramePacket) -> None:
        detections = self.ctx.detector.predict(packet)
        filtered = filter_detections(detections, self.ctx.detector.config.conf_threshold)
        tracks = self.ctx.tracker.update(packet.stream.name, filtered)
        self.ctx.metrics.update_counters(
            stream=packet.stream.name,
            frames_processed=1,
            detections_emitted=len(filtered),
            active_tracks=len(tracks),
        )
        await self.ctx.kafka.send_tracks(
            stream_name=packet.stream.name,
            frame_id=packet.frame_id,
            tracks=tracks,
        )


class AnalyticsPipeline:
    """Entry point for running the analytics platform."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.tracker = IOUTracker(config.tracker)
        self.kafka = KafkaSink(config.kafka)
        self.metrics = MetricsPublisher(config.prometheus)
        self._tasks: List[asyncio.Task] = []
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        LOGGER.info("Booting analytics pipeline")
        await self.metrics.start()
        await self.kafka.connect()
        for stream in self.config.streams:
            if not stream.enabled:
                LOGGER.info("Skipping disabled stream '%s'", stream.name)
                continue
            detector = Detector(self.config.detector)
            context = StreamWorkerContext(
                stream=stream,
                detector=detector,
                tracker=self.tracker,
                kafka=self.kafka,
                metrics=self.metrics,
            )
            worker = StreamWorker(context)
            task = asyncio.create_task(worker.run(), name=f"stream-{stream.name}")
            self._tasks.append(task)

        if not self._tasks:
            raise RuntimeError("No stream workers started")

        LOGGER.info("Started %d stream workers", len(self._tasks))
        self._install_signal_handlers()
        await self._stop_event.wait()

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.initiate_shutdown)
            except NotImplementedError:
                # Signal handling not supported (e.g. on Windows event loop)
                LOGGER.debug("Signal handler not supported on platform for %s", sig)

    def initiate_shutdown(self) -> None:
        if self._stop_event.is_set():
            return
        LOGGER.info("Shutdown requested, cancelling stream workers")
        self._stop_event.set()
        for task in self._tasks:
            task.cancel()

    async def wait_closed(self) -> None:
        if not self._tasks:
            return
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                LOGGER.exception("Stream worker task raised")
        await self.kafka.close()
        await self.metrics.stop()

    async def run_forever(self) -> None:
        try:
            await self.start()
        finally:
            await self.wait_closed()


def run_from_config(config: PipelineConfig) -> None:
    """Convenience helper for CLI entrypoints."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        stream=sys.stdout,
    )

    pipeline = AnalyticsPipeline(config)
    asyncio.run(pipeline.run_forever())
