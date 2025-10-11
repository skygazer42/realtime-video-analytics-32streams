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

from .config import DetectorConfig, PipelineConfig, StreamConfig
from .detector import BaseDetector, Detection, create_detector, filter_detections
from .sinks import KafkaSink
from .telemetry import MetricsPublisher
from .tracker import IOUTracker
from .video_stream import FramePacket, VideoStream
from .utils import MotionFilter, MotionFilterConfig, apply_roi, downsample

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class StreamWorkerContext:
    stream: StreamConfig
    detector: BaseDetector
    tracker: IOUTracker
    kafka: KafkaSink
    metrics: MetricsPublisher
    motion_filter: MotionFilter | None = None


class StreamWorker:
    """Runs detection/tracking for a single video stream."""

    def __init__(self, context: StreamWorkerContext):
        self.ctx = context
        self.roi_polygons = context.stream.roi_polygons or []
        self.downsample_ratio = context.stream.downsample_ratio
        if context.stream.motion_filter:
            self.motion_filter_config = MotionFilterConfig(
                enable=True,
                threshold=context.stream.motion_threshold,
            )
        else:
            self.motion_filter_config = None
        self.motion_filter: MotionFilter | None = None
        self._frame_index = 0
        self._idle_frames = 0
        self._process_every = 1
        self._adaptive_enabled = context.stream.adaptive_fps
        if self._adaptive_enabled:
            target = context.stream.target_fps or 30.0
            min_fps = max(context.stream.min_target_fps, 1.0)
            ratio = max(1, int(round(target / min_fps)))
            self._max_process_every = ratio
            self._idle_tolerance = max(int(context.stream.idle_frame_tolerance), 1)
        else:
            self._max_process_every = 1
            self._idle_tolerance = 0

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
        self._frame_index += 1

        frame_for_detection = packet.frame
        if self.roi_polygons:
            frame_for_detection = apply_roi(frame_for_detection, self.roi_polygons)

        ratio = self.downsample_ratio
        if ratio < 0.999:
            frame_for_detection = downsample(frame_for_detection, ratio)

        if self.motion_filter_config:
            if self.motion_filter is None:
                self.motion_filter = MotionFilter(self.motion_filter_config, frame_for_detection.shape)
            if not self.motion_filter.should_process(frame_for_detection):
                await self._skip_frame(packet)
                return

        if self._adaptive_enabled and self._process_every > 1:
            if (self._frame_index - 1) % self._process_every != 0:
                await self._skip_frame(packet)
                return

        detection_packet = FramePacket(
            stream=packet.stream,
            frame=frame_for_detection,
            frame_id=packet.frame_id,
            timestamp=packet.timestamp,
        )

        detections = self.ctx.detector.predict(detection_packet)
        if ratio < 0.999:
            detections = self._rescale_detections(detections, ratio)
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
            frame=packet.frame,  # 将原始帧传递给 Kafka，用于可选的可视化
        )
        self._adjust_adaptive_state(len(filtered), len(tracks))

    async def _skip_frame(self, packet: FramePacket) -> None:
        tracks = self.ctx.tracker.update(packet.stream.name, [])
        self.ctx.metrics.update_counters(
            stream=packet.stream.name,
            frames_processed=1,
            detections_emitted=0,
            active_tracks=len(tracks),
        )
        self._adjust_adaptive_state(0, len(tracks))

    def _rescale_detections(self, detections: List[Detection], ratio: float) -> List[Detection]:
        if ratio >= 0.999:
            return detections
        scale = 1.0 / max(ratio, 1e-6)
        scaled: List[Detection] = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            scaled.append(
                Detection(
                    stream_name=det.stream_name,
                    frame_id=det.frame_id,
                    class_id=det.class_id,
                    confidence=det.confidence,
                    bbox_xyxy=(x1 * scale, y1 * scale, x2 * scale, y2 * scale),
                )
            )
        return scaled

    def _adjust_adaptive_state(self, detections_count: int, tracks_count: int) -> None:
        if not self._adaptive_enabled:
            return
        if detections_count > 0 or tracks_count > 0:
            if self._process_every != 1:
                LOGGER.debug(
                    "Stream '%s' restoring full FPS after activity",
                    self.ctx.stream.name,
                )
            self._idle_frames = 0
            self._process_every = 1
        else:
            self._idle_frames += 1
            if self._idle_frames >= self._idle_tolerance:
                if self._process_every != self._max_process_every:
                    LOGGER.debug(
                        "Stream '%s' reducing detection frequency (process every %d frames)",
                        self.ctx.stream.name,
                        self._max_process_every,
                    )
                self._process_every = max(self._max_process_every, 1)


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

        detector_configs: dict[str, DetectorConfig] = {"__default__": self.config.detector}
        detector_configs.update(self.config.detectors)
        detector_instances: dict[str, BaseDetector] = {
            key: create_detector(cfg)
            for key, cfg in detector_configs.items()
        }

        for stream in self.config.streams:
            if not stream.enabled:
                LOGGER.info("Skipping disabled stream '%s'", stream.name)
                continue
            key = stream.detector_id or "__default__"
            detector = detector_instances.get(key)
            if detector is None:
                LOGGER.warning(
                    "Stream '%s' references unknown detector_id='%s', falling back to default",
                    stream.name,
                    key,
                )
                detector = detector_instances["__default__"]
            motion_filter = None
            if stream.motion_filter:
                motion_filter = MotionFilter(MotionFilterConfig(enable=True, threshold=stream.motion_threshold), (0, 0, 0))
            context = StreamWorkerContext(
                stream=stream,
                detector=detector,
                tracker=self.tracker,
                kafka=self.kafka,
                metrics=self.metrics,
                motion_filter=motion_filter,
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
