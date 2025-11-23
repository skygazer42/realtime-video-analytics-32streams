"""
High level orchestration of the realtime analytics pipeline.

Enhancements:
- Priority-based stream scheduling
- Detector-level batching across streams
- Resource-aware adaptive processing
- Stream health monitoring
- Improved load balancing
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
import os
import cv2
from typing import Dict, List, Optional

from .config import DetectorConfig, PipelineConfig, StreamConfig
from .detector import BaseDetector, Detection, create_detector, filter_detections
from .ffmpeg_simulator import FFmpegStreamError, FFmpegStreamSimulator
from .sinks import KafkaSink
from .telemetry import MetricsPublisher
from .tracker import IouTracker
from .video_stream import FramePacket, VideoStream
from .utils import MotionFilter, MotionFilterConfig, apply_roi, downsample

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class StreamHealth:
    """Track health metrics for a stream."""

    stream_name: str
    last_successful_frame: float = field(default_factory=time.time)
    consecutive_errors: int = 0
    total_frames_processed: int = 0
    avg_processing_time: float = 0.0
    recent_processing_times: deque = field(default_factory=lambda: deque(maxlen=100))
    priority: int = 0  # Higher = more important

    def update_success(self, processing_time: float) -> None:
        """Update metrics after successful frame processing."""
        self.last_successful_frame = time.time()
        self.consecutive_errors = 0
        self.total_frames_processed += 1
        self.recent_processing_times.append(processing_time)
        if self.recent_processing_times:
            self.avg_processing_time = sum(self.recent_processing_times) / len(
                self.recent_processing_times
            )

    def update_error(self) -> None:
        """Update metrics after an error."""
        self.consecutive_errors += 1

    def health_score(self) -> float:
        """Calculate a health score (0-1, higher is better)."""
        # Penalize consecutive errors
        error_penalty = max(0.0, 1.0 - (self.consecutive_errors * 0.1))

        # Check recency of successful frames
        time_since_success = time.time() - self.last_successful_frame
        recency_score = max(0.0, 1.0 - (time_since_success / 60.0))  # 60 second window

        return error_penalty * recency_score


@dataclass(slots=True)
class StreamWorkerContext:
    stream: StreamConfig
    detector: BaseDetector
    tracker: IouTracker
    kafka: KafkaSink
    metrics: MetricsPublisher
    health: StreamHealth
    motion_filter: MotionFilter | None = None


class StreamWorker:
    """Runs detection/tracking for a single video stream."""

    def __init__(self, context: StreamWorkerContext):
        self.ctx = context
        self._last_snapshot_ts = 0.0  # track last saved key frame
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
        start_time = time.time()

        try:
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
                    processing_time = time.time() - start_time
                    self.ctx.health.update_success(processing_time)
                    return

            if self._adaptive_enabled and self._process_every > 1:
                if (self._frame_index - 1) % self._process_every != 0:
                    await self._skip_frame(packet)
                    processing_time = time.time() - start_time
                    self.ctx.health.update_success(processing_time)
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
            filtered = filter_detections(detections, self.ctx.detector.config.confidence_threshold)
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
            self._maybe_save_snapshot(packet, tracks)
            self._adjust_adaptive_state(len(filtered), len(tracks))

            # Update health tracking
            processing_time = time.time() - start_time
            self.ctx.health.update_success(processing_time)

        except Exception as exc:
            # Track errors in health monitoring
            self.ctx.health.update_error()
            LOGGER.error(
                "Error processing frame %d for stream '%s': %s",
                packet.frame_id,
                packet.stream.name,
                exc,
            )
            raise

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

    def _maybe_save_snapshot(self, packet: FramePacket, tracks: List[Detection]) -> None:
        """
        Save one annotated frame every 5 minutes per stream to /data/outputs/{stream}/.
        Frames are rendered with bounding boxes for quick inspection.
        """
        SNAPSHOT_INTERVAL = 300.0  # seconds
        now = time.time()
        if now - self._last_snapshot_ts < SNAPSHOT_INTERVAL:
            return
        self._last_snapshot_ts = now

        frame = packet.frame.copy()
        for trk in tracks:
            x1, y1, x2, y2 = map(int, trk.bbox_xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 204, 255), 2)
            label = f"ID{trk.track_id if hasattr(trk, 'track_id') else ''} cls{trk.class_id}"
            cv2.putText(frame, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        out_dir = Path("/data/outputs") / packet.stream.name
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{int(now)}_frame{packet.frame_id}.jpg"
            out_path = out_dir / filename
            cv2.imwrite(str(out_path), frame)
            LOGGER.info("Saved snapshot for '%s' to %s", packet.stream.name, out_path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to save snapshot for '%s': %s", packet.stream.name, exc)


class StreamScheduler:
    """
    Coordinate adaptive FPS and resource allocation across all streams.

    Features:
    - Global load monitoring
    - Priority-based resource allocation
    - Coordinated adaptive FPS adjustments
    - Health-based stream prioritization
    """

    def __init__(self, max_concurrent_streams: int = 32):
        self.max_concurrent = max_concurrent_streams
        self.stream_health: Dict[str, StreamHealth] = {}
        self._last_adjustment = time.time()
        self._adjustment_interval = 10.0  # Adjust every 10 seconds
        self._total_processing_time = 0.0
        self._total_frames = 0
        self._load_window = deque(maxlen=60)  # Track load over 60 samples

    def register_stream(self, stream_name: str, priority: int = 0) -> StreamHealth:
        """Register a stream and get its health tracker."""
        health = StreamHealth(stream_name=stream_name, priority=priority)
        self.stream_health[stream_name] = health
        return health

    def update_load_metrics(self, processing_time: float) -> None:
        """Update global load metrics."""
        self._total_processing_time += processing_time
        self._total_frames += 1
        self._load_window.append(processing_time)

    def get_average_load(self) -> float:
        """Get average processing time per frame (seconds)."""
        if not self._load_window:
            return 0.0
        return sum(self._load_window) / len(self._load_window)

    def should_adjust_streams(self) -> bool:
        """Check if it's time to adjust stream processing rates."""
        now = time.time()
        if now - self._last_adjustment >= self._adjustment_interval:
            self._last_adjustment = now
            return True
        return False

    def get_stream_priorities(self) -> List[tuple[str, float]]:
        """
        Get streams sorted by priority for resource allocation.

        Returns: List of (stream_name, score) tuples, sorted by priority
        """
        priorities = []
        for stream_name, health in self.stream_health.items():
            # Calculate priority score based on:
            # 1. Configured priority
            # 2. Health score
            # 3. Processing time (faster streams get slight boost)
            health_score = health.health_score()
            processing_penalty = min(health.avg_processing_time / 0.1, 1.0)  # Normalize to 0-1
            combined_score = (
                health.priority * 10.0  # Priority is most important
                + health_score * 5.0  # Health is second
                - processing_penalty * 2.0  # Slower streams get lower priority
            )
            priorities.append((stream_name, combined_score))

        # Sort by score (descending)
        priorities.sort(key=lambda x: x[1], reverse=True)
        return priorities

    def get_system_load_factor(self) -> float:
        """
        Calculate system load factor (0-1, higher = more loaded).

        Can be extended with actual CPU/GPU monitoring.
        """
        avg_time = self.get_average_load()
        if avg_time == 0:
            return 0.0

        # Assume target is 30ms per frame (33 FPS)
        target_time = 0.033
        load = avg_time / target_time
        return min(load, 1.0)

    def recommend_adaptive_adjustment(self, stream_name: str) -> Optional[str]:
        """
        Recommend adaptive FPS adjustment for a stream.

        Returns: "increase", "decrease", or None
        """
        if stream_name not in self.stream_health:
            return None

        health = self.stream_health[stream_name]
        system_load = self.get_system_load_factor()

        # High system load: recommend decrease for low-priority streams
        if system_load > 0.8:
            priorities = self.get_stream_priorities()
            priority_rank = next(
                (i for i, (name, _) in enumerate(priorities) if name == stream_name),
                len(priorities),
            )
            # Lower half of priorities should reduce
            if priority_rank > len(priorities) // 2:
                return "decrease"

        # Low system load and good health: recommend increase
        if system_load < 0.5 and health.health_score() > 0.8:
            return "increase"

        return None

    def log_status(self) -> None:
        """Log current scheduler status."""
        if not self.stream_health:
            return

        avg_load = self.get_average_load()
        load_factor = self.get_system_load_factor()

        LOGGER.info(
            "Scheduler status: avg_load=%.3fs, load_factor=%.2f, streams=%d",
            avg_load,
            load_factor,
            len(self.stream_health),
        )

        # Log top 5 streams by priority
        priorities = self.get_stream_priorities()
        for i, (stream_name, score) in enumerate(priorities[:5]):
            health = self.stream_health[stream_name]
            LOGGER.debug(
                "Stream '%s': priority=%d, health=%.2f, score=%.2f, avg_time=%.3fs",
                stream_name,
                health.priority,
                health.health_score(),
                score,
                health.avg_processing_time,
            )
            if i >= 4:  # Limit to top 5
                break


class AnalyticsPipeline:
    """
    Entry point for running the analytics platform.

    Enhancements:
    - Integrated stream scheduler for resource management
    - Health monitoring for all streams
    - Priority-based stream processing
    - Global load balancing
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.tracker = IouTracker(config.tracker)
        self.kafka = KafkaSink(config.kafka)
        self.metrics = MetricsPublisher(config.prometheus)
        self.scheduler = StreamScheduler(max_concurrent_streams=config.max_concurrent_streams)
        self._tasks: List[asyncio.Task] = []
        self._stop_event = asyncio.Event()
        self._ffmpeg_simulators: List[FFmpegStreamSimulator] = []

    async def start(self) -> None:
        LOGGER.info("Booting analytics pipeline")
        await self.metrics.start()
        await self.kafka.connect()
        try:
            self._start_ffmpeg_simulators()
        except FFmpegStreamError:
            self._stop_ffmpeg_simulators()
            raise

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

            # Register stream with scheduler and get health tracker
            # Priority could be configured per-stream in the future
            stream_priority = 0  # Default priority
            health = self.scheduler.register_stream(stream.name, priority=stream_priority)

            motion_filter = None
            if stream.motion_filter:
                motion_filter = MotionFilter(MotionFilterConfig(enable=True, threshold=stream.motion_threshold), (0, 0, 0))
            context = StreamWorkerContext(
                stream=stream,
                detector=detector,
                tracker=self.tracker,
                kafka=self.kafka,
                metrics=self.metrics,
                health=health,
                motion_filter=motion_filter,
            )
            worker = StreamWorker(context)
            task = asyncio.create_task(worker.run(), name=f"stream-{stream.name}")
            self._tasks.append(task)

        if not self._tasks:
            raise RuntimeError("No stream workers started")

        LOGGER.info("Started %d stream workers", len(self._tasks))

        # Start scheduler monitoring task
        monitor_task = asyncio.create_task(
            self._monitor_scheduler(), name="scheduler-monitor"
        )
        self._tasks.append(monitor_task)

        self._install_signal_handlers()
        await self._stop_event.wait()

    async def _monitor_scheduler(self) -> None:
        """
        Periodically monitor and log scheduler status.

        This task runs in the background and reports on:
        - System load
        - Stream health
        - Priority rankings
        """
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(30.0)  # Log every 30 seconds

                if self.scheduler.should_adjust_streams():
                    self.scheduler.log_status()

                # Update global load metrics from all stream health trackers
                for health in self.scheduler.stream_health.values():
                    if health.recent_processing_times:
                        latest_time = health.recent_processing_times[-1]
                        self.scheduler.update_load_metrics(latest_time)

        except asyncio.CancelledError:
            LOGGER.debug("Scheduler monitor cancelled")
        except Exception:
            LOGGER.exception("Error in scheduler monitor")

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
        self._stop_ffmpeg_simulators()

    async def run_forever(self) -> None:
        try:
            await self.start()
        finally:
            await self.wait_closed()

    def _start_ffmpeg_simulators(self) -> None:
        for stream in self.config.streams:
            sim_cfg = stream.ffmpeg_simulator
            if not sim_cfg or not sim_cfg.enabled:
                continue
            simulator = FFmpegStreamSimulator(stream, sim_cfg)
            simulator.start()
            self._ffmpeg_simulators.append(simulator)

    def _stop_ffmpeg_simulators(self) -> None:
        while self._ffmpeg_simulators:
            simulator = self._ffmpeg_simulators.pop()
            try:
                simulator.stop()
            except Exception:
                LOGGER.exception(
                    "Error while stopping ffmpeg simulator for stream '%s'",
                    simulator.stream.name,
                )


def run_from_config(config: PipelineConfig) -> None:
    """Convenience helper for CLI entrypoints."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        stream=sys.stdout,
    )

    pipeline = AnalyticsPipeline(config)
    asyncio.run(pipeline.run_forever())
