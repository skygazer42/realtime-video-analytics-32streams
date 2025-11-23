"""
Prometheus metrics helper utilities.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from ..config import PrometheusConfig

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineCounters:
    """Aggregated counters that feed into Prometheus."""

    frames_processed: int = 0
    detections_emitted: int = 0
    tracks_active: int = 0


class MetricsPublisher:
    """Expose basic metrics via an HTTP endpoint."""

    def __init__(self, config: PrometheusConfig):
        self.config = config
        self._registry = None
        self._frame_counter = None
        self._detection_counter = None
        self._active_tracks_gauge = None

        # Temporal detection metrics
        self._temporal_sequences_counter = None
        self._temporal_buffer_size_gauge = None
        self._temporal_inference_duration_histogram = None

        self._task: Optional[asyncio.Task] = None

    def _lazy_init(self) -> None:
        if self._registry is not None:
            return
        try:
            from prometheus_client import Gauge, Counter, Histogram, CollectorRegistry, start_http_server
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Prometheus metrics enabled but prometheus_client is not installed. "
                "Install it with `pip install prometheus-client`."
            ) from exc

        self._registry = CollectorRegistry()
        self._frame_counter = Counter(
            "stream_frames_total",
            "Total frames processed per stream",
            ["stream"],
            registry=self._registry,
        )
        self._detection_counter = Counter(
            "stream_detections_total",
            "Total object detections emitted per stream",
            ["stream"],
            registry=self._registry,
        )
        self._active_tracks_gauge = Gauge(
            "stream_active_tracks",
            "Number of active tracks per stream",
            ["stream"],
            registry=self._registry,
        )

        # Temporal detection metrics
        self._temporal_sequences_counter = Counter(
            "temporal_sequences_total",
            "Total temporal sequences processed per stream",
            ["stream", "model_type"],
            registry=self._registry,
        )
        self._temporal_buffer_size_gauge = Gauge(
            "temporal_buffer_size",
            "Current size of temporal frame buffer per stream",
            ["stream"],
            registry=self._registry,
        )
        self._temporal_inference_duration_histogram = Histogram(
            "temporal_inference_duration_seconds",
            "Temporal model inference duration in seconds",
            ["stream", "model_type"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self._registry,
        )

        start_http_server(port=self.config.port, addr=self.config.host, registry=self._registry)
        LOGGER.info(
            "Prometheus endpoint available at http://%s:%d/metrics",
            self.config.host,
            self.config.port,
        )

    async def start(self) -> None:
        if not self.config.enabled:
            return
        self._lazy_init()
        if self._task:
            return
        self._task = asyncio.create_task(self._ticker())

    async def _ticker(self) -> None:
        while True:
            await asyncio.sleep(self.config.interval_seconds)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def update_counters(
        self,
        stream: str,
        frames_processed: int = 0,
        detections_emitted: int = 0,
        active_tracks: int | None = None,
    ) -> None:
        if not self.config.enabled or self._registry is None:
            return
        assert self._frame_counter and self._detection_counter and self._active_tracks_gauge
        if frames_processed:
            self._frame_counter.labels(stream=stream).inc(frames_processed)
        if detections_emitted:
            self._detection_counter.labels(stream=stream).inc(detections_emitted)
        if active_tracks is not None:
            self._active_tracks_gauge.labels(stream=stream).set(active_tracks)

    def update_temporal_metrics(
        self,
        stream: str,
        model_type: str,
        sequences_processed: int = 0,
        buffer_size: int | None = None,
        inference_duration: float | None = None,
    ) -> None:
        """
        Update metrics specific to temporal video analysis.

        Args:
            stream: Stream name
            model_type: Temporal model type (cnn_lstm, 3d_cnn, etc.)
            sequences_processed: Number of sequences processed
            buffer_size: Current frame buffer size
            inference_duration: Inference duration in seconds
        """
        if not self.config.enabled or self._registry is None:
            return

        if sequences_processed and self._temporal_sequences_counter:
            self._temporal_sequences_counter.labels(
                stream=stream, model_type=model_type
            ).inc(sequences_processed)

        if buffer_size is not None and self._temporal_buffer_size_gauge:
            self._temporal_buffer_size_gauge.labels(stream=stream).set(buffer_size)

        if inference_duration is not None and self._temporal_inference_duration_histogram:
            self._temporal_inference_duration_histogram.labels(
                stream=stream, model_type=model_type
            ).observe(inference_duration)
