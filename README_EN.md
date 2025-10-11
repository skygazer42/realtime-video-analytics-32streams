# realtime-video-analytics-32streams

Multi-stream (≤32) real-time video analytics pipeline with the following building blocks:

- RTSP/RTMP ingestion with asynchronous OpenCV capture.
- Ultralytics YOLOv8 detector (TensorRT-friendly layout; pluggable).
- Lightweight IOU tracker (ByteTrack/DeepSORT drop-in ready).
- Kafka sink for publishing structured detection/tracking events.
- Prometheus metrics exporter for Grafana dashboards.
- Configurable via YAML, packaged as an installable Python package/CLI.

> ⚠️ This repository currently ships with a reference Python-only implementation that you can run end-to-end.
> TensorRT / DeepStream optimisations can be layered on top by swapping the detector/tracker modules.

## Quick start

```bash
# Create a virtualenv (Python 3.10+)
python -m venv .venv
source .venv/bin/activate

# Install core dependencies + optional detector & Kafka extras
pip install -e ".[full]"

# Copy the sample config and edit stream URLs (RTSP/RTMP/file paths supported)
cp config/sample-pipeline.yaml my-pipeline.yaml
vi my-pipeline.yaml

# Run the analytics pipeline
python scripts/run_pipeline.py --config my-pipeline.yaml --log-level INFO
```

When running in production you can install the package and use the entrypoint:

```bash
pip install realtime-video-analytics-32streams[full]
realtime-analytics --config /etc/analytics/pipeline.yaml
```

## Configuration reference

Everything is expressed in YAML (see `config/sample-pipeline.yaml`). Key sections:

- `streams`: declarative list of up to 32 RTSP/RTMP sources. Each item accepts `name`, `url`, optional `target_fps`, warm-up, and reconnect controls.
- `detector`: YOLO model path (`yolov8n.pt` by default), device (`auto` / `cuda:0` / `cpu`), confidence & IoU thresholds, class filtering.
- `tracker`: parameters for the IOU tracker (acts as a ByteTrack-compatible shim).
- `kafka`: optional sink to publish events through `aiokafka` (disabled by default).
- `prometheus`: enable the HTTP `/metrics` endpoint and configure listen address/port.
- `max_concurrent_streams`: hard guard against accidental over-subscription.

`src/realtime_analytics/config.py` declares the dataclasses, validation logic, and defaults.

## Module layout

- `video_stream.py` – async wrapper around OpenCV capture, handles warm-up, retries, and frame pacing.
- `detector.py` – Ultralytics YOLO adaptor (lazy loading, warm-up, detection filtering).
- `tracker.py` – lightweight IOU-based tracker maintaining per-stream state.
- `pipeline.py` – orchestrates stream workers, detector/tracker execution, sink dispatch, and graceful shutdown.
- `sinks/kafka_sink.py` – optional Kafka producer (`aiokafka`) with JSON payloads.
- `telemetry/metrics.py` – Prometheus counters/gauges for frames, detections, active tracks.
- `scripts/run_pipeline.py` – CLI entrypoint, works directly from the repo or via package install.

## Next steps / roadmap

- Swap the reference detector with TensorRT/DeepStream bindings for lower latency.
- Introduce rule-engine hooks to trigger alerts/actions based on detection metadata.
- Provide Docker Compose for Kafka + Prometheus/Grafana observability stack.
- Add unit/integration tests (mock RTSP sources, synthetic frames).
- Harden stream reconnection/back-off strategies and implement health probes.
