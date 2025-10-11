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

## Quick start (uv)

```bash
# (Optional) Ensure a managed Python runtime is available
uv python install 3.11

# Sync dependencies (installs the local package + extras into .venv using pylock.toml)
uv sync --extra full

# Copy the sample config and edit stream URLs (RTSP/RTMP/file paths supported)
cp config/sample-pipeline.yaml my-pipeline.yaml
vi my-pipeline.yaml

# Run the analytics pipeline
uv run realtime-analytics --config my-pipeline.yaml --log-level INFO
```

Use `uv run python scripts/run_pipeline.py --config ...` if you prefer to execute the module directly.

### Alternative: pip / virtualenv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[full]"
python scripts/run_pipeline.py --config my-pipeline.yaml --log-level INFO
```

When running in production you can install the package (via `uv sync` or `pip install`) and use the entrypoint:

```bash
pip install realtime-video-analytics-32streams[full]
realtime-analytics --config /etc/analytics/pipeline.yaml
```

## Docker

```bash
# Build the container image
docker compose build

# Run the pipeline (uses ./config/pipeline.yaml by default)
docker compose up pipeline

# Optional: run the dashboard alongside
docker compose up dashboard
```

Mounted volumes:

- `./config` ➜ `/data/config` – place your pipeline YAML here (e.g. `config/pipeline.yaml`).
- `./models` ➜ `/app/models` – store Ultralytics `.pt` 或 TensorRT `.engine` 模型。

Environment overrides:

- `PIPELINE_CONFIG` – pipeline service config path，默认 `/data/config/pipeline.yaml`。
- `DASHBOARD_CONFIG` / `DASHBOARD_PORT` – 控制看板引用的配置与端口。

> 若需在容器中使用 TensorRT 引擎，请确保宿主机具备 GPU 与匹配的 NVIDIA 驱动，并基于官方 TensorRT/CUDA 基础镜像构建，运行时添加 `--gpus all`。

## Dashboard

A lightweight FastAPI dashboard is available to visualize detection metadata (per-stream track counts). The server listens to the same Kafka topic that the pipeline publishes.

```bash
# Install optional dashboard dependencies (uv)
uv sync --extra dashboard

# Start the dashboard; reuse pipeline Kafka settings via config
uv run realtime-analytics-dashboard --config my-pipeline.yaml --port 8080

# Open your browser
open http://localhost:8080
```

The dashboard exposes:

- `/` – realtime table of streams and active tracks (WebSocket updates)
- `/api/snapshot` – JSON snapshot of the latest detections
- `/ws` – WebSocket endpoint (for custom clients)

To visualize annotated frames, enable Kafka frame payloads in your pipeline config:

```yaml
kafka:
  enabled: true
  include_frames: true
```

When Kafka is disabled, the dashboard still runs but no events are shown until detections are published.

## Configuration reference

Everything is expressed in YAML (see `config/sample-pipeline.yaml`). Key sections:

- `streams`: declarative list of up to 32 RTSP/RTMP sources. Each item accepts `name`, `url`, optional `target_fps`, warm-up, and reconnect controls.
- `detector`: backend (`ultralytics` \| `tensorrt`), model/engine path, device, confidence & IoU thresholds, class filtering.
- `tracker`: parameters for the IOU tracker (acts as a ByteTrack-compatible shim).
- `kafka`: optional sink to publish events through `aiokafka` (disabled by default).
- `kafka.include_frames`: toggle inline JPEG previews (set to `true` to enable dashboard thumbnails).
- `prometheus`: enable the HTTP `/metrics` endpoint and configure listen address/port.
- `max_concurrent_streams`: hard guard against accidental over-subscription.

`src/realtime_analytics/config.py` declares the dataclasses, validation logic, and defaults. For backend-specific setup guides (Ultralytics YOLO / TensorRT) see `docs/model_integration_cn.md`.

## Module layout

- `video_stream.py` – async wrapper around OpenCV capture, handles warm-up, retries, and frame pacing.
- `detector.py` – Pluggable detector backends (Ultralytics YOLO / TensorRT helpers).
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
