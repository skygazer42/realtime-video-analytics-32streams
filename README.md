# 🎥 Realtime Video Analytics - 32 Streams

A production-ready, multi-stream (≤32) real-time video analytics pipeline with AI-powered object detection and tracking, featuring a modern web dashboard for monitoring and visualization.

## ✨ Key Features

### 🔧 Core Pipeline
- **Multi-Stream Processing**: Handle up to 32 concurrent RTSP/RTMP video streams
- **RTSP/RTMP Ingestion**: Asynchronous OpenCV capture with auto-reconnection
- **AI Detection**: Multiple inference backends (Ultralytics, ONNX Runtime, OpenVINO, TensorRT)
- **Object Tracking**: Lightweight IOU tracker (ByteTrack/DeepSORT compatible)
- **Event Streaming**: Kafka sink for publishing structured detection/tracking events
- **Observability**: Prometheus metrics exporter for Grafana dashboards
- **Flexible Configuration**: YAML-based configuration with per-stream customization

### 📊 Modern Web Dashboard
- **Real-time Monitoring**: Live WebSocket updates for instant stream status
- **Interactive UI**: Modern, responsive design with dark theme
- **Statistics Panel**: Real-time metrics (active streams, tracks, detections/sec, uptime)
- **Stream Management**: Search and filter streams by name or activity status
- **Visual Feedback**: Live annotated frame preview with bounding boxes
- **Track Details**: Comprehensive track information with confidence levels and coordinates
- **Performance Metrics**: FPS monitoring and stream health indicators

### 🚀 Advanced Capabilities
- **Frame Simulation**: Built-in FFmpeg simulator for testing without real cameras
- **ROI Filtering**: Polygon-based region of interest masking
- **Motion Detection**: Adaptive FPS based on scene activity
- **Frame Optimization**: Configurable downsampling for resource efficiency
- **Per-Stream Config**: Different models, FPS targets, and ROIs per camera

> ⚠️ This repository ships with a reference Python implementation that you can run end-to-end.
> TensorRT / DeepStream optimizations can be layered on top by swapping the detector/tracker modules.

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
# Install with Ultralytics YOLO (default)
pip install realtime-video-analytics-32streams[full]

# Install with ONNX Runtime support
pip install realtime-video-analytics-32streams[full-onnx]

# Install with OpenVINO support
pip install realtime-video-analytics-32streams[full-openvino]

# Run pipeline
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

## 📊 Dashboard

A modern, feature-rich FastAPI dashboard provides real-time visualization and monitoring of all video streams. The dashboard connects via WebSocket to display live detection events and stream statistics.

### 🚀 Quick Start

```bash
# Install optional dashboard dependencies (uv)
uv sync --extra dashboard

# Start the dashboard; reuse pipeline Kafka settings via config
uv run realtime-analytics-dashboard --config my-pipeline.yaml --port 8080

# Open your browser
open http://localhost:8080
```

### 🎯 Dashboard Features

#### Real-time Statistics Panel
- **Active Streams**: Total number of streams currently processing
- **Total Tracks**: Aggregate count of tracked objects across all streams
- **Detections/sec**: Real-time detection throughput
- **Uptime**: Dashboard connection uptime (HH:MM:SS)

#### Stream Management
- **Search**: Filter streams by name with instant search
- **Status Filter**: Show all streams, active only, or idle only
- **Stream Table**: View all streams with:
  - Frame ID (current frame number)
  - Track count (number of tracked objects)
  - FPS (frames per second)
  - Status indicator (Active/Idle)
  - Last update timestamp

#### Detection Visualization
- **Frame Preview**: Live annotated frames with bounding boxes
- **Frame Info**: Resolution and timestamp for each frame
- **Track Details**: Detailed table showing:
  - Track ID (unique identifier)
  - Object class (detection type)
  - Confidence score (color-coded: green >80%, yellow >60%, red <60%)
  - Bounding box coordinates (x1, y1, x2, y2)
  - Object size (width × height in pixels)

### 🔌 API Endpoints

The dashboard exposes:

- **`GET /`** – Interactive web dashboard with real-time updates
- **`GET /api/snapshot`** – JSON snapshot of the latest detections from all streams
- **`WebSocket /ws`** – WebSocket endpoint for real-time event streaming (custom clients)

### 🎨 Enabling Frame Visualization

To view annotated frames with bounding boxes, enable Kafka frame payloads in your pipeline config:

```yaml
kafka:
  enabled: true
  include_frames: true  # Enable frame encoding in Kafka events
```

**Note**: When `include_frames: false` or Kafka is disabled, the dashboard shows stream metadata (frame IDs, track counts) but no visual previews.

### 🌐 WebSocket Protocol

The dashboard uses WebSocket for real-time updates:

```javascript
// Message types
{
  "type": "snapshot",
  "payload": {
    "streams": [...]  // Full state of all streams
  }
}

{
  "type": "event",
  "payload": {
    "stream": "camera-1",
    "frame_id": 1234,
    "tracks": [...],
    "received_at": "2025-11-07T12:34:56Z",
    "frame_jpeg": "data:image/jpeg;base64,..."  // Optional
  }
}
```

### 💡 Tips

- Use the search bar to quickly locate specific streams in large deployments
- Click any stream row to view its detailed track information
- The dashboard automatically reconnects if the connection is lost
- Filter by "Active Only" to focus on streams with current detections
- Track confidence scores are color-coded for quick visual assessment

## Configuration reference

Everything is expressed in YAML (see `config/sample-pipeline.yaml`). Key sections:

- `streams`: declarative list of up to 32 RTSP/RTMP sources. Each item accepts `name`, `url`, optional `target_fps`, warm-up, and reconnect controls. Optional per-stream knobs include `detector_id`, `roi_polygons`, `motion_filter` / `motion_threshold`, `downsample_ratio`, and `adaptive_fps` (`min_target_fps`, `idle_frame_tolerance`).
- `detector`: default backend (`ultralytics` \| `tensorrt`), model/engine path, device, confidence & IoU thresholds, class filtering.
- `detectors`: optional mapping (`id -> detector config`) when different streams need distinct models; reference via `streams[].detector_id`.
- `tracker`: parameters for the IOU tracker (acts as a ByteTrack-compatible shim).
- `kafka`: optional sink to publish events through `aiokafka` (disabled by default).
- `kafka.include_frames`: toggle inline JPEG previews (set to `true` to enable dashboard thumbnails).
- `prometheus`: enable the HTTP `/metrics` endpoint and configure listen address/port.
- `max_concurrent_streams`: hard guard against accidental over-subscription.

`src/realtime_analytics/config.py` declares the dataclasses, validation logic, and defaults. For backend-specific setup guides (Ultralytics YOLO / TensorRT) see `docs/model_integration_cn.md`.

## Simulating cameras with ffmpeg

Need a disposable RTSP source for development? Enable the built-in ffmpeg simulator on any stream (requires `ffmpeg` in `PATH`). The pipeline will launch an ffmpeg subprocess that listens on the stream URL and loops your media file.

```yaml
streams:
  - name: camera-simulated
    url: rtsp://127.0.0.1:8554/camera-sim
    enabled: true
    ffmpeg_simulator:
      enabled: true
      input: data/samples/demo.mp4     # any local file / network source supported by ffmpeg
      listen_host: 0.0.0.0             # optional; defaults to 0.0.0.0
      loop: true                       # replays endlessly
      extra_args:
        - "-vf"
        - "scale=1280:720"             # example: resize before streaming
```

- The stream `url` must be RTSP. The simulator exposes an RTSP server (`-rtsp_flags listen`) bound to `listen_host` (defaults to `0.0.0.0`) and the port encoded in the URL (default 8554).
- Video is re-encoded with `libx264` by default. Set `video_codec: copy` or append to `extra_args` to customise the pipeline.
- Set `audio_enabled: true` to forward audio via `AAC`, or keep the default silent stream.

When the pipeline shuts down it terminates the ffmpeg subprocess automatically. Logs from ffmpeg are forwarded at DEBUG level.

## Module layout

- `video_stream.py` – async wrapper around OpenCV capture, handles warm-up, retries, and frame pacing.
- `detector.py` – Pluggable detector backends (Ultralytics YOLO / TensorRT helpers).
- `tracker.py` – lightweight IOU-based tracker maintaining per-stream state.
- `pipeline.py` – orchestrates stream workers, detector/tracker execution, sink dispatch, and graceful shutdown.
- `sinks/kafka_sink.py` – optional Kafka producer (`aiokafka`) with JSON payloads.
- `telemetry/metrics.py` – Prometheus counters/gauges for frames, detections, active tracks.
- `scripts/run_pipeline.py` – CLI entrypoint, works directly from the repo or via package install.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Video Sources (≤32 streams)                 │
│              RTSP / RTMP / Local Files / FFmpeg Sim             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Video Stream Manager                          │
│         (Async OpenCV, Auto-reconnect, Frame Pacing)            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Detection Pipeline                           │
│        ┌──────────────┐      ┌──────────────┐                   │
│        │   Detector   │──────▶   Tracker    │                   │
│        │ (YOLO/TRT)   │      │  (IOU-based) │                   │
│        └──────────────┘      └──────────────┘                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
   ┌─────────┐    ┌──────────┐    ┌──────────┐
   │  Kafka  │    │Prometheus│    │Dashboard │
   │  Sink   │    │ Metrics  │    │ (WebUI)  │
   └─────────┘    └──────────┘    └──────────┘
```

## 🎯 Use Cases

- **Security & Surveillance**: Monitor multiple security cameras with real-time person/vehicle detection
- **Retail Analytics**: Track customer movement and behavior across store locations
- **Traffic Management**: Monitor traffic flow, count vehicles, detect violations
- **Industrial Monitoring**: Track objects on assembly lines, detect anomalies
- **Smart City**: Integrate multiple camera feeds for urban monitoring
- **Research & Development**: Test and benchmark detection/tracking algorithms

## 🚀 Performance Tips

1. **GPU Acceleration**: Use TensorRT for 2-5x faster inference
2. **Frame Downsampling**: Reduce resolution for non-critical streams
3. **Motion Detection**: Enable adaptive FPS to skip static frames
4. **ROI Filtering**: Process only relevant regions of frames
5. **Batch Processing**: Group frames for efficient GPU utilization
6. **Stream Prioritization**: Assign different models/configs per stream importance

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Framework | FastAPI, Uvicorn |
| Computer Vision | OpenCV |
| Inference Engines | Ultralytics, ONNX Runtime, OpenVINO, TensorRT |
| Object Detection | YOLOv8, YOLOv9, YOLOv10, Custom YOLO models |
| Async I/O | asyncio, aiofiles |
| Messaging | Kafka (aiokafka) |
| Metrics | Prometheus |
| Frontend | Vanilla JavaScript, WebSocket |
| Container | Docker, Docker Compose |
| Configuration | YAML |

## 🚀 Inference Backends

The pipeline supports multiple inference backends for different hardware and performance requirements:

| Backend | Device Support | Speed | Best For |
|---------|---------------|-------|----------|
| **Ultralytics** | CPU, CUDA | Good | Development, prototyping |
| **ONNX Runtime** | CPU, CUDA | Better | Cross-platform deployment |
| **OpenVINO** | CPU, GPU, NPU | Best (Intel) | Intel hardware optimization |
| **TensorRT** | CUDA | Best (NVIDIA) | NVIDIA GPU maximum performance |

### Quick Backend Setup

**Ultralytics YOLO** (Default):
```bash
uv sync --extra detector
```

**ONNX Runtime** (CPU):
```bash
uv sync --extra onnx
```

**ONNX Runtime** (GPU):
```bash
uv sync --extra onnx-gpu
```

**OpenVINO**:
```bash
uv sync --extra openvino
```

**TensorRT**:
```bash
# Requires manual installation of NVIDIA TensorRT
pip install tensorrt pycuda
```

### Configuration Examples

**ONNX Runtime**:
```yaml
detector:
  backend: onnx
  model_path: models/yolov8n.onnx
  device: cuda  # or cpu
  conf_threshold: 0.5
  iou_threshold: 0.45
```

**OpenVINO**:
```yaml
detector:
  backend: openvino
  model_path: models/yolov8n.xml
  device: cpu  # or gpu, auto, npu
  conf_threshold: 0.5
  iou_threshold: 0.45
```

**TensorRT**:
```yaml
detector:
  backend: tensorrt
  model_path: models/yolov8n.engine
  device: cuda
  input_size: [640, 640]
  conf_threshold: 0.5
  iou_threshold: 0.45
  half: true
```

For detailed backend documentation and model conversion guides, see [docs/inference_backends.md](docs/inference_backends.md).

## 🔧 Troubleshooting

### Dashboard shows no streams
- Ensure Kafka is enabled in the pipeline config
- Verify the dashboard is using the same Kafka settings
- Check that the pipeline is running and publishing events

### Streams not connecting
- Verify RTSP/RTMP URLs are accessible
- Check network connectivity and firewall rules
- Review stream logs for connection errors
- Ensure credentials are correct if authentication is required

### Low FPS / High latency
- Enable frame downsampling in stream config
- Use TensorRT for faster inference
- Reduce the number of concurrent streams
- Check CPU/GPU utilization
- Adjust target_fps per stream

### Frames not showing in dashboard
- Set `kafka.include_frames: true` in pipeline config
- Ensure sufficient bandwidth for JPEG frame transmission
- Check browser console for WebSocket errors

## 📚 Additional Resources

- **Ultralytics YOLOv8**: [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
- **TensorRT**: [https://developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt)
- **FastAPI**: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
- **Kafka**: [https://kafka.apache.org/](https://kafka.apache.org/)
- **Prometheus**: [https://prometheus.io/](https://prometheus.io/)

## 🗺️ Roadmap

### Short-term
- [x] Modern web dashboard with real-time updates
- [x] Stream search and filtering
- [x] Performance statistics and metrics
- [ ] Export detection data (CSV, JSON)
- [ ] Alert system for custom detection rules
- [ ] User authentication and access control

### Mid-term
- [ ] TensorRT/DeepStream optimization guides
- [ ] Rule-engine hooks for event-driven actions
- [ ] Complete Docker Compose stack (Kafka + Prometheus + Grafana)
- [ ] Unit and integration tests
- [ ] Performance benchmarking suite
- [ ] Multi-language support for dashboard

### Long-term
- [ ] Advanced analytics (heatmaps, path tracking)
- [ ] Model management and versioning
- [ ] Distributed processing across multiple nodes
- [ ] Cloud deployment guides (AWS, GCP, Azure)
- [ ] Video recording with detection highlights
- [ ] Mobile app for monitoring

## 📄 License

This project is open source. Please check the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📧 Support

For questions or support, please open an issue on the GitHub repository.
