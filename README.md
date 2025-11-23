# 🎥 Realtime Video Analytics - 32 Streams

English | [简体中文](./README_CN.md)

A production-ready, multi-stream (up to 32) real-time video analytics pipeline with AI-powered object detection and tracking, featuring a modern web dashboard for monitoring and visualization.

## Docs & Config quick links
- Live demo (4 RTSP simulators): `docs/local_rtsp_demo.md`
- Full deployment (build, simulators, optional Kafka, pipeline, dashboard, exports): `docs/deployment.md`
- Sample configs: `config/sample-pipeline.yaml` (basic), `config/pipeline-rtsp.yaml` (4 local RTSP), `config/pipeline-sim.yaml` (file-source smoke test)
- Temporal/action example: `sample-temporal-pipeline.yaml`
- Dashboard static assets: `src/realtime_analytics/api/static/`


## ✨ Key Features

### 🔧 Core Pipeline
- **Multi-Stream Processing**: Handle up to 32 concurrent RTSP/RTMP video streams
- **RTSP/RTMP Ingestion**: Asynchronous OpenCV capture with auto-reconnection
- **H.265/HEVC Support**: Full support for H.265 video codec via FFmpeg backend
- **AI Detection**: Multiple inference backends (Ultralytics, ONNX Runtime 1.23.0+, OpenVINO, TensorRT, RKNN)
- **Multiple Model Types**: YOLOv8, YOLOv5, ResNet classification, and temporal models (CNN-LSTM, 3D CNN, ConvGRU) for action recognition
- **Object Tracking**: Lightweight IOU tracker (ByteTrack/DeepSORT compatible)
- **Event Streaming**: Kafka sink with adaptive quality and frame rate limiting
- **Smart Scheduling**: Priority-based stream management with health monitoring
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

#### Quick 4-stream demo
`docs/local_rtsp_demo.md` is the quickest way to see detections (PowerShell/bash, 4 RTSP streams from the bundled video, Docker pipeline).

#### Full deployment
`docs/deployment.md` covers end-to-end steps: build image, start simulators, (optional) Kafka, run pipeline + dashboard, export data.

Mounted volumes:

- `./config` → `/data/config` – place your pipeline YAML here (e.g. `config/pipeline.yaml`).
- `./models` → `/app/models` – store Ultralytics `.pt` or TensorRT `.engine` models.

Environment overrides:

- `PIPELINE_CONFIG` – pipeline service config path, defaults to `/data/config/pipeline.yaml`.
- `DASHBOARD_CONFIG` / `DASHBOARD_PORT` – control dashboard config and port.

> If you need to use TensorRT in the container, ensure the host has a GPU with matching NVIDIA drivers, build from official TensorRT/CUDA base images, and add `--gpus all` at runtime.

## 📝 Advanced Logging

The pipeline and dashboard support enhanced logging features for production debugging and monitoring:

### Logging Options

```bash
# Standard logging (colored console output)
realtime-analytics --config config.yaml --log-level INFO

# Detailed logging with thread/process info
realtime-analytics --config config.yaml --log-format detailed

# JSON logging for machine parsing
realtime-analytics --config config.yaml --log-format json

# Log to file with rotation (10MB x 5 files)
realtime-analytics --config config.yaml --log-file logs/pipeline.log --log-rotate

# Disable colored output (for piping or file output)
realtime-analytics --config config.yaml --no-color

# Combined: detailed format + file + rotation
realtime-analytics --config config.yaml \
  --log-level DEBUG \
  --log-format detailed \
  --log-file /var/log/analytics/pipeline.log \
  --log-rotate
```

### Log Formats

**Standard** (default):
```
2025-11-07 12:34:56 [INFO    ] realtime_analytics.detector | Loading ONNX model 'models/yolov8n.onnx'
```

**Detailed** (with process/thread info):
```
2025-11-07 12:34:56 [INFO    ] [12345:67890] realtime_analytics.detector:__init__:429 | Loading ONNX model
```

**JSON** (machine-readable):
```json
{"time":"2025-11-07 12:34:56","level":"INFO","logger":"realtime_analytics.detector","message":"Loading ONNX model"}
```

### Log Levels

- `DEBUG`: Verbose output including warmup, shape detection, preprocessing details
- `INFO`: Standard operation logs (recommended for production)
- `WARNING`: Important issues that don't stop execution
- `ERROR`: Critical errors that may affect stream processing
- `CRITICAL`: System-level failures

### Dashboard Logging

The dashboard supports the same logging options:

```bash
realtime-analytics-dashboard \
  --config config.yaml \
  --log-level INFO \
  --log-file logs/dashboard.log \
  --log-rotate
```

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

### 🎬 Temporal Video Analysis (Action Recognition)

The system supports temporal models that analyze sequences of frames for action recognition and event detection:

**Supported Temporal Models**:
- **CNN-LSTM**: Combines CNN spatial features with LSTM temporal modeling
- **3D CNN**: 3D convolutions for spatiotemporal feature learning (C3D, I3D, ResNet3D)
- **ConvGRU**: Efficient convolutional GRU for video analysis
- **SlowFast**: Dual-pathway networks for action recognition

**Use Cases**: Security (person falling, fighting), traffic (accidents, violations), behavior analysis

**Example Configuration**:
```yaml
detectors:
  action_detector:
    model_type: "cnn_lstm"  # or "3d_cnn", "conv_gru", "slow_fast"
    backend: "onnxruntime"  # or "openvino"
    model_path: "/models/action_recognition.onnx"
    device: "cuda"

    # Temporal parameters
    sequence_length: 16  # Number of frames in sequence
    sequence_stride: 2   # Sample every Nth frame
    temporal_overlap: 0.5  # Overlap between sequences
    num_action_classes: 400  # Kinetics-400 classes

    # Optional action labels
    action_classes:
      - "walking"
      - "running"
      - "falling"
      # ... more classes

streams:
  - name: "security_camera"
    url: "rtsp://camera/stream"
    target_fps: 10  # Lower FPS for temporal models
    detector_id: "action_detector"
```

📖 **Full Documentation**: See [docs/TEMPORAL_DETECTION.md](docs/TEMPORAL_DETECTION.md) for detailed guide on temporal models, model preparation, optimization strategies, and examples.

📋 **Example Config**: See [sample-temporal-pipeline.yaml](sample-temporal-pipeline.yaml) for complete temporal detection configuration.

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
┌──────────────────────────────────────────────────────────────────┐
│                    Video Sources (up to 32 streams)              │
│             RTSP / RTMP / Local Files / FFmpeg Sim               │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│                  Video Stream Manager                            │
│        (Async OpenCV, Auto-reconnect, Frame Pacing)              │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│                   Detection Pipeline                             │
│       ┌──────────────┐     ┌──────────────┐                     │
│       │  Detector    │─────│  Tracker     │                     │
│       │ (YOLO/TRT)   │     │ (IOU-based)  │                     │
│       └──────────────┘     └──────────────┘                     │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
         ┌───────────────┼───────────────┐
         ↓               ↓               ↓
         ↓               ↓               ↓
   ┌─────────┐   ┌──────────┐   ┌──────────┐
   │ Kafka   │   │Prometheus│   │Dashboard │
   │ Sink    │   │Metrics   │   │(WebUI)   │
   └─────────┘   └──────────┘   └──────────┘
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

The pipeline supports multiple inference backends optimized for different hardware platforms and performance requirements:

| Backend | Device Support | Speed | Best For | Version |
|---------|---------------|-------|----------|---------|
| **Ultralytics** | CPU, CUDA | Good | Development, prototyping | 8.0.0+ |
| **ONNX Runtime** | CPU, CUDA | Better | Cross-platform deployment | 1.23.0+ |
| **OpenVINO** | CPU, GPU, NPU | Best (Intel) | Intel hardware optimization | 2023.0+ |
| **TensorRT** | CUDA | Best (NVIDIA) | NVIDIA GPU maximum performance | 8.6+ |
| **RKNN** | RK3588 NPU | Best (Rockchip) | RK3588 edge devices (up to 6 TOPS) | 2.0.0+ |

### Quick Backend Setup

**Ultralytics YOLO** (Default):
```bash
uv sync --extra detector
```

**ONNX Runtime 1.23.0+** (CPU):
```bash
uv sync --extra onnx
```

**ONNX Runtime 1.23.0+** (GPU):
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

**RKNN (RK3588)**:
```bash
# For x86 development/conversion
uv sync --extra rknn

# For RK3588 runtime (on device)
pip install rknnlite
```

### Configuration Examples

**ONNX Runtime 1.23.0** (with optimizations):
```yaml
detector:
  backend: onnx
  model_path: models/yolov8n.onnx
  device: cuda  # or cpu
  confidence_threshold: 0.5
  iou_threshold: 0.45
  warmup: true  # Recommended for ONNX Runtime 1.23.0+
```

**OpenVINO**:
```yaml
detector:
  backend: openvino
  model_path: models/yolov8n.xml
  device: cpu  # or gpu, auto, npu
  confidence_threshold: 0.5
  iou_threshold: 0.45
```

**TensorRT**:
```yaml
detector:
  backend: tensorrt
  model_path: models/yolov8n.engine
  device: cuda
  input_size: [640, 640]
  confidence_threshold: 0.5
  iou_threshold: 0.45
  half: true
```

**RKNN (RK3588)**:
```yaml
detector:
  backend: rknn  # or rk3588
  model_path: models/yolov8n.rknn
  device: npu  # RK3588 NPU
  input_size: [640, 640]
  confidence_threshold: 0.5
  iou_threshold: 0.45
  warmup: true  # Recommended for NPU warmup
```

For detailed backend documentation, model conversion guides, and optimization tips, see [docs/inference_backends.md](docs/inference_backends.md).

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
