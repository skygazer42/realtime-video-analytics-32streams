# Deployment Guide (Local Demo + Docker)

This guide walks through building the image, spinning up simulated RTSP streams, running the analytics pipeline and dashboard, and exporting live data.

## Prerequisites
- Docker Desktop
- FFmpeg in host `PATH`
- Project checked out locally (contains `data/samples/demo.mp4`, `models/yolo/yolov8n.pt`)

## 1) Build the image
```powershell
cd C:\Users\luke\Desktop\realtime-video-analytics-32streams
docker build -t realtime-video-analytics:latest .
```

## 2) Start local RTSP simulators (Windows PowerShell)
Keep this window open while testing.
```powershell
Stop-Process -Name ffmpeg -ErrorAction SilentlyContinue
.\scripts\rtsp-multistream.ps1 -Input data\samples\demo.mp4 -Streams 4 -PortStart 8554
```
Ports 8554–8557 will host:
- rtsp://0.0.0.0:8554/stream01
- rtsp://0.0.0.0:8555/stream02
- rtsp://0.0.0.0:8556/stream03
- rtsp://0.0.0.0:8557/stream04

If prompted, allow Windows Firewall for these ports (8554-8560).

## 3) Pipeline config (real-time + Kafka)
Use `config/pipeline-rtsp.yaml` (already wired to the simulators):
```yaml
streams:
  - { name: cam01, url: rtsp://host.docker.internal:8554/stream01, target_fps: 12 }
  - { name: cam02, url: rtsp://host.docker.internal:8555/stream02, target_fps: 12 }
  - { name: cam03, url: rtsp://host.docker.internal:8556/stream03, target_fps: 12 }
  - { name: cam04, url: rtsp://host.docker.internal:8557/stream04, target_fps: 12 }
kafka:
  enabled: true
  bootstrap_servers: host.docker.internal:9092
  topic: analytics.events
  include_frames: false
prometheus:
  enabled: false
detector:
  model_path: /app/models/yolo/yolov8n.pt
  device: cpu
  backend: ultralytics
  warmup: false
```
(Set `enabled: false` if Kafka is not available, but live dashboard events require Kafka.)

## 4) Run the analytics pipeline
```powershell
docker run --rm --add-host host.docker.internal:host-gateway `
  -v "C:\Users\luke\Desktop\realtime-video-analytics-32streams\config:/data/config" `
  -v "C:\Users\luke\Desktop\realtime-video-analytics-32streams\models:/app/models:ro" `
  realtime-video-analytics:latest `
  bash -lc "PIPELINE_CONFIG=/data/config/pipeline-rtsp.yaml python scripts/run_pipeline.py --config /data/config/pipeline-rtsp.yaml --log-level INFO"
```

## 5) Run the dashboard
```powershell
docker run --rm -p 8080:8080 --add-host host.docker.internal:host-gateway `
  -v "C:\Users\luke\Desktop\realtime-video-analytics-32streams\config:/data/config:ro" `
  realtime-video-analytics:latest `
  bash -lc "DASHBOARD_CONFIG=/data/config/pipeline-rtsp.yaml docker/run_dashboard.sh"
```

Open http://localhost:8080/  
- Top live bar shows connection status, last update, stream/track counts.  
- Export button or hotkey `E` opens export modal (can close via X, overlay click, or `Esc`).

## 6) Direct export endpoints
- JSON: `http://localhost:8080/api/export/json`
- CSV : `http://localhost:8080/api/export/csv`
- Filter streams: append `?streams=cam01,cam02`
Exports include: stream, frame_id, received_at, track_id, class_id, confidence, bbox (x1,y1,x2,y2), action_label, temporal_score.

## 7) Compose (optional)
If you prefer docker-compose, ensure compose services include:
```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```
Then:
```powershell
docker compose up --build pipeline
docker compose up --build dashboard
```

## 8) Troubleshooting
- Streams unreachable: verify ffmpeg simulators running and ports 8554-8557 open; use `netstat -ano | findstr 8554`.
- Container to host: ensure `--add-host host.docker.internal:host-gateway` is set.
- Blank dashboard: Kafka disabled or no events yet; enable Kafka and wait for detections.
- Stuck modal: press `Esc` or click overlay (fixed in current build).

## 9) Assets and notes
- Favicon is served from `/favicon.ico` (built-in 1×1 png).
- Static design page remains at `/static/modern-dashboard.html`; live data UI is `/`.
