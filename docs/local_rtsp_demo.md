# Local RTSP Demo (Docker + Sample Video)

This guide shows how to spin up 4 simulated RTSP streams from the bundled sample video and run the pipeline inside Docker on Windows. The same approach works on Linux/macOS with the `.sh` script.

## Prerequisites
- Docker Desktop
- FFmpeg in host `PATH` (for the simulator)
- Project files present locally (`data/samples/demo.mp4`, `models/yolo/yolov8n.pt`)

## 1) Build the image
```powershell
cd C:\Users\luke\Desktop\realtime-video-analytics-32streams
docker build -t realtime-video-analytics:latest .
```

## 2) Start simulated RTSP streams (host)
Run in PowerShell and keep the window open:
```powershell
.\scripts\rtsp-multistream.ps1 -Input data\samples\demo.mp4 -Streams 4 -PortStart 8554
```
This publishes four endpoints:
- rtsp://0.0.0.0:8554/stream01
- rtsp://0.0.0.0:8555/stream02
- rtsp://0.0.0.0:8556/stream03
- rtsp://0.0.0.0:8557/stream04

If Windows prompts for firewall access, allow it. You can also open the range explicitly:
```powershell
netsh advfirewall firewall add rule name="RTSP Sim 8554-8560" dir=in action=allow protocol=TCP localport=8554-8560
```

## 3) Use the ready-made config
`config/pipeline-rtsp.yaml` is prewired to the four streams above using `host.docker.internal` and ports 8554–8557.

## 4) Run the pipeline container
```powershell
docker run --rm `
  --add-host host.docker.internal:host-gateway `
  -v "C:\Users\luke\Desktop\realtime-video-analytics-32streams\config:/data/config" `
  -v "C:\Users\luke\Desktop\realtime-video-analytics-32streams\models:/app/models:ro" `
  realtime-video-analytics:latest `
  bash -lc "PIPELINE_CONFIG=/data/config/pipeline-rtsp.yaml python scripts/run_pipeline.py --config /data/config/pipeline-rtsp.yaml --log-level INFO"
```
You should see each stream open and detections/tracks being produced.

## 5) (Optional) docker-compose
Either copy `config/pipeline-rtsp.yaml` to `config/pipeline.yaml`, or set `PIPELINE_CONFIG=/data/config/pipeline-rtsp.yaml`. The compose file already injects `host.docker.internal:host-gateway`.
```powershell
docker compose up --build pipeline
```

## 6) Validate connectivity from inside the image (optional)
```powershell
docker run --rm --add-host host.docker.internal:host-gateway realtime-video-analytics:latest `
  bash -lc "ffprobe -v error -rtsp_transport tcp rtsp://host.docker.internal:8554/stream01 -show_streams -count_frames -read_intervals %+#0.5"
```

## 7) Dashboard (optional)
Kafka is off in this config, but you can still start the UI:
```powershell
docker run --rm -p 8080:8080 `
  --add-host host.docker.internal:host-gateway `
  -v "C:\Users\luke\Desktop\realtime-video-analytics-32streams\config:/data/config:ro" `
  realtime-video-analytics:latest `
  bash -lc "DASHBOARD_CONFIG=/data/config/pipeline-rtsp.yaml docker/run_dashboard.sh"
```
Open http://localhost:8080. Enable Kafka in the config if you need live event streaming.

## 8) Stop
- Ctrl+C in the pipeline container
- Press Enter in the PowerShell window running `rtsp-multistream.ps1` to kill ffmpeg processes
