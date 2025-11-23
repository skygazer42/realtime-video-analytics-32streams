# Deployment Guide (Docker demo + optional RTSP)

> TL;DR: `docker compose -f docker-compose-full.yaml up -d` then open http://localhost:8080

## Prerequisites
- Docker Desktop
- Repo cloned（包含 `data/samples/demo.mp4`、`models/yolo/yolov8n.pt`）

## A. 快速演示（默认用本地文件，不依赖 RTSP）
1) 构建并启动
```powershell
cd C:\Users\luke\Desktop\realtime-video-analytics-32streams
docker compose -f docker-compose-full.yaml up -d --build
```
2) 访问前端：http://localhost:8080
3) 关键帧落盘：每路流每 5 分钟自动保存一张带检测框的 JPEG 到 `outputs/<stream>/`（主机目录），文件名形如 `<timestamp>_frame<id>.jpg`。

配置要点（已写好）：
- `config/pipeline-full.yaml` 当前使用本地 `demo.mp4` 两路流，Kafka/Prometheus 开启。
- `docker-compose-full.yaml` 映射 `./outputs:/data/outputs` 以保存关键帧。

## B. 切换到 RTSP 模拟流（可选）
1) 宿主起 ffmpeg 推流（保持窗口开着）：
```powershell
./scripts/rtsp-multistream.ps1 -Input data\samples\demo.mp4 -Streams 4 -PortStart 8554 -ListenHost 0.0.0.0
```
生成端点：8554/8555/8556/8557 对应 stream01~04。
2) 修改 `config/pipeline-full.yaml` 中每路 `url` 为 `rtsp://host.docker.internal:8554/stream01`（其余类推），并设 `ffmpeg_simulator.enabled: false`。
3) 重启管线：`docker compose -f docker-compose-full.yaml restart pipeline`
4) 若连接被拒：放行防火墙端口 8554-8557，并确保 compose 的 `extra_hosts: host.docker.internal:host-gateway` 存在。

## C. 手工运行（不想用 compose）
- Pipeline：
```powershell
docker run --rm --add-host host.docker.internal:host-gateway `
  -v "$PWD/config:/data/config" -v "$PWD/models:/app/models:ro" -v "$PWD/data:/app/data:ro" -v "$PWD/outputs:/data/outputs" `
  realtime-video-analytics:latest \
  python scripts/run_pipeline.py --config /data/config/pipeline-full.yaml --log-level INFO
```
- Dashboard：
```powershell
docker run --rm -p 8080:8080 --add-host host.docker.internal:host-gateway `
  -v "$PWD/config:/data/config:ro" `
  realtime-video-analytics:latest \
  python scripts/run_dashboard.py --config /data/config/pipeline-full.yaml --port 8080 --kafka-bootstrap kafka:9092 --kafka-topic analytics.events
```
(若单机跑，无 Kafka，可加 `--no-kafka`，但前端实时事件需 Kafka。)

## D. 导出接口
- JSON: `http://localhost:8080/api/export/json`
- CSV : `http://localhost:8080/api/export/csv`
- 过滤流：`?streams=demo-stream-01,demo-stream-02`

## E. 排错速查
- 前端“等待检测数据”：看 `docker compose -f docker-compose-full.yaml logs -f pipeline`，确认无 “Unable to open stream”；Kafka 运行中。
- RTSP 连接拒绝：ffmpeg 推流是否在跑？端口是否被防火墙拦截？容器是否能解析 host.docker.internal？
- 关键帧未生成：等待 5 分钟；检查 `outputs/` 是否有文件；需更短间隔可调 `SNAPSHOT_INTERVAL`（`pipeline.py`）。
- 占用 8554 的进程：在宿主 `netstat -ano | findstr 8554`，必要时关闭其他 ffmpeg。

## F. 清理
```powershell
docker compose -f docker-compose-full.yaml down
Remove-Item outputs -Recurse -Force   # 如需清空快照
```
