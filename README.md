# realtime-video-analytics-32streams（中文指南）

这是一个面向多路（最多 32 路）实时视频流的检测与跟踪平台，默认使用 Ultralytics YOLOv8 作为检测器，提供轻量级 IOU 跟踪器、Kafka 事件输出以及 Prometheus 指标上报能力。本文档将帮助你快速启动项目、准备模型文件、配置多流检测并运行整套流水线。

> ⚠️ 当前仓库提供的是参考级的纯 Python 实现，方便本地开发与验证。若要接入 TensorRT / DeepStream，可在后续替换 `detector.py`、`tracker.py` 中的实现。

## 环境与依赖

- Python 3.10+
- OpenCV、NumPy、PyYAML（基础依赖）  
- 可选依赖：
  - `ultralytics`：YOLOv8 推理
  - `aiokafka`：Kafka 异步生产者
  - `prometheus-client`：Prometheus 指标暴露

项目根目录提供了 `pyproject.toml`，可使用 `pip install -e ".[full]"` 安装所有可选组件。

## 快速上手

```bash
# 1. 创建虚拟环境（可选）
python -m venv .venv
source .venv/bin/activate

# 2. 安装依赖（安装所有可选组件）
pip install -e ".[full]"

# 3. 准备配置文件
cp config/sample-pipeline.yaml my-pipeline.yaml
vi my-pipeline.yaml

# 4. 运行流水线
python scripts/run_pipeline.py --config my-pipeline.yaml --log-level INFO
```

生产环境下，可直接安装并使用 CLI 入口：

```bash
pip install realtime-video-analytics-32streams[full]
realtime-analytics --config /etc/analytics/pipeline.yaml
```

## 模型文件如何放置？

1. 默认配置使用 `yolov8n.pt`。你可以从 [Ultralytics](https://github.com/ultralytics/ultralytics) 下载所需模型（如 `yolov8s.pt`、`yolov8m.pt` 等）。  
2. 将模型文件放入自定义目录，例如 `models/yolov8n.pt`。  
3. 在配置文件 `detector.model_path` 中填入模型路径。支持：
   - 绝对路径：`/opt/models/yolov8n.pt`
   - 相对路径：相对于运行命令时的当前工作目录，例如 `models/yolov8n.pt`
4. 如果你已经转换为 TensorRT Engine（例如 `.engine`），可以在 `detector` 段落里指向该文件，并在 `detector.py` 中扩展对应装载逻辑。

示例：

```yaml
detector:
  model_path: models/yolov8s.pt
  device: cuda:0      # auto / cpu / cuda:0 / cuda:1...
  conf_threshold: 0.5
  iou_threshold: 0.45
  classes: null       # 指定类别 ID 列表时仅输出这些类别
  half: true          # 使用半精度（需 GPU 支持）
  warmup: true        # 首次运行进行一次暖机
```

## 多流检测如何配置？

`config/sample-pipeline.yaml` 中的 `streams` 字段定义了所有视频源。每项包含：

- `name`：唯一的流名称（用于日志、指标、Kafka 消息）。
- `url`：RTSP/RTMP 地址，也可以是本地视频文件路径。
- `enabled`：是否启用该流。
- `target_fps`：限制该流的处理帧率；留空则按源实际帧率处理。
- `warmup_seconds`：打开流后等待的秒数，用于避免起播瞬间花帧。
- `reconnect_backoff`：读取失败时的重试间隔。
- `max_retries`：连续失败的最大重试次数，`null` 表示无限重试。

要扩展到 32 路，只需在 `streams` 列表中追加条目。注意：

- 确保 `max_concurrent_streams` ≥ 实际流数量，否则会触发校验错误。
- 32 路同时运行需要相当的硬件资源（GPU 内存、CPU、网络带宽）。建议逐步扩展并监控系统负载。

示例片段：

```yaml
streams:
  - name: camera-entrance
    url: rtsp://192.168.1.10/live
    target_fps: 15
  - name: camera-parking
    url: rtsp://192.168.1.11/live
    target_fps: 12
  - name: camera-warehouse
    url: rtsp://192.168.1.12/live
    enabled: false   # 临时关闭，后期可直接启用
```

## 运行流程概览

1. **VideoStream**（`video_stream.py`）：异步调用 OpenCV 打开 RTSP/RTMP 流，按配置进行暖机、限帧与断线重连。
2. **Detector**（`detector.py`）：懒加载 YOLO 模型完成推理；可在未来替换为 TensorRT/DeepStream 推理引擎。
3. **IOUTracker**（`tracker.py`）：轻量级 IOU 匹配器，维护每路流的跟踪状态。
4. **KafkaSink**（`sinks/kafka_sink.py`）：可选，将每帧的跟踪结果推送到 Kafka topic。
5. **MetricsPublisher**（`telemetry/metrics.py`）：在 `http://host:port/metrics` 暴露 Prometheus 指标：帧数量、检测数量、活跃轨迹数等。
6. **AnalyticsPipeline**（`pipeline.py`）：为每路流启动协程任务，串联检测、跟踪、指标与事件输出，并处理优雅停机。

## Kafka 与 Prometheus

- **Kafka**：在配置中启用 `kafka.enabled` 并填写 `bootstrap_servers`、`topic` 等字段。确保已经安装 `aiokafka` 并有权限访问 Kafka 集群。
- **Prometheus**：默认启用。可将 `prometheus.host` 设置为 `0.0.0.0` 以供外部抓取，`port` 默认为 `9000`。在 Grafana 中添加数据源即可构建监控看板。

## 常见问题

1. **模型加载失败**：确认 `ultralytics` 已安装，且模型路径正确（存在且可读）。  
2. **RTSP 无法打开**：检查网络连通性、流 URL 正确性、账号密码（如有）。  
3. **性能不足**：可调整 `target_fps` 降低处理频率，或在 `detector` 中使用更轻量的模型（如 `yolov8n`）。也可以使用 `half: true` 在 GPU 上启用半精度推理。  
4. **Kafka 报错**：确保 Kafka 服务正常运行，使用 `kafka-console-producer` 进行基本测试，确认 topic 存在。

## 下一步建议

- 将默认 YOLO 推理替换为 TensorRT/DeepStream，实现更低延迟和更高吞吐。
- 引入规则引擎：根据检测/跟踪信息触发业务事件、报警或联动设备。
- 使用 Docker Compose 搭建 Kafka、Prometheus、Grafana 的配套服务。
- 编写自动化测试或模拟视频源，提高上线前的稳定性验证。
- 扩展健康检查和告警机制（例如监控断流、帧率下降）。

如需进一步定制，可直接修改 `src/realtime_analytics` 目录下的各模块，或继承现有类实现自己的处理逻辑。
