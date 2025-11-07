# 🎥 实时视频分析 - 32 路并发

[English](./README.md) | 简体中文

一个面向生产环境的多路（最多 32 路）实时视频分析流水线，提供 AI 驱动的目标检测与跟踪功能，配备现代化 Web 控制台用于监控和可视化。

## ✨ 核心特性

### 🔧 核心流水线
- **多路流处理**：支持最多 32 路并发的 RTSP/RTMP 视频流
- **RTSP/RTMP 接入**：基于 OpenCV 的异步捕获，支持自动重连
- **H.265/HEVC 支持**：通过 FFmpeg 后端完整支持 H.265 视频编解码
- **AI 检测**：多种推理后端（Ultralytics、ONNX Runtime 1.23.0+、OpenVINO、TensorRT、RKNN）
- **多模型类型支持**：YOLOv8、YOLOv5、ResNet 分类，以及时序模型（CNN-LSTM、3D CNN、ConvGRU）用于动作识别
- **目标跟踪**：轻量级 IOU 跟踪器（兼容 ByteTrack/DeepSORT）
- **事件流**：Kafka 输出，支持自适应质量和帧率限制
- **智能调度**：基于优先级的流管理与健康监控
- **可观测性**：Prometheus 指标导出，支持 Grafana 仪表板
- **灵活配置**：基于 YAML 的配置，支持每路流自定义

### 📊 现代化 Web 控制台
- **实时监控**：WebSocket 实时更新，即时显示流状态
- **交互式界面**：现代化响应式设计，深色主题
- **统计面板**：实时指标（活跃流数、跟踪数、检测率、运行时长）
- **流管理**：按名称或活跃状态搜索和过滤流
- **视觉反馈**：实时带边界框的帧预览
- **跟踪详情**：完整的跟踪信息，包含置信度和坐标
- **性能指标**：FPS 监控和流健康指标

### 🚀 高级功能
- **帧模拟**：内置 FFmpeg 模拟器，无需真实摄像头即可测试
- **ROI 过滤**：基于多边形的感兴趣区域遮罩
- **运动检测**：基于场景活跃度的自适应 FPS
- **帧优化**：可配置的降采样以提升资源效率
- **每路流配置**：不同摄像头可使用不同模型、FPS 目标和 ROI

> ⚠️ 本仓库提供的是可端到端运行的 Python 参考实现。
> TensorRT / DeepStream 优化可通过替换检测器/跟踪器模块进行分层叠加。

## 🆕 最新功能

### 多模型支持
- **YOLOv5**：全面支持 TensorRT、ONNX Runtime、OpenVINO 和 RKNN 后端
- **ResNet 分类**：支持 OpenVINO 和 ONNX Runtime 的 ImageNet 分类模型
- **灵活配置**：通过 `model_type` 参数轻松切换模型架构

### H.265/HEVC 视频支持
- 通过 OpenCV FFmpeg 后端自动支持 H.265/HEVC 编解码
- 硬件解码加速（CUDA、VAAPI、QSV）
- 自动编解码检测和日志记录
- 无需配置更改，透明操作

### 增强的调度系统
- **StreamScheduler**：全局资源协调和负载监控
- **StreamHealth**：每路流健康跟踪和性能指标
- **优先级管理**：基于健康评分的智能流优先级
- **自适应 FPS**：系统负载高时自动降低低优先级流的帧率

### 视频质量优化
- **自适应质量**：基于检测数量动态调整帧质量
  - 0 个检测：基础质量 - 10
  - 1-3 个检测：基础质量
  - 4-10 个检测：基础质量 + 5
  - 10+ 个检测：基础质量 + 10
- **帧率限制**：每路流最多 10 FPS 发送至 Kafka，减少带宽
- **渐进式 JPEG**：更好的流式传输性能
- **WebP 支持**：高质量帧使用 WebP 格式（质量 ≥ 80 时）
- **自动降尺寸**：超过 1920x1080 的大帧自动缩放

> ⚠️ 当前仓库提供的是参考级的纯 Python 实现，方便本地开发与验证。若要接入 TensorRT / DeepStream，可在后续替换 `detector.py`、`tracker.py` 中的实现。

如需了解不同推理后端（Ultralytics、ONNX Runtime、OpenVINO、TensorRT、RKNN）的接入方式，请参阅 `docs/inference_backends.md`。

## 环境与依赖

- Python 3.10+
- OpenCV、NumPy、PyYAML（基础依赖）  
- 可选依赖：
  - `ultralytics`：YOLOv8 推理
  - `aiokafka`：Kafka 异步生产者
  - `prometheus-client`：Prometheus 指标暴露

项目根目录提供了 `pyproject.toml` 以及通过 `uv` 生成的 `pylock.toml`，推荐使用 `uv` 进行环境管理与依赖安装。

## 快速上手（推荐使用 uv）

```bash
# 1. （可选）安装 uv 管理的 Python 版本
uv python install 3.11

# 2. 同步依赖（会在 .venv 下创建虚拟环境并安装本地包及 full 可选项）
uv sync --extra full

# 3. 准备配置文件
cp config/sample-pipeline.yaml my-pipeline.yaml
vi my-pipeline.yaml

# 4. 运行流水线（可使用脚本入口或 CLI）
uv run realtime-analytics --config my-pipeline.yaml --log-level INFO
# 或
uv run python scripts/run_pipeline.py --config my-pipeline.yaml --log-level INFO
```

### 备用方案：pip + virtualenv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[full]"
python scripts/run_pipeline.py --config my-pipeline.yaml --log-level INFO
```

生产环境可在 uv 管理的环境或系统环境中安装并使用 CLI 入口：

```bash
pip install realtime-video-analytics-32streams[full]
realtime-analytics --config /etc/analytics/pipeline.yaml
```

## Docker 部署

```bash
# 构建容器镜像
docker compose build

# 运行检测流水线（默认读取 ./config/pipeline.yaml）
docker compose up pipeline

# 可选：同时启动可视化看板
docker compose up dashboard
```

挂载目录：

- `./config` ➜ `/data/config`（放置你的 YAML 配置，例如 `config/pipeline.yaml`）
- `./models` ➜ `/app/models`（存放 Ultralytics `.pt` 或 TensorRT `.engine` 模型）

环境变量：

- `PIPELINE_CONFIG` 控制流水线在容器内读取的配置路径，默认 `/data/config/pipeline.yaml`
- `DASHBOARD_CONFIG`、`DASHBOARD_PORT` 控制看板引用的配置与监听端口

> 如果需要在容器中启用 TensorRT，引擎应基于 NVIDIA 官方 TensorRT/CUDA 镜像构建，并在运行时加入 `--gpus all`，确保宿主机安装了对应的驱动。

## 可视化看板

项目内置一个基于 FastAPI 的轻量看板，用于实时查看各路视频流的检测概况（帧号、轨迹数量、轨迹详情）。看板通过 Kafka 订阅流水线推送的事件。

```bash
# 安装看板相关依赖（uv）
uv sync --extra dashboard

# 启动看板，沿用流水线配置中的 Kafka 参数
uv run realtime-analytics-dashboard --config my-pipeline.yaml --port 8080

# 浏览器访问
open http://localhost:8080
```

看板提供以下接口：

- `/`：实时表格界面，WebSocket 推送最新的每路流数据。
- `/api/snapshot`：返回当前所有流的检测快照（JSON）。
- `/ws`：WebSocket 端点，可供自定义前端或服务接入。

如需在界面中同步显示带检测框的最新帧，请在流水线配置里开启：

```yaml
kafka:
  enabled: true
  include_frames: true
```

若 Kafka 未启用，看板依然可以启动，但在收到检测事件前不会显示数据。

## 模型配置

### 支持的模型类型

流水线现在支持多种模型架构，通过 `model_type` 参数配置：

#### YOLOv8（默认）
```yaml
detector:
  backend: onnx
  model_type: yolov8
  model_path: models/yolov8n.onnx
  device: cuda
  confidence_threshold: 0.5
  iou_threshold: 0.45
```

#### YOLOv5
```yaml
detector:
  backend: tensorrt  # 或 onnx、openvino
  model_type: yolov5
  model_path: models/yolov5s.engine
  device: cuda
  confidence_threshold: 0.5
  iou_threshold: 0.45
```

**与 YOLOv8 的主要区别：**
- YOLOv5 输出包含显式的 objectness 分数
- 不同的输出张量格式：`[x, y, w, h, objectness, class_probs...]`

#### ResNet 分类模型
```yaml
detector:
  backend: openvino  # 或 onnx
  model_type: resnet
  model_path: models/resnet50.xml
  device: cpu
  confidence_threshold: 0.5
  resnet_num_classes: 1000  # 类别数量（如 ImageNet）
  resnet_top_k: 5  # 返回 top-K 预测结果
```

**分类模型专用选项：**
- `resnet_num_classes`：总类别数（ImageNet 默认为 1000）
- `resnet_top_k`：返回的 top 预测数量（默认为 5）

**注意**：ResNet 返回全帧边界框的检测结果，因为它是分类而非检测任务。

### 模型文件放置

1. 从 [Ultralytics](https://github.com/ultralytics/ultralytics) 下载所需模型（如 `yolov8n.pt`、`yolov5s.pt` 等）
2. 将模型文件放入自定义目录，例如 `models/yolov8n.pt`
3. 在配置文件中设置 `detector.model_path`，支持：
   - 绝对路径：`/opt/models/yolov8n.pt`
   - 相对路径：相对于运行命令时的当前工作目录，例如 `models/yolov8n.pt`
4. 对于优化后的模型格式（TensorRT `.engine`、OpenVINO `.xml`、ONNX `.onnx`），在配置中指定相应的后端和路径

### TensorRT 示例

```yaml
detector:
  backend: tensorrt
  model_type: yolov5  # 或 yolov8
  model_path: models/yolov5s_fp16.engine
  device: cuda
  input_size: [640, 640]
  confidence_threshold: 0.5
  iou_threshold: 0.45
  half: true  # 若引擎以 FP16 构建
```

### OpenVINO 示例

```yaml
detector:
  backend: openvino
  model_type: yolov8  # 或 yolov5、resnet
  model_path: models/yolov8n.xml  # 需要同时存在 .xml 和 .bin
  device: auto  # 或 cpu、gpu、npu
  confidence_threshold: 0.5
  iou_threshold: 0.45
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
- `detector_id`：可选，引用顶层 `detectors` 映射中的某个模型配置，为不同流指定不同权重/引擎。

可选的高级字段：

- `roi_polygons`：仅在指定区域内检测（支持多个多边形）。
- `motion_filter` / `motion_threshold`：启用简单移动检测，静止画面自动跳过推理。
- `downsample_ratio`：在推理前缩放画面，降低分辨率以节省算力（0.1~1.0）。
- `adaptive_fps` + `min_target_fps` + `idle_frame_tolerance`：在长时间无目标时降低检测频率，有目标时恢复。

在 YAML 顶层还可以声明：

- `detector`：默认的检测模型配置。
- `detectors`：可选的字典（`标识符 -> 检测配置`），当不同摄像头需要加载不同模型时使用。

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
2. **Detector**（`detector.py`）：可插拔的推理后端（Ultralytics、TensorRT），统一输出检测结果。
3. **IouTracker**（`tracker.py`）：轻量级 IOU 匹配器，维护每路流的跟踪状态。
4. **KafkaSink**（`sinks/kafka_sink.py`）：可选，将每帧的跟踪结果推送到 Kafka topic。
5. **MetricsPublisher**（`telemetry/metrics.py`）：在 `http://host:port/metrics` 暴露 Prometheus 指标：帧数量、检测数量、活跃轨迹数等。
6. **AnalyticsPipeline**（`pipeline.py`）：为每路流启动协程任务，串联检测、跟踪、指标与事件输出，并处理优雅停机。

## Kafka 与 Prometheus

- **Kafka**：在配置中启用 `kafka.enabled` 并填写 `bootstrap_servers`、`topic` 等字段。确保已经安装 `aiokafka` 并有权限访问 Kafka 集群。
- 若要在看板中查看带检测框的实时图片，可将 `kafka.include_frames` 设置为 `true`，并根据需要调整 `frame_quality`。
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

## 使用 ffmpeg 模拟摄像头

在没有真实摄像头的情况下，可以在任意流上开启 `ffmpeg_simulator`，流水线会自动启动一个 ffmpeg 子进程，将本地/网络视频循环推送为 RTSP 服务（需确保系统已安装 `ffmpeg`）。

```yaml
streams:
  - name: camera-sim
    url: rtsp://127.0.0.1:8554/camera-sim
    enabled: true
    ffmpeg_simulator:
      enabled: true
      input: data/samples/demo.mp4   # 任意 ffmpeg 可读取的来源
      listen_host: 0.0.0.0           # 可选，默认 0.0.0.0
      loop: true                     # 循环播放
      extra_args:
        - "-vf"
        - "scale=1280:720"           # 示例：推流前缩放分辨率
```

- `url` 必须为 RTSP 地址。模拟器会在该端口上以 `-rtsp_flags listen` 方式对外提供服务。
- 默认使用 `libx264` 重编码视频。如需转封装可将 `video_codec` 设置为 `copy`，或通过 `extra_args` 追加自定义参数。
- 如需音频推流，将 `audio_enabled` 设为 `true`（默认使用 AAC）；否则保持静音流。

流水线关闭时会自动回收 ffmpeg 子进程，ffmpeg 输出会以 DEBUG 日志级别打印。
