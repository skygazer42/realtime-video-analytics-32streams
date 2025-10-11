# Kafka 在实时多流检测平台中的作用

## 0. Kafka 是什么？为什么要用？

Kafka 是 Apache 提供的分布式消息流平台，核心能力是以极高吞吐量、低延迟地处理发布/订阅数据。它可以像“实时日志总线”一样，将不同服务产生的流式数据写入主题（Topic），由多个消费者并行读取。典型用途包括：

- **多消费者解耦**：一个生产者可以同时服务监控、告警、BI 等多个下游，不需要彼此了解。
- **削峰填谷**：在检测高峰期用 Kafka 暂存，错峰由消费者慢慢处理，保护后端服务。
- **持久化与回放**：Topic 会持久存储指定时长的消息，方便重新消费或离线复盘。

在我们的多流检测平台里，32 路视频会持续产出大量检测数据。直接让每个消费者都连到检测服务会让系统耦合且难以扩展，而 Kafka 能把这部分流量聚合起来、统一分发给看板、告警或其他实时分析模块，因此是整套架构的“数据总线”。

---

本文档帮助你快速理解项目里 Kafka 的位置、消息格式以及如何启用和调试它。

## 1. 为什么需要 Kafka？

在 32 路检测场景中，检测结果不仅要本地处理，还要在外部系统里做告警、统计或可视化。Kafka 作为高吞吐、可扩展的消息队列，可以把每一帧的检测轨迹实时推送给多个消费者（仪表盘、告警服务、流式计算等）。

## 2. 数据从哪里来？

1. `src/realtime_analytics/pipeline.py` 中的 `StreamWorker` 会不断读取视频帧，并调用模型完成检测与跟踪。
2. 处理完的结果传给 `KafkaSink.send_tracks`（`src/realtime_analytics/sinks/kafka_sink.py`）。这里会：
   - 把每个 `Track`（轨迹）转成可序列化的字典；
   - 可选地对原始帧绘制检测框，并编码成 `data:image/jpeg;base64,...`，方便前端直接展示；
   - 利用 `aiokafka.AIOKafkaProducer` 异步发送到 Kafka Topic。

## 3. 消息长什么样？

默认 JSON 结构如下：

```json
{
  "stream": "camera-1",
  "frame_id": 12345,
  "tracks": [
    {
      "track_id": 7,
      "class_id": 0,
      "confidence": 0.91,
      "bbox_xyxy": [105.2, 88.7, 240.5, 320.1]
    }
  ],
  "frame_jpeg": "data:image/jpeg;base64,...."   // 仅当 kafka.include_frames=true 时存在
}
```

## 4. 消费者是谁？

- 项目自带的 FastAPI 看板（`scripts/run_dashboard.py`）内置 `DetectionConsumer`，会订阅同一个 Kafka Topic，并把收到的数据推送到 WebSocket。
- 你可以在 `/src/realtime_analytics/api/kafka_consumer.py` 查看细节，也可以按需实现自定义消费者，例如写入数据库、触发告警等。

## 5. 如何启用/配置 Kafka？

在 YAML 配置里（示例：`config/sample-pipeline.yaml`）：

```yaml
kafka:
  enabled: true                # 默认 false
  bootstrap_servers: localhost:9092
  topic: analytics.events
  linger_ms: 10                # 批量发送延迟
  max_batch_size: 16384        # 批量大小
  include_frames: true         # 是否附带 JPEG 预览
  frame_quality: 75            # JPEG 压缩质量 (1-100)
```

> 提示：确保已安装 `aiokafka`（`uv sync --extra kafka` 或 `uv sync --extra full`），并且 Kafka Broker 正在运行。

## 6. 启动顺序参考

```bash
# 1. 启动 Kafka（本地或集群）
# 2. 启动多流检测流水线
uv run realtime-analytics --config my-pipeline.yaml

# 3. （可选）启动看板订阅 Kafka 并可视化
uv run realtime-analytics-dashboard --config my-pipeline.yaml --port 8080
```

打开 `http://localhost:8080` 后，在配置中打开 `include_frames` 就能看到带检测框的实时缩略图；否则也可以仅查看轨迹列表。

## 7. 常见问题排查

| 情况 | 排查建议 |
|------|----------|
| 生产者启动报错 “aiokafka not installed” | 安装 `aiokafka`；确认虚拟环境正确。 |
| 看板没有数据 | 检查 `kafka.enabled`、Topic 名称、Kafka 服务是否可达。 |
| 带图片的消息过大 | 调整 `frame_quality`、 `include_frames: false`、或者只对关键流开帧预览。 |

理解 Kafka 流程后，你可以很方便地在 Topic 上追加新的消费者，实现告警、BI 分析、离线存储等能力。
