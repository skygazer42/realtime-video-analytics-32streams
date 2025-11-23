# 检测模型接入指南

本文介绍如何在实时多流检测平台中接入 Ultralytics YOLO（默认）与 TensorRT 引擎后端，满足从快速验证到高性能部署的不同需求。

---

## 1. 总体思路

配置文件 `detector.backend` 控制使用哪种推理后端：

```yaml
detector:
  backend: ultralytics   # ultralytics | tensorrt
  model_path: yolov8n.pt
  device: auto           # auto/cpu/cuda:0/...
  confidence_threshold: 0.5
  iou_threshold: 0.45
```

流水线会通过 `create_detector` 工厂自动创建对应的检测器，输出结构化的 `Detection` 对象（统一包含 `stream_name`、`frame_id`、`class_id`、`confidence`、`bbox_xyxy`）。

`detector.py` 内置了三个实现：

| backend       | 依赖                     | 说明 |
|---------------|--------------------------|------|
| `ultralytics` | `pip install ultralytics` | 默认使用 YOLOv8 Python API；适合快速验证与 GPU 推理。 |
| `tensorrt`    | NVIDIA TensorRT + pycuda  | 适合追求极致性能的 GPU 推理（需要预先生成 `.engine`）。 |

---

## 2. Ultralytics YOLO（默认）

1. 安装依赖：
   ```bash
   uv sync --extra detector
   # 或
   pip install ultralytics
   ```
2. 配置示例：
   ```yaml
   detector:
     backend: ultralytics
     model_path: models/yolov8n.pt
     device: cuda:0       # auto / cpu / cuda:0 ...
     confidence_threshold: 0.5
     iou_threshold: 0.45
     half: true           # GPU 上可选用半精度
     warmup: true
   ```
3. 支持直接加载 Ultralytics 官方模型或用户 finetune 后的 `.pt` 文件；也可以替换为 YOLOv5/v8/v9 系列。

---

## 3. TensorRT

1. 安装依赖（需 NVIDIA 官方包）：
   ```bash
   # 示例（版本号请根据本地环境调整）
   pip install tensorrt==8.6.0 pycuda
   ```
   > `pycuda.autoinit` 会在运行时初始化 CUDA 上下文，无需手动创建。

2. 准备 `.engine`：通常需要先将 ONNX 转为 TensorRT Engine，例如：
   ```bash
   trtexec --onnx=models/yolov8s.onnx \
           --saveEngine=models/yolov8s_fp16.engine \
           --fp16 --workspace=2048
   ```

3. 配置示例：
   ```yaml
   detector:
     backend: tensorrt
     model_path: models/yolov8s_fp16.engine
     input_size: [640, 640]        # 必须设置静态尺寸
     confidence_threshold: 0.45
     iou_threshold: 0.5
     half: true                    # 若 engine 采用 FP16

    detectors:
      parking:
        backend: tensorrt
        model_path: models/tensorrt/parking.engine
        input_size: [640, 640]

    streams:
      - name: camera-parking
        detector_id: parking
   ```

4. 引擎要求：
   - 当前实现假设单输入单输出，输出结构与 YOLO ONNX 类似。
   - 如果引擎仍是动态 shape，需要在代码或导出阶段固定输入尺寸，否则会抛出异常。
   - 若输出张量结构不同，可在 `TensorRTDetector._postprocess` 处自定义解析。

---

## 4. 自定义后端扩展

想接入其他推理框架（如 OpenVINO、TVM 等），可参考以下步骤：

1. 在 `detector.py` 中实现新的 `BaseDetector` 子类，负责：
   - 预处理（resize/letterbox）
   - 推理（调用你自己的 runtime）
   - 后处理（阈值、NMS、坐标映射）
2. 在 `create_detector` 工厂中注册你的后端标识。
3. 在配置文件里新增 backend 名称，并按需扩展配置字段。

这样流水线、Kafka 导出和 UI 查看都可以复用，无需改动其他模块。

---

## 5. 常见问题

| 问题 | 解决方案 |
|------|----------|
| TensorRT 启动报错 | 检查 CUDA 驱动版本、TensorRT 版本以及 engine 是否与当前 GPU 匹配；确保 `input_size` 与 engine 一致。 |
| 检测框漂移 | 通常是导出模型输出格式不同，可在 `_postprocess` 中调整坐标解析逻辑。 |
| 想在多后端间切换 | 修改 `detector.backend` 并重启流水线即可，配合虚拟环境安装对应依赖。 |

---

通过以上配置，可以在同一个实时检测平台中灵活切换模型推理后端，满足调试、生产不同阶段的性能与部署需求。

> 如果不同流需要不同模型，可在顶层 `detectors` 中定义多个配置，并在 `streams[].detector_id` 中引用，示例见下文 TensorRT 部分。
