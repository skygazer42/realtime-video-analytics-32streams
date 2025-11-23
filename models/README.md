# 模型存放说明

建议将权重与引擎文件统一放在 `models/` 目录下，方便配置引用与版本管理。

目录结构示例：

```
models/
├── yolo/
│   ├── yolov8n.pt           # Ultralytics 默认权重
│   └── custom.pt            # 自训练的 YOLO 权重
└── tensorrt/
    ├── yolov8s_fp16.engine  # TensorRT 引擎（FP16）
    └── yolov8m_int8.engine  # TensorRT 引擎（INT8）
```

对应的配置示例：

```yaml
# Ultralytics 推理
detector:
  backend: ultralytics
  model_path: models/yolo/yolov8n.pt

# TensorRT 推理
# detector:
#   backend: tensorrt
#   model_path: models/tensorrt/yolov8s_fp16.engine
#   input_size: [640, 640]
```

> 注意：.engine 文件通常与 GPU、TensorRT 版本强相关，建议在生成时记录使用的环境信息。
