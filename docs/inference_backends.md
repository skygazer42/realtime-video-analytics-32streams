# Inference Backend Guide

This guide explains how to use different inference backends (Ultralytics YOLO, ONNX Runtime, OpenVINO, TensorRT) for object detection in the realtime video analytics pipeline.

## 📋 Table of Contents

- [Overview](#overview)
- [Backend Comparison](#backend-comparison)
- [Ultralytics YOLO](#ultralytics-yolo)
- [ONNX Runtime](#onnx-runtime)
- [OpenVINO](#openvino)
- [TensorRT](#tensorrt)
- [Model Conversion](#model-conversion)
- [Performance Tips](#performance-tips)

## Overview

The pipeline supports multiple inference backends, each with different strengths:

| Backend | Device Support | Speed | Ease of Use | Best For |
|---------|---------------|-------|-------------|----------|
| **Ultralytics** | CPU, CUDA | Good | ⭐⭐⭐⭐⭐ | Development, Prototyping |
| **ONNX Runtime** | CPU, CUDA | Better | ⭐⭐⭐⭐ | Cross-platform deployment |
| **OpenVINO** | CPU, GPU, NPU | Best (Intel) | ⭐⭐⭐ | Intel hardware optimization |
| **TensorRT** | CUDA only | Best (NVIDIA) | ⭐⭐ | NVIDIA GPU optimization |

## Backend Comparison

### Ultralytics YOLO
- **Pros**: Easy to use, PyTorch-based, active development
- **Cons**: Slower than optimized backends, requires PyTorch
- **Use when**: Rapid development, testing, or limited optimization requirements

### ONNX Runtime
- **Pros**: Cross-platform, good performance, GPU acceleration
- **Cons**: Requires model conversion, moderate setup complexity
- **Use when**: Deploying across different hardware platforms

### OpenVINO
- **Pros**: Excellent Intel CPU/GPU/NPU performance, comprehensive tooling
- **Cons**: Intel-focused, requires model conversion
- **Use when**: Deploying on Intel hardware (especially edge devices)

### TensorRT
- **Pros**: Maximum NVIDIA GPU performance, FP16/INT8 quantization
- **Cons**: NVIDIA-only, complex setup, requires compilation
- **Use when**: Maximum performance on NVIDIA GPUs

## Ultralytics YOLO

### Installation

```bash
# Install Ultralytics detector
uv sync --extra detector
# or
pip install ".[detector]"
```

### Configuration

```yaml
detector:
  backend: ultralytics
  model_path: models/yolov8n.pt  # or yolov8s.pt, yolov8m.pt, etc.
  device: cuda  # or cpu, cuda:0, cuda:1, etc.
  conf_threshold: 0.5
  iou_threshold: 0.45
  half: false  # Set to true for FP16 inference (faster on GPU)
  warmup: true
  classes: null  # or [0, 1, 2] to filter specific classes
```

### Example

```yaml
detector:
  backend: ultralytics
  model_path: models/yolov8n.pt
  device: cuda:0
  conf_threshold: 0.5
  iou_threshold: 0.45
  half: true  # Enable FP16 for faster inference
```

### Model Downloads

YOLOv8 models auto-download on first use:
- `yolov8n.pt` - Nano (fastest, 80 classes)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (most accurate)

## ONNX Runtime

ONNX Runtime provides a cross-platform inference engine with good performance on both CPU and GPU.

### Installation

```bash
# CPU version
uv sync --extra onnx
pip install ".[onnx]"

# GPU version (CUDA required)
uv sync --extra onnx-gpu
pip install ".[onnx-gpu]"
```

### Configuration

```yaml
detector:
  backend: onnx  # or onnxruntime
  model_path: models/yolov8n.onnx
  device: cuda  # or cpu
  input_size: [640, 640]  # Model input dimensions (optional, auto-detected)
  conf_threshold: 0.5
  iou_threshold: 0.45
  half: false  # Note: ONNX models are typically exported as FP32
  warmup: true
```

### Model Export

Export YOLOv8 models to ONNX format:

```python
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Export to ONNX
model.export(
    format='onnx',
    imgsz=640,
    simplify=True,  # Simplify model for better compatibility
    opset=12,  # ONNX opset version
)
# Output: yolov8n.onnx
```

Or using CLI:

```bash
yolo export model=yolov8n.pt format=onnx imgsz=640 simplify=True
```

### GPU Acceleration

ONNX Runtime automatically detects CUDA availability:

```yaml
detector:
  backend: onnx
  model_path: models/yolov8n.onnx
  device: cuda  # Uses CUDAExecutionProvider if available
```

The detector will fall back to CPU if CUDA is not available.

## OpenVINO

OpenVINO is Intel's inference engine optimized for Intel CPUs, integrated GPUs, and Neural Processing Units (NPUs).

### Installation

```bash
uv sync --extra openvino
# or
pip install ".[openvino]"
```

### Configuration

```yaml
detector:
  backend: openvino
  model_path: models/yolov8n.xml  # OpenVINO IR format (.xml + .bin)
  device: cpu  # or GPU, AUTO, NPU
  input_size: [640, 640]  # Optional, auto-detected
  conf_threshold: 0.5
  iou_threshold: 0.45
  warmup: true
```

### Model Export

#### Option 1: Export from YOLOv8 to OpenVINO IR

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# Export to OpenVINO IR format
model.export(
    format='openvino',
    imgsz=640,
    half=False,  # Use FP32 (set to True for FP16 on supported hardware)
)
# Output: yolov8n_openvino_model/ directory with .xml and .bin files
```

#### Option 2: Convert ONNX to OpenVINO

```bash
# Install OpenVINO conversion tools
pip install openvino-dev

# Convert ONNX model to OpenVINO IR
mo --input_model yolov8n.onnx \
   --output_dir models/yolov8n_openvino \
   --input_shape [1,3,640,640] \
   --data_type FP32
```

### Device Selection

OpenVINO supports multiple device targets:

```yaml
# CPU (default, works everywhere)
device: cpu

# Integrated GPU (Intel GPUs)
device: gpu

# Auto device selection (OpenVINO picks best available)
device: auto

# Neural Processing Unit (Intel NPUs in newer laptops)
device: npu
```

### Performance Optimization

For Intel CPUs, OpenVINO typically provides 2-3x speedup over ONNX Runtime.

## TensorRT

TensorRT provides the best performance on NVIDIA GPUs with optimizations like FP16 and INT8 quantization.

### Installation

TensorRT requires manual installation:

```bash
# Install NVIDIA TensorRT
# See: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/

# Install Python bindings
pip install tensorrt
pip install pycuda
```

### Configuration

```yaml
detector:
  backend: tensorrt
  model_path: models/yolov8n.engine  # TensorRT engine file
  device: cuda  # TensorRT only supports CUDA
  input_size: [640, 640]  # Required for dynamic engines
  conf_threshold: 0.5
  iou_threshold: 0.45
  half: true  # Match engine precision (FP16/FP32)
  warmup: true
```

### Model Export

#### Option 1: Export from YOLOv8

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# Export to TensorRT (requires TensorRT installed)
model.export(
    format='engine',
    imgsz=640,
    half=True,  # Use FP16 for faster inference
    device=0,  # GPU device
)
# Output: yolov8n.engine
```

#### Option 2: Build from ONNX

```bash
# Using trtexec (TensorRT CLI tool)
trtexec \
  --onnx=yolov8n.onnx \
  --saveEngine=yolov8n.engine \
  --fp16 \
  --workspace=4096 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:1x3x640x640 \
  --maxShapes=images:1x3x640x640
```

### INT8 Quantization

For maximum performance with minimal accuracy loss:

```bash
trtexec \
  --onnx=yolov8n.onnx \
  --saveEngine=yolov8n_int8.engine \
  --int8 \
  --workspace=4096
```

Note: INT8 may require calibration data for best results.

## Model Conversion

### Quick Reference

```bash
# YOLOv8 → ONNX
yolo export model=yolov8n.pt format=onnx imgsz=640 simplify=True

# YOLOv8 → OpenVINO
yolo export model=yolov8n.pt format=openvino imgsz=640

# YOLOv8 → TensorRT (requires GPU)
yolo export model=yolov8n.pt format=engine imgsz=640 half=True device=0

# ONNX → OpenVINO
mo --input_model yolov8n.onnx --output_dir models/ --data_type FP32

# ONNX → TensorRT
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.engine --fp16
```

### Conversion Pipeline Recommendation

```
YOLOv8 (.pt)
    ↓
  ONNX (.onnx) ← Start here for maximum compatibility
    ├→ Keep ONNX (for ONNX Runtime backend)
    ├→ Convert to OpenVINO (.xml + .bin) (for Intel hardware)
    └→ Convert to TensorRT (.engine) (for NVIDIA GPUs)
```

## Performance Tips

### General

1. **Use FP16**: Enable half-precision on supported hardware for 2x speedup
2. **Warmup**: Always enable warmup to pre-allocate resources
3. **Batch Size**: Keep batch size at 1 for real-time streaming
4. **Input Size**: Use 640x640 for balance, 320x320 for speed, 1280x1280 for accuracy

### Per-Backend Recommendations

#### ONNX Runtime
```yaml
detector:
  backend: onnx
  device: cuda
  half: false  # ONNX models are typically FP32
  warmup: true
```

#### OpenVINO
```yaml
detector:
  backend: openvino
  device: auto  # Let OpenVINO choose best device
  warmup: true
```

Export with FP16 for Intel GPUs:
```python
model.export(format='openvino', half=True)
```

#### TensorRT
```yaml
detector:
  backend: tensorrt
  device: cuda
  half: true  # Use FP16 engine
  warmup: true
```

Build optimized engines:
```bash
trtexec --onnx=model.onnx --saveEngine=model.engine \
  --fp16 --workspace=4096 --verbose
```

### Throughput Optimization

For maximum throughput with multiple streams:

1. **Use smaller models**: `yolov8n` or `yolov8s` for real-time performance
2. **Reduce resolution**: Downsample frames to 640x640 or even 320x320
3. **Enable half-precision**: FP16 on GPU, FP16 IR for OpenVINO
4. **Optimize per-stream**: Use different models for different stream priorities

### Example: High-Throughput Configuration

```yaml
# Fast configuration for 32 concurrent streams
detector:
  backend: onnx  # or openvino, tensorrt
  model_path: models/yolov8n.onnx
  device: cuda
  conf_threshold: 0.5
  iou_threshold: 0.45
  half: false
  warmup: true

streams:
  - name: high-priority-stream
    url: rtsp://camera1
    target_fps: 30
    detector_id: accurate  # Use different detector

  - name: low-priority-stream-1
    url: rtsp://camera2
    target_fps: 10
    downsample_ratio: 0.5  # Process at half resolution

detectors:
  accurate:
    backend: onnx
    model_path: models/yolov8m.onnx  # Larger, more accurate model
    device: cuda
```

## Troubleshooting

### ONNX Runtime

**Issue**: `CUDAExecutionProvider` not available

**Solution**:
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

### OpenVINO

**Issue**: Model loading fails

**Solution**: Ensure both `.xml` and `.bin` files are present:
```yaml
model_path: models/yolov8n.xml  # .bin must be in same directory
```

### TensorRT

**Issue**: Engine built on different GPU fails

**Solution**: Rebuild engine on target GPU - TensorRT engines are device-specific.

**Issue**: Dynamic shapes error

**Solution**: Specify `input_size` in config:
```yaml
detector:
  input_size: [640, 640]
```

## Benchmarks

Performance comparison on sample hardware (YOLOv8n, 640x640):

| Backend | Hardware | FPS (single stream) | Notes |
|---------|----------|---------------------|-------|
| Ultralytics | RTX 3090 | ~120 | PyTorch, FP16 |
| ONNX Runtime | RTX 3090 | ~180 | CUDA EP, FP32 |
| TensorRT | RTX 3090 | ~300 | FP16 engine |
| Ultralytics | i9-12900K | ~25 | CPU only |
| ONNX Runtime | i9-12900K | ~35 | CPU only |
| OpenVINO | i9-12900K | ~70 | CPU optimized |

*Benchmarks are approximate and vary by hardware, model, and configuration.*

## Additional Resources

- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [OpenVINO Documentation](https://docs.openvino.ai/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [YOLOv8 Model Zoo](https://github.com/ultralytics/ultralytics)
