# Inference Backend Guide

This guide explains how to use different inference backends (Ultralytics YOLO, ONNX Runtime, OpenVINO, TensorRT, RKNN) for object detection in the realtime video analytics pipeline.

## 📋 Table of Contents

- [Overview](#overview)
- [Backend Comparison](#backend-comparison)
- [Ultralytics YOLO](#ultralytics-yolo)
- [ONNX Runtime](#onnx-runtime)
- [OpenVINO](#openvino)
- [TensorRT](#tensorrt)
- [RKNN (RK3588)](#rknn-rk3588)
- [Model Conversion](#model-conversion)
- [Performance Optimizations](#performance-optimizations)
- [Performance Tips](#performance-tips)

## Overview

The pipeline supports multiple inference backends, each optimized for different hardware platforms:

| Backend | Device Support | Speed | Ease of Use | Best For | Version |
|---------|---------------|-------|-------------|----------|---------|
| **Ultralytics** | CPU, CUDA | Good | ⭐⭐⭐⭐⭐ | Development, Prototyping | 8.0.0+ |
| **ONNX Runtime** | CPU, CUDA | Better | ⭐⭐⭐⭐ | Cross-platform deployment | 1.23.0+ |
| **OpenVINO** | CPU, GPU, NPU | Best (Intel) | ⭐⭐⭐ | Intel hardware optimization | 2023.0+ |
| **TensorRT** | CUDA only | Best (NVIDIA) | ⭐⭐ | NVIDIA GPU optimization | 8.6+ |
| **RKNN** | RK3588 NPU | Best (Rockchip) | ⭐⭐⭐ | RK3588 edge devices | 2.0.0+ |

## Backend Comparison

### Ultralytics YOLO
- **Pros**: Easy to use, PyTorch-based, active development, no conversion needed
- **Cons**: Slower than optimized backends, requires PyTorch
- **Use when**: Rapid development, testing, or limited optimization requirements

### ONNX Runtime 1.23.0+
- **Pros**: Cross-platform, good performance, GPU acceleration, enhanced optimizations
- **Cons**: Requires model conversion, moderate setup complexity
- **Use when**: Deploying across different hardware platforms
- **New in 1.23.0**: Improved memory management, parallel execution, enhanced graph optimizations

### OpenVINO
- **Pros**: Excellent Intel CPU/GPU/NPU performance, comprehensive tooling, auto-optimization
- **Cons**: Intel-focused, requires model conversion
- **Use when**: Deploying on Intel hardware (especially edge devices)
- **Optimizations**: AVX-512, oneDNN, performance hints (LATENCY/THROUGHPUT)

### TensorRT
- **Pros**: Maximum NVIDIA GPU performance, FP16/INT8 quantization
- **Cons**: NVIDIA-only, complex setup, requires compilation
- **Use when**: Maximum performance on NVIDIA GPUs

### RKNN (Rockchip RK3588)
- **Pros**: Optimized for RK3588 NPU (up to 6 TOPS), low power consumption, quantized inference
- **Cons**: Rockchip-only, requires RKNN model conversion, uint8 quantization
- **Use when**: Deploying on RK3588 edge devices (Orange Pi 5, Radxa Rock 5, etc.)
- **Key Features**: Triple-core NPU, automatic quantization, efficient preprocessing

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

## RKNN (RK3588)

RKNN (Rockchip Neural Network) is the inference runtime for Rockchip's RK3588 SoC, featuring a powerful NPU (up to 6 TOPS) designed for edge AI applications.

### Hardware Overview

The RK3588 NPU features:
- **Triple-core NPU**: Up to 6 TOPS of compute power
- **Low power consumption**: Ideal for edge devices
- **Quantized inference**: INT8/INT16 operations for efficiency
- **Popular devices**: Orange Pi 5/5B/5+, Radxa Rock 5B, ArmSoM-W3

### Installation

**For x86 development/model conversion:**
```bash
# Install RKNN Toolkit2 for model conversion
pip install rknn-toolkit2>=2.0.0

# Or with uv
uv sync --extra rknn
```

**For RK3588 device runtime:**
```bash
# Install RKNN Lite (lighter weight, runtime only)
pip install rknnlite
```

### Configuration

```yaml
detector:
  backend: rknn  # or 'rk3588'
  model_path: models/yolov8n.rknn  # RKNN model file
  device: npu  # RK3588 NPU
  input_size: [640, 640]
  conf_threshold: 0.5
  iou_threshold: 0.45
  warmup: true  # Recommended for NPU initialization
```

### Model Conversion

RKNN models must be converted from ONNX format with quantization.

#### Step 1: Export YOLOv8 to ONNX

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.export(format='onnx', imgsz=640, simplify=True)
# Output: yolov8n.onnx
```

#### Step 2: Convert ONNX to RKNN

```python
from rknn.api import RKNN

# Create RKNN object
rknn = RKNN(verbose=True)

# Pre-process config
print('--> Config model')
rknn.config(
    mean_values=[[0, 0, 0]],  # Mean normalization
    std_values=[[255, 255, 255]],  # Std normalization
    target_platform='rk3588',
)

# Load ONNX model
print('--> Loading ONNX model')
ret = rknn.load_onnx(model='yolov8n.onnx')

# Build model with quantization
print('--> Building model')
ret = rknn.build(do_quantization=True, dataset='./dataset.txt')

# Export RKNN model
print('--> Export RKNN model')
ret = rknn.export_rknn('yolov8n.rknn')

print('Done')
```

#### Step 3: Prepare Quantization Dataset

Create `dataset.txt` with paths to calibration images:

```txt
./calibration_data/image1.jpg
./calibration_data/image2.jpg
./calibration_data/image3.jpg
...
```

**Recommendation**: Use 100-500 representative images from your target domain.

### Performance Characteristics

**Advantages:**
- Low power consumption (~5W typical)
- Fast inference (20-40 FPS for YOLOv8n)
- No need for active cooling in most cases
- Cost-effective edge deployment

**Considerations:**
- Quantization may slightly reduce accuracy (typically <2% mAP loss)
- Model conversion required on x86 development machine
- NPU memory limits (typically 1GB)

### Optimization Tips

1. **Use YOLOv8n or YOLOv8s**: Larger models may not fit in NPU memory
2. **Quantization-aware training**: Train models with quantization in mind
3. **Representative calibration data**: Use diverse, domain-specific images
4. **Enable NPU warmup**: Set `warmup: true` in config
5. **Multi-stream optimization**: RK3588 handles 4-8 streams efficiently

### Example Workflow

```bash
# 1. Development machine: Export and convert
python export_rknn.py  # Creates yolov8n.rknn

# 2. Transfer to RK3588 device
scp yolov8n.rknn orangepi5:~/models/

# 3. On RK3588: Install runtime
ssh orangepi5
pip install rknnlite

# 4. Run pipeline
realtime-analytics --config pipeline.yaml --log-level INFO
```

### Supported Models

- YOLOv8n, YOLOv8s (recommended)
- YOLOv5n, YOLOv5s
- YOLOv7-tiny
- Custom YOLO models (with RKNN-compatible architecture)

### Troubleshooting

**Issue**: Model conversion fails

**Solution**: Ensure ONNX model is simplified and opset is compatible:
```python
model.export(format='onnx', imgsz=640, simplify=True, opset=12)
```

**Issue**: Low NPU utilization

**Solution**:
- Enable multi-core NPU: Use `core_mask=RKNN.NPU_CORE_AUTO`
- Optimize input resolution (640x640 or 512x512)
- Reduce batch size to 1

**Issue**: Accuracy drop after quantization

**Solution**:
- Use more calibration images (500+)
- Try hybrid quantization (mix of INT8/INT16)
- Consider quantization-aware training

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

# ONNX → RKNN (for RK3588)
# Use Python RKNN API (see RKNN section above)
python convert_to_rknn.py yolov8n.onnx yolov8n.rknn
```

### Conversion Pipeline Recommendation

```
YOLOv8 (.pt)
    ↓
  ONNX (.onnx) ← Start here for maximum compatibility
    ├→ Keep ONNX (for ONNX Runtime backend)
    ├→ Convert to OpenVINO (.xml + .bin) (for Intel hardware)
    ├→ Convert to TensorRT (.engine) (for NVIDIA GPUs)
    └→ Convert to RKNN (.rknn) (for RK3588 edge devices)
```

## Performance Optimizations

The pipeline includes several performance optimizations for each backend:

### Preprocessing Optimizations

All backends now use optimized preprocessing with:
- **Contiguous arrays**: Better cache performance and memory access
- **Minimized copies**: Reduced memory allocations
- **Efficient dtype conversion**: Combined normalization and type casting
- **Optimized letterboxing**: Fast resize and padding operations

### ONNX Runtime 1.23.0+ Optimizations

The pipeline leverages ONNX Runtime 1.23.0's enhanced features:
- **Enhanced graph optimizations**: All optimization levels enabled
- **Memory pattern optimization**: Reduced memory fragmentation
- **Parallel execution**: Optimized for multi-stream scenarios
- **Provider-specific tuning**: CUDA and CPU providers configured for optimal performance

**CUDA Provider optimizations:**
```python
cuda_options = {
    "device_id": 0,
    "arena_extend_strategy": "kNextPowerOfTwo",
    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB
    "cudnn_conv_algo_search": "EXHAUSTIVE",
    "do_copy_in_default_stream": True,
}
```

**CPU Provider optimizations:**
```python
cpu_options = {
    "intra_op_num_threads": 0,  # Auto-detect
    "inter_op_num_threads": 0,  # Auto-detect
}
```

### OpenVINO Optimizations

OpenVINO backend includes performance hints:
- **CPU mode**: `PERFORMANCE_HINT=LATENCY` for real-time processing
- **GPU/AUTO mode**: `PERFORMANCE_HINT=THROUGHPUT` for maximum FPS
- **Thread management**: Auto-detection of optimal CPU threads
- **Intel optimizations**: AVX-512, oneDNN acceleration

### RKNN Optimizations

RK3588 NPU optimizations:
- **Multi-core NPU**: Automatic core scheduling with `NPU_CORE_AUTO`
- **uint8 preprocessing**: Native format for NPU, no float conversion
- **Zero-copy inference**: Direct NPU memory access
- **Quantized operations**: INT8 inference for maximum efficiency

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

#### RKNN (RK3588)
```yaml
detector:
  backend: rknn
  device: npu
  input_size: [640, 640]  # Or 512 for higher FPS
  warmup: true
```

Optimize for multi-stream:
```python
# Enable multi-core NPU in conversion
rknn.init_runtime(target='rk3588', core_mask=RKNN.NPU_CORE_AUTO)
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
| ONNX Runtime 1.23.0 | RTX 3090 | ~200 | CUDA EP, optimized |
| TensorRT | RTX 3090 | ~300 | FP16 engine |
| Ultralytics | i9-12900K | ~25 | CPU only |
| ONNX Runtime 1.23.0 | i9-12900K | ~40 | CPU, optimized |
| OpenVINO | i9-12900K | ~70 | CPU, AVX-512 |
| RKNN | RK3588 | ~30 | NPU, INT8 quantized |

**Multi-stream performance (YOLOv8n, 640x640):**

| Backend | Hardware | Streams | Total FPS | Notes |
|---------|----------|---------|-----------|-------|
| TensorRT | RTX 3090 | 8 | ~2000 | Batch optimization |
| ONNX Runtime 1.23.0 | RTX 3090 | 8 | ~1400 | Parallel execution |
| OpenVINO | i9-12900K | 8 | ~450 | CPU multi-threading |
| RKNN | RK3588 | 6 | ~150 | Triple-core NPU |

*Benchmarks are approximate and vary by hardware, model, configuration, and workload.*

## Additional Resources

- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [ONNX Runtime 1.23.0 Release Notes](https://github.com/microsoft/onnxruntime/releases/tag/v1.23.0)
- [OpenVINO Documentation](https://docs.openvino.ai/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [RKNN Toolkit2 Documentation](https://github.com/rockchip-linux/rknn-toolkit2)
- [RK3588 NPU Guide](https://wiki.radxa.com/Rock5/guide/rknn)
- [YOLOv8 Model Zoo](https://github.com/ultralytics/ultralytics)
