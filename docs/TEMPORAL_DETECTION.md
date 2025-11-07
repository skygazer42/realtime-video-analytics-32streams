# Temporal Video Detection - CNN-LSTM, 3D CNN, ConvGRU

This document describes the temporal video analysis capabilities added to the realtime video analytics system.

## Overview

Temporal detectors analyze sequences of video frames to recognize actions, events, and temporal patterns that cannot be detected from single frames. This is useful for:

- **Action Recognition**: Detecting activities like walking, running, jumping, etc.
- **Event Detection**: Identifying events that occur over time (e.g., person falling, vehicle turning)
- **Behavior Analysis**: Understanding complex behaviors requiring temporal context
- **Video Classification**: Categorizing video clips into action categories

## Supported Models

### 1. CNN-LSTM (Convolutional LSTM)

**Architecture**: Combines CNN for spatial feature extraction with LSTM for temporal modeling.

**Use Cases**:
- Action recognition in videos
- Gesture recognition
- Video event detection

**Configuration**:
```yaml
detector:
  model_type: "cnn_lstm"
  backend: "onnxruntime"  # or "openvino"
  model_path: "/path/to/cnn_lstm_model.onnx"
  sequence_length: 16  # Number of frames
  sequence_stride: 2   # Sample every Nth frame
  temporal_overlap: 0.5  # Overlap between sequences
  num_action_classes: 400
```

**Input Format**: `[batch, time, channels, height, width]`

### 2. 3D CNN (3D Convolutional Network)

**Architecture**: Uses 3D convolutions to process spatial and temporal dimensions simultaneously.

**Use Cases**:
- Action recognition
- Video classification
- Spatiotemporal pattern detection

**Configuration**:
```yaml
detector:
  model_type: "3d_cnn"
  backend: "onnxruntime"
  model_path: "/path/to/3d_cnn_model.onnx"
  sequence_length: 16
  sequence_stride: 1
  input_size: [112, 112]  # Smaller resolution for 3D CNNs
```

**Input Format**: `[batch, channels, time, height, width]` (NCTHW)

**Popular Models**: C3D, I3D, ResNet3D, SlowFast

### 3. ConvGRU (Convolutional GRU)

**Architecture**: Similar to CNN-LSTM but uses GRU (Gated Recurrent Unit) which is more efficient.

**Advantages**:
- Fewer parameters than LSTM
- Faster training and inference
- Competitive performance

**Configuration**:
```yaml
detector:
  model_type: "conv_gru"
  backend: "openvino"
  model_path: "/path/to/convgru_model.onnx"
  sequence_length: 10  # Can use shorter sequences
  temporal_overlap: 0.5
```

### 4. SlowFast Networks

**Architecture**: Dual-pathway network with slow pathway for spatial semantics and fast pathway for motion.

**Configuration**:
```yaml
detector:
  model_type: "slow_fast"
  backend: "onnxruntime"
  model_path: "/path/to/slowfast_model.onnx"
  sequence_length: 32  # Longer sequences
  temporal_overlap: 0.5
```

## Configuration Parameters

### Required Parameters

- `model_type`: One of `cnn_lstm`, `3d_cnn`, `conv_gru`, `slow_fast`
- `backend`: `onnxruntime` or `openvino`
- `model_path`: Path to ONNX or OpenVINO model file
- `sequence_length`: Number of frames in each sequence (typically 8-32)

### Optional Parameters

- `sequence_stride`: Stride between frames (default: 1)
  - `1`: Use every frame
  - `2`: Use every 2nd frame (reduces computation by 50%)
  - Higher values reduce computation but may miss temporal details

- `temporal_overlap`: Overlap between consecutive sequences (default: 0.5)
  - `0.0`: No overlap (process each frame once)
  - `0.5`: 50% overlap (smoother temporal predictions)
  - `0.75`: 75% overlap (very smooth but more computation)

- `temporal_pooling`: Method to aggregate predictions (default: "avg")
  - `avg`: Average pooling across time
  - `max`: Max pooling across time
  - `last`: Use only last frame's prediction

- `num_action_classes`: Number of action classes (default: 400 for Kinetics)
- `action_classes`: List of action class names (optional, for better logging)
- `input_size`: Input spatial resolution `[H, W]`
- `confidence_threshold`: Confidence threshold for predictions (default: 0.5)

## Model Preparation

### Converting PyTorch Models to ONNX

```python
import torch
import torch.onnx

# Example: CNN-LSTM model
model = CNNLSTMModel(num_classes=400)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Create dummy input: [batch, time, channels, height, width]
dummy_input = torch.randn(1, 16, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "cnn_lstm_model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch'},
        'output': {0: 'batch'}
    },
    opset_version=17
)
```

### Converting to OpenVINO

```bash
# Convert ONNX to OpenVINO IR
mo --input_model cnn_lstm_model.onnx \
   --output_dir openvino_models/ \
   --data_type FP32

# For FP16 (faster on Intel GPUs)
mo --input_model cnn_lstm_model.onnx \
   --output_dir openvino_models/ \
   --data_type FP16
```

## Pretrained Models

### Kinetics-400 Dataset

Popular dataset for action recognition with 400 action classes.

**Available Models**:
- CNN-LSTM: [torchvision models](https://pytorch.org/vision/stable/models.html)
- I3D: [Inception-3D](https://github.com/deepmind/kinetics-i3d)
- SlowFast: [facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast)
- ResNet3D: [MMAction2](https://github.com/open-mmlab/mmaction2)

### UCF-101 Dataset

101 action categories (smaller dataset, good for testing).

### Custom Models

You can train custom temporal models for specific use cases:
- Security: Person falling, fighting, suspicious behavior
- Traffic: Lane changing, U-turn, accident detection
- Retail: Customer actions, shelf interaction

## Performance Considerations

### Computational Cost

Temporal models are more computationally expensive than single-frame models:

| Model Type | Relative Cost | Recommended FPS | GPU Memory |
|------------|---------------|-----------------|------------|
| YOLO (single-frame) | 1x | 30 FPS | 2GB |
| CNN-LSTM | 8-16x | 5-10 FPS | 4-6GB |
| 3D CNN | 10-20x | 5-8 FPS | 6-8GB |
| ConvGRU | 6-12x | 8-12 FPS | 4-5GB |
| SlowFast | 15-30x | 3-8 FPS | 8-10GB |

### Optimization Strategies

1. **Reduce sequence_length**: Use shorter sequences (e.g., 8-12 frames instead of 16-32)
2. **Increase sequence_stride**: Sample every 2nd or 3rd frame
3. **Reduce temporal_overlap**: Use less overlap (e.g., 0.25 instead of 0.5)
4. **Lower input_size**: Use smaller spatial resolution (e.g., 112x112 instead of 224x224)
5. **Use FP16**: Enable half-precision for 2x speedup on GPUs
6. **Lower target_fps**: Process fewer frames per second from video stream

### Example Optimized Configuration

```yaml
detector:
  model_type: "cnn_lstm"
  backend: "onnxruntime"
  device: "cuda"
  half: true  # FP16 precision

  # Aggressive optimization
  sequence_length: 8  # Reduced from 16
  sequence_stride: 3  # Sample every 3rd frame
  temporal_overlap: 0.25  # Reduced overlap
  input_size: [112, 112]  # Smaller resolution

stream:
  target_fps: 9  # Lower FPS (9 frames = 3 used per sequence)
```

## Output Format

Temporal detectors return `TemporalDetection` objects with additional fields:

```python
@dataclass
class TemporalDetection(Detection):
    stream_name: str
    frame_id: int  # Last frame ID in sequence
    class_id: int
    confidence: float
    bbox_xyxy: tuple  # Full frame for action recognition

    # Temporal-specific fields
    action_label: str | None  # Human-readable action name
    temporal_score: float  # Temporal confidence
    sequence_start_frame: int  # First frame in sequence
    sequence_end_frame: int  # Last frame in sequence
```

## Examples

### Example 1: Action Recognition on Security Camera

```yaml
streams:
  - name: "lobby_camera"
    url: "rtsp://camera1/stream"
    target_fps: 10
    detector_id: "action_detector"

detectors:
  action_detector:
    model_type: "cnn_lstm"
    backend: "onnxruntime"
    model_path: "/models/action_recognition.onnx"
    device: "cuda"

    sequence_length: 16
    sequence_stride: 2
    temporal_overlap: 0.5
    num_action_classes: 10  # Custom classes

    action_classes:
      - "walking"
      - "running"
      - "standing"
      - "sitting"
      - "falling"
      - "fighting"
      - "normal_activity"
      - "suspicious_behavior"
      - "package_delivery"
      - "maintenance"
```

### Example 2: Traffic Event Detection

```yaml
streams:
  - name: "intersection_cam"
    url: "rtsp://traffic_cam/stream"
    target_fps: 15
    detector_id: "traffic_detector"

detectors:
  traffic_detector:
    model_type: "3d_cnn"
    backend: "onnxruntime"
    model_path: "/models/traffic_events.onnx"
    device: "cuda"
    half: true

    sequence_length: 12
    sequence_stride: 1
    temporal_overlap: 0.5
    input_size: [112, 112]

    num_action_classes: 8
    action_classes:
      - "normal_traffic"
      - "vehicle_turning_left"
      - "vehicle_turning_right"
      - "vehicle_stopping"
      - "pedestrian_crossing"
      - "accident"
      - "traffic_violation"
      - "congestion"
```

## Integration with Existing Pipeline

Temporal detectors integrate seamlessly with the existing pipeline:

1. **Frame Buffering**: Automatically managed per stream
2. **Detections**: Compatible with existing tracker and output systems
3. **Metrics**: Standard metrics (FPS, latency) apply
4. **API**: Same REST API and WebSocket interface

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce `sequence_length`, increase `sequence_stride`, or use smaller `input_size`

### Issue: Low FPS

**Solution**:
- Enable `half: true` for FP16 precision
- Reduce `temporal_overlap`
- Lower stream `target_fps`

### Issue: Poor Detection Accuracy

**Solution**:
- Increase `sequence_length` for more temporal context
- Reduce `sequence_stride` to capture more frames
- Adjust `confidence_threshold`

### Issue: Model Loading Fails

**Solution**:
- Verify model path exists
- Check model format matches backend (ONNX for onnxruntime, .xml/.bin for OpenVINO)
- Ensure model input shape matches `sequence_length` and `input_size`

## References

- [Kinetics Dataset](https://deepmind.com/research/open-source/kinetics)
- [UCF-101 Dataset](https://www.crcv.ucf.edu/data/UCF101.php)
- [MMAction2 Toolbox](https://github.com/open-mmlab/mmaction2)
- [SlowFast Networks](https://github.com/facebookresearch/SlowFast)
- [ONNX Runtime](https://onnxruntime.ai/)
- [OpenVINO Toolkit](https://docs.openvino.ai/)
