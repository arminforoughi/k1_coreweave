# Booster Camera AI Detection System

AI-enhanced camera feed viewer and object detection system for the Booster Camera Bridge with low-confidence detection extraction capabilities.

## Overview

This repository contains two main Python scripts for real-time camera feed processing using YOLO object detection and neural network-based depth estimation:

1. **camera_feed_detection.py** - Web-based AI detection viewer with depth estimation
2. **extract_low_confidence_objects.py** - Extract and save objects with low confidence scores

## Features

### Camera Feed Detection (`camera_feed_detection.py`)

- **Real-time Object Detection** using YOLOv8
- **Neural Network Depth Estimation** with multiple model options:
  - DPT-BEiT-Base-384 (Intel)
  - MiDaS
  - CREStereo (placeholder)
- **Stereo Depth Mapping** using dual camera feeds
- **Web-based Viewer** with live streaming at http://localhost:8080
- **Multiple Camera Views**:
  - Detected objects with bounding boxes
  - Depth map visualization
  - Raw left/right camera feeds

### Low-Confidence Object Extractor (`extract_low_confidence_objects.py`)

- **Automatic Detection** of objects with confidence scores below threshold (default: 0.65)
- **Frame Extraction** - Saves complete frames with low-confidence detections
- **Object Cropping** - Individual crops of each detected object
- **Detailed Logging** - Timestamps and confidence scores for all detections
- **Organized Output** - Structured directory layout for easy review

## Requirements

### Dependencies

```bash
pip install rclpy sensor_msgs cv_bridge opencv-python numpy ultralytics torch torchvision transformers pillow
```

### System Requirements

- ROS 2 (for camera bridge topics)
- CUDA-capable GPU (optional, for faster inference)
- Python 3.8+
- Active camera topics: `/booster_camera_bridge/image_left_raw` and `/booster_camera_bridge/image_right_raw`

## Installation

1. Clone this repository:
```bash
git clone https://github.com/arminforoughi/k1_coreweave.git
cd k1_coreweave
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make scripts executable:
```bash
chmod +x camera_feed_detection.py
chmod +x extract_low_confidence_objects.py
```

## Usage

### Camera Feed Detection with Web Viewer

**Basic usage (monocular):**
```bash
python3 camera_feed_detection.py
```

**Stereo mode with depth estimation:**
```bash
python3 camera_feed_detection.py --stereo --depth-model dpt-beit
```

**Custom settings:**
```bash
python3 camera_feed_detection.py \
    --stereo \
    --host 0.0.0.0 \
    --port 8080 \
    --model yolov8n.pt \
    --depth-model dpt-beit
```

**Available Arguments:**
- `--stereo` - Enable stereo depth estimation
- `--host` - Web server host (default: 0.0.0.0)
- `--port` - Web server port (default: 8080)
- `--model` - YOLO model path (default: yolov8n.pt)
- `--depth-model` - Depth model: dpt-beit, midas, crestereo (default: dpt-beit)

**Access the viewer:**
Open your browser to `http://localhost:8080`

### Low-Confidence Object Extraction

**Basic usage (default threshold 0.65):**
```bash
python3 extract_low_confidence_objects.py
```

**Custom confidence threshold:**
```bash
python3 extract_low_confidence_objects.py --threshold 0.50
```

**Full options:**
```bash
python3 extract_low_confidence_objects.py \
    --threshold 0.65 \
    --model yolov8n.pt \
    --output low_confidence_detections
```

**Available Arguments:**
- `--threshold` - Confidence threshold, save objects below this value (default: 0.65)
- `--model` - YOLO model to use (default: yolov8n.pt)
- `--output` - Output directory path (default: low_confidence_detections)

## Output Structure

### Low-Confidence Detection Output

```
low_confidence_detections/
├── frames/
│   ├── 20260201_143052_123456_frame_3objects.jpg
│   └── 20260201_143055_789012_frame_1objects.jpg
└── crops/
    ├── 20260201_143052_123456_obj0_person_0.543.jpg
    ├── 20260201_143052_123456_obj1_car_0.612.jpg
    └── 20260201_143055_789012_obj0_bicycle_0.489.jpg
```

**File Naming Convention:**
- **Frames:** `YYYYMMDD_HHMMSS_microseconds_frame_Nobjects.jpg`
- **Crops:** `YYYYMMDD_HHMMSS_microseconds_objN_classname_confidence.jpg`

## Camera Topics

Both scripts subscribe to ROS 2 topics from the Booster Camera Bridge:
- `/booster_camera_bridge/image_left_raw` - Left camera feed
- `/booster_camera_bridge/image_right_raw` - Right camera feed (stereo mode only)

Supported image encodings:
- NV12 (default for Jetson cameras)
- BGR8

## YOLO Models

The scripts support any YOLOv8 model:
- `yolov8n.pt` - Nano (fastest, least accurate)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (slowest, most accurate)

Download models from: https://github.com/ultralytics/ultralytics

## Depth Estimation Models

### DPT-BEiT-Base-384 (Recommended)
- High quality depth estimation
- Good for both monocular and stereo
- Requires transformers library

### MiDaS
- Fast inference on edge devices
- Good for real-time applications
- Smaller model size

### CREStereo
- Optimized for stereo pairs
- Currently falls back to MiDaS

## Use Cases

### Camera Feed Detection
- Real-time object tracking and monitoring
- Depth-aware robotics applications
- Visual inspection systems
- Security and surveillance

### Low-Confidence Extraction
- Dataset quality improvement
- Model performance analysis
- Edge case identification
- Training data augmentation
- False positive detection

## Performance Tips

1. **GPU Acceleration:** Ensure CUDA is available for faster inference
2. **Model Selection:** Use smaller YOLO models (yolov8n.pt) for real-time performance
3. **Frame Rate:** Adjust camera feed rate if processing is slow
4. **Depth Model:** MiDaS is faster than DPT-BEiT on edge devices

## Troubleshooting

**No camera feed:**
- Verify ROS 2 topics are publishing: `ros2 topic list`
- Check topic names match: `ros2 topic echo /booster_camera_bridge/image_left_raw`

**YOLO model not found:**
- Download model: `wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt`

**Out of memory errors:**
- Use a smaller YOLO model
- Reduce camera resolution
- Use MiDaS instead of DPT-BEiT for depth

## License

MIT License

## Authors

Developed for the Booster Camera Bridge platform.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss proposed changes.
