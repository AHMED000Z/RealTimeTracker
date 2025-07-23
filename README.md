# RealTimeTracker

A real-time object detection and tracking application using YOLOv8 and Ultralytics built-in tracking capabilities.

## Implementation Details

### Architecture

The application is built using an object-oriented approach with a `RealTimeTracker` class that encapsulates all tracking functionality. This design eliminates global variables and provides clean state management with comprehensive threading support for optimal performance.

### Object Detection

- **Model**: YOLOv8s (small variant) for optimal balance between speed and accuracy
- **Detection Filter**: Excludes person class to focus on objects only as per task requirements
- **Confidence Threshold**: 0.5 (configurable)
- **Image Size**: 416px for faster inference
- **IoU Threshold**: 0.5 for non-maximum suppression

### Object Tracking

- **Algorithm**: BoT-SORT (Byte Track + OCSort) via Ultralytics built-in tracking
- **Configuration**: Custom `botsort.yaml` with optimized parameters for fast-moving objects
- **Track Buffer**: 30 frames for persistence during occlusions
- **Re-Identification**: Enabled with proximity threshold of 0.3
- **Match Threshold**: 0.6 for robust tracking
- **Global Motion Compensation**: Sparse optical flow for camera movement handling

### Performance Optimizations

- **Threading Architecture**: Separate detection thread to prevent UI blocking
- **Frame Queue**: Asynchronous processing with configurable frame skipping
- **Camera Settings**: 640x480 resolution at 30 FPS with buffer size of 1
- **Half Precision**: FP16 inference for CUDA devices
- **Frame Processing**: Efficient tracking with persist=True to maintain track states
- **FPS Monitoring**: Real-time performance feedback displayed on screen
- **MJPG Codec**: Optimized video capture format when supported

### User Interaction

- **Click-to-Track**: Mouse callback system for selecting specific objects
- **Visual Feedback**: Different colors for selected (green) vs unselected (red) objects
- **Dynamic Selection**: Click elsewhere to deselect and track all objects

### Key Features

- Real-time object detection and tracking with multi-threading
- Interactive object selection via mouse clicks
- Performance monitoring with real-time FPS counter
- Robust tracking for fast-moving objects with ReID support
- Asynchronous frame processing to prevent UI lag
- Smart camera optimization with automatic codec selection
- Configurable detection parameters and frame processing rates
- Clean shutdown with ESC key or 'q' key

## Requirements

- Python 3.8+
- Webcam
- CUDA-compatible GPU (optional, for better performance)

## Installation

1. Clone the repository

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:

```bash
python main.py
```

**Controls**:

- Click on any detected object to track it exclusively
- Click elsewhere to deselect and track all objects
- Press ESC or 'q' to exit

**Advanced Configuration**:

The tracker can be customized with the following parameters:

- `confidence`: Detection confidence threshold (default: 0.5)
- `target_fps`: Target frame rate (default: 30)
- `process_every_n_frames`: Frame processing interval (default: 1)

## Project Structure

```text
RealTimeTracker/
├── main.py              # Main application with RealTimeTracker class
├── Model/
│   ├── yolov8s.pt      # YOLOv8 small model (downloaded automatically)
│   └── botsort.yaml    # Custom BoT-SORT tracking configuration
├── requirements.txt     # Python dependencies with version specifications
├── README.md           # Comprehensive project documentation
└── LICENSE             # Project license
```

## Class Architecture

### RealTimeTracker Class

The main class implements the following key methods:

- `__init__()`: Initializes model, camera, and threading components
- `select_object()`: Mouse callback for interactive object selection
- `detection_worker()`: Threaded YOLO detection processing
- `run()`: Main execution loop with performance monitoring

## Technical Specifications

- **Detection Framework**: Ultralytics YOLOv8
- **Tracking Algorithm**: BoT-SORT with ReID and global motion compensation
- **Video Processing**: OpenCV with threading support
- **Model Format**: PyTorch (.pt)
- **Configuration**: YAML-based tracking parameters
- **Threading**: Asynchronous detection processing
- **Performance**: Half-precision inference on CUDA devices
- **Camera Optimization**: MJPG codec with optimized settings
