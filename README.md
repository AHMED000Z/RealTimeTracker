# RealTimeTracker

A real-time object detection and tracking application using YOLOv8 and Ultralytics built-in tracking capabilities.

## Implementation Details

### Architecture

The application is built using an object-oriented approach with a `RealTimeTracker` class that encapsulates all tracking functionality. This design eliminates global variables and provides clean state management.

### Object Detection

- **Model**: YOLOv8s (small variant) for optimal balance between speed and accuracy
- **Detection Filter**: Excludes person class to focus on objects only as per task requirements
- **Confidence Threshold**: 0.5 (configurable)

### Object Tracking

- **Algorithm**: BoT-SORT (Byte Track + OCSort) via Ultralytics built-in tracking
- **Configuration**: Custom `botsort.yaml` with optimized parameters for fast-moving objects
- **Track Buffer**: 60 frames for persistence during occlusions
- **Re-Identification**: Enabled with proximity threshold of 0.3
- **Match Threshold**: 0.6 for robust tracking

### Performance Optimizations

- **Camera Settings**: 640x480 resolution at 30 FPS with buffer size of 1
- **Frame Processing**: Efficient tracking with persist=True to maintain track states
- **FPS Monitoring**: Real-time performance feedback displayed on screen

### User Interaction

- **Click-to-Track**: Mouse callback system for selecting specific objects
- **Visual Feedback**: Different colors for selected (green) vs unselected (red) objects
- **Dynamic Selection**: Click elsewhere to deselect and track all objects

### Key Features

- Real-time object detection and tracking
- Interactive object selection via mouse clicks
- Performance monitoring with FPS counter
- Robust tracking for fast-moving objects
- Clean shutdown with ESC key

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
- Press ESC to exit

## Project Structure

```text
RealTimeTracker/
├── main.py              # Main application file
├── Model/
│   ├── yolov8s.pt      # YOLOv8 small model
│   └── botsort.yaml    # Custom tracking configuration
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Technical Specifications

- **Detection Framework**: Ultralytics YOLOv8
- **Tracking Algorithm**: BoT-SORT with ReID
- **Video Processing**: OpenCV
- **Model Format**: PyTorch (.pt)
- **Configuration**: YAML-based tracking parameters
