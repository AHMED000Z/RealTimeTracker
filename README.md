# RealTimeTracker

A real-time object detection and tracking application using YOLOv8 and DeepSORT.

## Features

- Real-time object detection using YOLOv8
- Object tracking with DeepSORT algorithm
- Click to select and track specific objects
- Webcam video input support

## Requirements

- Python 3.8+
- Webcam

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Run the application:

```bash
python main.py
```

## Usage

1. Run the program to start webcam feed
2. Click on any detected object to track it exclusively
3. Press ESC to exit

## Model

The application uses YOLOv8s model located in the `Model/` directory.
