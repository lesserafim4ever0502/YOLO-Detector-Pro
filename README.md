# YOLO-Detector-Pro

A professional and modern YOLO-based object detection GUI application built with PySide6.

## Overview

YOLO-Detector-Pro is a modern and feature-rich GUI application designed for real-time object detection and batch processing. It provides a sleek interface for working with YOLO models, offering flexible detection modes across images, videos, folders, and live camera streams.

## Key Features

- 🖼️ **Image Detection** - High-precision detection for single image files with preview and result saving.
- 📁 **Folder Batch Processing** - Efficiently process entire directories of images with comprehensive Text report generation.
- 📹 **Video Detection** - Real-time inference on video files with **automatic saving** of annotated results to the `results` folder.
- 📹 **Live Camera Stream** - Low-latency detection from connected USB/Integrated cameras.
- 📊 **Dynamic Dashboard** - Visual statistics and real-time tracking of detection metrics.
- ⚖️ **Comparison Mode** - Side-by-side performance evaluation of two different YOLO models on the same content.
- ⚙️ **Advanced Controls** - Fine-tune confidence thresholds, IOU thresholds, line widths, and inference delays in real-time.

## Technology Stack

- **UI Framework**: PySide6 (Qt for Python)
- **AI Engine**: Ultralytics YOLO (v8/v9/v10/v11 support)
- **Computer Vision**: OpenCV
- **Backend Architecture**: Multi-threaded processing (QThread) for smooth UI performance.

## Project Structure

```text
YOLO-Detector-Pro/
├── main.py                 # Application entry point
├── main_window.py          # Primary UI and logic integration
├── core/
│   ├── yolo_worker.py      # Core inference engine
│   └── comparison_worker.py # Model comparison logic
├── widgets/
│   ├── dashboard_widget.py # Statistics visualization
│   └── comparison_widget.py # Comparison UI components
├── ui/
│   └── style.qss           # Modern dark-theme styling
├── utils/
│   └── helper.py           # Image/result processing utilities
├── models/                 # Local directory for .pt model files
└── assets/                 # Icons and media resources
```

## Installation

### Prerequisites
- Python 3.8+
- (Optional) CUDA-enabled GPU for faster inference

### Install Dependencies
```bash
pip install PySide6 opencv-python ultralytics numpy
```

## Quick Start

1. Place your YOLO models (`.pt` files) in the `models/` folder.
2. Launch the application:
```bash
python main.py
```

## User Guide

### Navigation Sidebar
- **Image**: Select and run detection on a single image.
- **Folder**: Select a directory for batch processing.
- **Camera**: Start real-time detection from your webcam.
- **Dashboard**: View aggregate statistics of recent detections.
- **Compare**: Load two models to compare their performance and results side-by-side.

### Top Status Bar
- **Classes**: Number of unique object classes detected.
- **Targets**: Total number of objects identified in the current frame.
- **FPS**: Real-time processing speed (Frames Per Second).
- **Model**: Displays the currently active YOLO model.

### Settings Panel (Right)
- **Model Selection**: Switch between available models in the `models/` folder on the fly.
- **Confidence Threshold**: Adjust minimum confidence (0.01 - 0.99).
- **IOU Threshold**: Fine-tune Non-Maximum Suppression (NMS) overlap.
- **Line Width**: Customize the thickness of the bounding boxes.
- **Inference Delay**: Add a delay for testing or presentation purposes.
- **Run/Stop**: Control the detection lifecycle.
- **Save Results**: Export the current detection to the `results/` directory.

## Roadmap

- [x] Modern Responsive UI (QSS)
- [x] Multi-format Detection (Image, Video, Folder, Camera)
- [x] Dynamic Model Switching
- [x] Side-by-Side Model Comparison
- [x] Automated Video Saving
- [x] Data Dashboard Visualization
- [ ] Export results to JSON/CSV
- [ ] Support for custom ONNX/TensorRT models

## License

This project is licensed under the MIT License.

## Author
**Lee jimmy**
GitHub: [lesserafim4ever0502/YOLO-Detector-Pro](https://github.com/lesserafim4ever0502/YOLO-Detector-Pro)
