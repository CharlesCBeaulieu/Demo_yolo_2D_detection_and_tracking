# 🔍 YOLO 2D Object Detection, Classification, and Tracking Showcase

This repository is an interactive demo showcasing state-of-the-art 2D object detection, classification, and object tracking using pretrained **Ultralytics YOLO** models (YOLO11 and YOLOv8).

The project is designed to run both as a robust **Command-Line Interface (CLI) utility** and as a **premium Streamlit Web Application** featuring real-time visual output, parameters optimization, and tracking trajectory paths.

---

## 🌟 Key Features

1. **2D Object Detection**: Detect and locate 80 standard COCO object classes (pedestrians, vehicles, animals, laptops, etc.) with bounding boxes and confidence scores.
2. **Multi-Object Tracking**: Track multiple detected objects across video frames with unique, persistent track IDs. It includes custom **motion path trails** to visualize movement trajectory.
3. **Image Classification**: Classify entire images into ImageNet classes (e.g. transport, animals, objects) and view the top-5 confidence score breakdown.
4. **Dynamic Web Dashboard**: A Streamlit interface to upload your own files, tweak confidence levels in real-time, test webcam feeds, and see metrics side-by-side.

---

## 📂 Project Structure

```text
Demo_yolo_2D_detection_and_tracking/
├── data/                    # Directory for sample images/videos (auto-created)
├── app.py                   # Streamlit web application
├── detect_and_track.py      # Standalone CLI python script
├── download_sample.py       # Utility to download test files
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## ⚡ Quick Start

### 1. Environment Setup

It is highly recommended to use a virtual environment (`venv` or `conda`):

```bash
# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Download Sample Media

To download a sample image (`bus.jpg`) and a sample video (`california.mp4`) for testing:

```bash
python download_sample.py
```

---

## 🖥️ Usage (CLI Utility)

Run inference directly from the command line using `detect_and_track.py`:

### **2D Object Detection** (Image/Video)
```bash
python detect_and_track.py --task detect --source data/bus.jpg
```

### **Multi-Object Tracking** (Video/Webcam)
```bash
python detect_and_track.py --task track --source data/california.mp4
```

### **Image Classification** (Image)
```bash
python detect_and_track.py --task classify --source data/bus.jpg
```

### **Advanced CLI Parameters**
- `--model`: Path or name of model (e.g., `yolo11n.pt`, `yolov8s.pt`, `yolo11n-cls.pt`). Defaults are automatically managed.
- `--conf`: Set confidence threshold (default: `0.25`).
- `--iou`: Set Intersection over Union threshold for Non-Maximum Suppression (default: `0.45`).
- `--device`: Target device (e.g. `cpu`, `cuda:0` for NVIDIA GPU, `mps` for Apple Silicon).
- `--show`: Display live OpenCV windows during inference.
- `--no-save`: Prevents saving annotated files (they save to `runs/` by default).

---

## 🌐 Interactive Web Dashboard (Streamlit)

Launch the modern web UI to upload files and interactively control detection/tracking:

```bash
streamlit run app.py
```

### **Dashboard Features:**
- **Interactive Configuration (Sidebar)**: Toggle tasks, choose between YOLO11 & YOLOv8 architectures, adjust size weights (nano, small, medium, etc.), and control sensitivity parameters.
- **Upload / Sample selector**: Drag and drop any image/video, run the pre-downloaded sample media, or hook up your local webcam stream.
- **Side-by-side Visuals**: View original frames alongside YOLO output overlays.
- **Class Distributions**: Explore color-coded bar charts detailing target counts and classification probabilities.
- **Motion Trails**: Watch tracking ID historical paths draw in real-time.

---

## 🚀 GPU Acceleration (Optional)

To speed up processing of video files and live feeds, you can run YOLO on an NVIDIA GPU. 
Ensure you have the matching [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) installed, then install the PyTorch package supporting CUDA:

```bash
# Example for PyTorch with CUDA 12.1 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
The script and web app will automatically utilize `cuda:0` if available, or you can manually target it via UI settings/CLI parameters.

---

## ⚖️ License
This project utilizes the official pretrained models from Ultralytics. Refer to the [Ultralytics License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) for usage policies (AGPL-3.0 / Enterprise).
