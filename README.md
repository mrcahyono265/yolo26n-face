<div align="center">
  <h1>YOLO26n-Face</h1>
  <p>
    Built on the powerful <a href="https://github.com/ultralytics/ultralytics">Ultralytics YOLO Framework</a>
  </p>

  <p>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9.25-blue?logo=python" alt="Python"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch" alt="PyTorch"></a>
    <a href="https://github.com/ultralytics/ultralytics"><img src="https://img.shields.io/badge/Powered_by-Ultralytics_YOLO-blue" alt="Ultralytics"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-AGPL%203.0-red" alt="License"></a>
  </p>

  </div>

<br>

## Introduction

YOLO26n-Face is a final year [research project that I developed]() using the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) framework. This project aims to train a very lightweight face detection model for resource-constrained devices such as Drones (UAV).

This model is the result of Transfer Learning from the official [Ultralytics](https://docs.ultralytics.com/models/yolo26/) *base model*, which was then fine-tuned specifically on the [WiderFace dataset](http://shuoyang1213.me/WIDERFACE/) to recognize faces in various scales and lighting conditions.

## 🔥 Key Features

- ✅ **Pure Detection:** Focuses on precise face bounding box detection.
- ✅ **Drone Optimized:** Trained with visual augmentations for aerial view simulation.
- ✅ **Lightweight:** Small model size, ideal for IoT deployment.

## Performance Metrics

The model was trained and validated using the [WiderFace dataset](http://shuoyang1213.me/WIDERFACE/) for 100 Epochs.

| Metric | Value |
| :--- | :--- |
| **mAP@50** | **62.8%** |
| **Precision** | **80.5%** |
| **Recall** | **53.9%** |
| **Inference Time**| **4.9 ms** |

<div align="center">
  <img src="YOLO26_Face\YOLO26n_Face_Run\results.png" width="80%" alt="Training Metrics Graphs">
  <img src="YOLO26_Face\YOLO26n_Face_Run\confusion_matrix.png" width="80%" alt="Confusion Matrix Graphs">
</div>

## Download Models

Below are the trained model and the link to the original Base Model from Ultralytics for reference:

| Model |
| :--- |
| [yolo26n-face.pt](MASUKKAN_LINK_GOOGLE_DRIVE_DISINI) |
| [yolo26n.pt (base)](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt) |

## Environment

This project was developed in the following environment:
- **OS:** Windows 11
- **Python:** 3.9.25
- **GPU:** NVIDIA GeForce RTX 2050 (4GB VRAM)

### Installation
This project requires the original Ultralytics library:
```bash
# 1. Clone repository ini
git clone https://github.com/mrcahyono265/yolo26n-face.git
cd yolo26n-face

# 2. Install Ultralytics & dependencies
pip install ultralytics opencv-python torch
```

## 🚀 Usage
1. Via Terminal
```bash
# Deteksi wajah pada gambar
yolo task=detect mode=predict model=yolo26n-face.pt source='test.jpg' show=True

# Deteksi real-time via Webcam
yolo task=detect mode=predict model=yolo26n-face.pt source=0 show=True
```

2. Via Python Script
```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolo26n-face.pt")

# Inference (Webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    results = model(frame, conf=0.5) 
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO26n Face Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Training Configuration
Detailed training configuration used to produce this model:
| Parameter | Value |
| :--- | :--- |
| **Epochs** | **100** |
| **Optimizer** | **SGD** |
| **Image Size** | **640** |
| **Batch Size**| **2** |

## <div align="center">License</div>

YOLO26 is available under two different licenses:

- **GPL-3.0 License**: See [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file for details.
- **Enterprise License**: Provides greater flexibility for commercial product development without the open-source
  requirements of GPL-3.0. Typical use cases are embedding Ultralytics software and AI models in commercial products and
  applications. Request an Enterprise License at [Ultralytics Licensing](https://ultralytics.com/license).

<div align="center"> <p>Developed by <strong>Mohammad Ridho Cahyono</strong></p> <p>Informatics Engineering - UNIDA Gontor (2026)</p> </div>