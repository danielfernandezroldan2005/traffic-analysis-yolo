
https://github.com/user-attachments/assets/407f0078-9205-4216-9802-3b4e554e9329
# 🚦 Real-Time Traffic Analytics & GDPR Anonymization (YOLO11 + ByteTrack)

A lightweight, modular Python pipeline built with **YOLO11**, **ByteTrack**, and **OpenCV** to detect, track, and count multi-class traffic in real time, including an optional **GDPR privacy blur** for pedestrians and license plates.

## 🎥 Demo

| Standard Tracking & Counting | GDPR Privacy Mode (`--anonymize`) |
| :---: | :---: |
| 



https://github.com/user-attachments/assets/163c5ba3-c0d9-4ece-8add-07a6cd9542ae

 | 


 
 |

## ✨ Features

- **Multi-Object Tracking:** Persistent vehicle IDs and fading trajectory trails using `YOLO11` + `ByteTrack`.
- **Bidirectional Line Counting:** Counts `Up` and `Down` crossings across a virtual line without duplicate IDs.
- **GDPR Anonymization (`--anonymize`):** Real-time Gaussian blur on pedestrians and lower vehicle regions (license plates).
- **Modular Structure:** Clean separation between inference (`tracker.py`), rendering (`visualizer.py`), and CLI (`main.py`).

## 🚀 Quickstart

```bash
# 1. Install dependencies
uv pip install ultralytics opencv-python

# 2. Standard execution
python -m src.main --source assets/video3.mp4 --output outputs/demo_traffic.mp4

# 3. With real-time GDPR anonymization
python -m src.main --source assets/video5.mp4 --output outputs/demo_gdpr.mp4 --anonymize
