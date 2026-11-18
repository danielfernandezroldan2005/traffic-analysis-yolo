# 🚦 Real-Time Traffic Analytics & GDPR Anonymization (YOLO11 + ByteTrack)

A lightweight, modular Python pipeline built with **YOLO11**, **ByteTrack**, and **OpenCV** to detect, track, and count multi-class traffic in real time, featuring an optional **GDPR privacy blur** for pedestrians and license plates.

## 🎥 Demo

Video demos showcasing standard line-counting and real-time GDPR anonymization are available on my social profiles:
- 💼 **LinkedIn:** [Watch demo post on LinkedIn](www.linkedin.com/in/danielfernandezroldan2005)
- 📸 **Instagram:** [Watch reel on Instagram](https://www.instagram.com/danielfr_2005)

## ✨ Features

- **Multi-Object Tracking:** Persistent vehicle IDs and fading trajectory trails using `YOLO11` + `ByteTrack`.
- **Bidirectional Line Counting:** Counts `Up` and `Down` crossings across a virtual line without duplicate IDs.
- **GDPR Anonymization (`--anonymize`):** Real-time Gaussian blur applied to pedestrians and lower vehicle regions (license plates).
- **Modular Structure:** Clean separation between inference (`tracker.py`), rendering (`visualizer.py`), and CLI orchestration (`main.py`).

## 📂 Project Structure

```text
yolo-traffic-analytics/
├── assets/          # Input clips (ignored by git)
├── outputs/         # Processed videos (ignored by git)
├── src/
│   ├── __init__.py
│   ├── main.py      # CLI entry point & video I/O loop
│   ├── tracker.py   # YOLO11 + ByteTrack engine & line-crossing math
│   └── visualizer.py# HUD overlay, motion trails & GDPR Gaussian blur
├── .gitignore
└── pyproject.toml
