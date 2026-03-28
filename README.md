# 🎯 YOLO26 Real-Time Object Detection System

<div align="center">

**Production-grade real-time object detection, tracking, and analysis platform**

Built with **YOLO26** · **FastAPI** · **Next.js** · **OpenCV**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14-black?logo=next.js)](https://nextjs.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📋 Overview

A full-stack computer vision application that performs **real-time object detection and tracking** using the latest **YOLO26** model. The system supports video file upload, live camera feed, and REST API inference — all accessible through a modern web interface.

### Key Capabilities

- **80+ COCO object classes** — person, car, bicycle, dog, and more
- **Real-time browser-based detection** — video upload and live camera
- **Dual-loop rendering architecture** — 60fps smooth playback with background AI inference
- **Multi-object tracking** — ByteTrack / BoTSORT with persistent IDs
- **REST API** — FastAPI backend for programmatic access
- **Hot-swappable models** — Switch between YOLO26 nano/small/medium/large at runtime

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   Next.js Web Frontend                       │
│  ┌──────────┐  ┌───────────────┐  ┌───────────────────────┐  │
│  │  Upload   │  │  Live Camera  │  │  Results + Stats      │  │
│  │  Video    │  │  getUserMedia │  │  Bounding Boxes       │  │
│  └────┬──────┘  └──────┬────────┘  └───────────────────────┘  │
└───────┼────────────────┼─────────────────────────────────────┘
        │ POST /detect-frame        │ REST API (JSON + JPEG)
┌───────▼────────────────▼─────────────────────────────────────┐
│                  FastAPI Backend (Python)                     │
│  ┌───────────┐  ┌────────────┐  ┌─────────┐  ┌───────────┐  │
│  │  YOLO26   │  │  ByteTrack │  │  Event   │  │  Video    │  │
│  │  Detector │→ │  Tracker   │→ │  Engine  │→ │  Writer   │  │
│  └───────────┘  └────────────┘  └─────────┘  └───────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Biydaalt/
├── config/
│   └── default.yaml            # Centralized configuration
├── core/
│   ├── detector.py             # YOLO26 inference engine (GPU/FP16)
│   ├── tracker.py              # Track history & trail management
│   ├── stream.py               # Threaded async frame grabber
│   ├── pipeline.py             # Main detection pipeline orchestrator
│   └── events.py               # ROI zone & line-crossing events
├── api/
│   ├── server.py               # FastAPI REST endpoints
│   └── schemas.py              # Pydantic request/response models
├── utils/
│   ├── logger.py               # Logging configuration
│   ├── visualizer.py           # Bounding box, label, trail drawing
│   └── video_writer.py         # Video output codec handler
├── web/                        # Next.js 14 frontend
│   ├── app/
│   │   ├── layout.tsx          # Root layout & metadata
│   │   ├── page.tsx            # Landing page
│   │   ├── demo/page.tsx       # Video upload + detection UI
│   │   └── camera/page.tsx     # Live camera detection UI
│   ├── tailwind.config.js
│   └── package.json
├── cli.py                      # CLI interface
├── main.py                     # Quick-start entry point
├── Dockerfile                  # Docker container definition
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for web frontend)
- NVIDIA GPU + CUDA (recommended) or CPU

### 1. Install Backend

```bash
cd Biydaalt
pip install -r requirements.txt
```

### 2. Start API Server

```bash
python cli.py serve --port 8000
```

### 3. Start Web Frontend

```bash
cd web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Alternative: CLI Usage

```bash
# Webcam real-time detection
python main.py

# Video file detection
python main.py --source path/to/video.mp4

# Single image detection
python cli.py detect --image photo.jpg --output result.jpg --json
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | API status & model info |
| `POST` | `/detect` | Single image detection → JSON |
| `POST` | `/detect-frame` | Frame detection → annotated JPEG + headers |
| `POST` | `/detect-video` | Full video processing |
| `POST` | `/model` | Switch YOLO model at runtime |
| `POST` | `/stream/start` | Start stream-based detection |
| `POST` | `/stream/stop` | Stop active stream |

### Example

```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@photo.jpg" | python -m json.tool
```

---

## ⚙️ Configuration

All parameters are configurable via `config/default.yaml`:

```yaml
model:
  name: yolo26s.pt          # yolo26n / yolo26s / yolo26m / yolo26l
  confidence: 0.25
  iou_threshold: 0.5
  device: "0"               # "0" = GPU, "cpu" = CPU
  half: true                # FP16 inference
  img_size: 640

tracker:
  enabled: true
  type: bytetrack           # bytetrack / botsort

visualization:
  show_fps: true
  show_tracks: true
  trail_length: 30
```

---

## 🐳 Docker

```bash
docker build -t yolo-detect .
docker run --gpus all -p 8000:8000 yolo-detect
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **AI Model** | YOLO26 (Ultralytics 8.4.30) |
| **Backend** | Python 3.12, FastAPI, Uvicorn |
| **Computer Vision** | OpenCV, NumPy, Pillow |
| **Tracking** | ByteTrack / BoTSORT |
| **Frontend** | Next.js 14, React 18, Tailwind CSS 3.4 |
| **Deployment** | Docker, Vercel |

---

## 📄 License

MIT License
