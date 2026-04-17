# 🎯 Smart Study Space Monitor — YOLO26 Real-Time Detection System

<div align="center">

**AI-powered study space monitoring with person-object ownership, seat occupancy, and unattended item detection**

Built with **YOLO26** · **FastAPI** · **Next.js** · **OpenCV**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14-black?logo=next.js)](https://nextjs.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📋 Overview

A full-stack computer vision application that monitors **study spaces** (libraries, co-working areas) using **YOLO26** real-time object detection and tracking. The system detects people and their belongings, assigns ownership via proximity analysis, tracks desk/seat occupancy, and alerts on unattended items.

### Key Capabilities

- **Smart Study Space Monitor** — person-object ownership, seat occupancy, unattended alerts
- **80+ COCO object classes** — person, laptop, phone, book, bag, cup, and more
- **Person-Object Ownership** — automatically assigns nearby objects to the closest person
- **Desk Zone Occupancy** — configurable polygon zones with real-time occupied/vacant status
- **Unattended Item Detection** — alerts when objects are left without a nearby person
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
│  │  YOLO26   │  │  ByteTrack │  │  Study   │  │  Video    │  │
│  │  Detector │→ │  Tracker   │→ │  Space   │→ │  Writer   │  │
│  └───────────┘  └────────────┘  └─────────┘  └───────────┘  │
│                                  ├ Proximity (ownership)     │
│                                  ├ Zones (occupancy)         │
│                                  └ Alerts (unattended)       │
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
│   ├── events.py               # ROI zone & line-crossing events
│   ├── proximity.py            # Person-object ownership engine
│   ├── zones.py                # Desk/seat zone occupancy manager
│   ├── alerts.py               # Unattended item alert system
│   └── study_space.py          # Smart Study Space orchestrator
├── api/
│   ├── server.py               # FastAPI REST endpoints
│   └── schemas.py              # Pydantic request/response models
├── utils/
│   ├── logger.py               # Logging configuration
│   ├── visualizer.py           # Bounding box, label, trail, study-space drawing
│   └── video_writer.py         # Video output codec handler
├── web/                        # Next.js 14 frontend
│   ├── app/
│   │   ├── layout.tsx          # Root layout & metadata
│   │   ├── page.tsx            # Landing page
│   │   ├── demo/page.tsx       # Video upload + detection UI
│   │   ├── camera/page.tsx     # Live camera detection UI
│   │   └── study-space/page.tsx# Smart Study Space dashboard
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

### Study Space Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/study-space/detect-frame` | Frame detection + ownership analysis → annotated JPEG |
| `GET` | `/study-space/status` | Current zones & active alerts |
| `POST` | `/study-space/zones` | Configure desk zones at runtime |
| `POST` | `/study-space/reset` | Reset alerts & zone timers |

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
  name: yolo26l.pt          # yolo26n / yolo26s / yolo26m / yolo26l (large = best accuracy)
  confidence: 0.15
  iou_threshold: 0.5
  device: "0"               # "0" = GPU, "cpu" = CPU
  half: true                # FP16 inference
  img_size: 1280            # higher res = better small object detection

tracker:
  enabled: true
  type: bytetrack           # bytetrack / botsort

study_space:
  ownership_max_distance: 200
  alerts:
    unattended_timeout: 30
    cooldown: 60

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
