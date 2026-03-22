# Real-Time Object Detection System (YOLO-based)

Production-grade, high-performance real-time object detection using **YOLO11** (Ultralytics).

## Architecture

```
Biydaalt/
├── config/
│   └── default.yaml          # All configurable parameters
├── core/
│   ├── detector.py            # YOLO inference engine (GPU/FP16)
│   ├── tracker.py             # Track history & trail management
│   ├── stream.py              # Threaded async frame grabber
│   ├── pipeline.py            # Main orchestrator
│   └── events.py              # ROI zone & line-crossing events
├── api/
│   ├── server.py              # FastAPI REST endpoints
│   └── schemas.py             # Pydantic models
├── utils/
│   ├── logger.py              # Logging configuration
│   ├── visualizer.py          # Bbox, label, trail drawing
│   └── video_writer.py        # Video output
├── cli.py                     # Full CLI interface
├── main.py                    # Quick-start entry
├── Dockerfile
├── requirements.txt
└── README.md
```

## Features

| Feature | Details |
|---|---|
| **Models** | YOLO11 nano/small/medium/large/xlarge — hot-swappable |
| **Input** | Webcam, video file, RTSP stream, multi-stream |
| **Performance** | CUDA + FP16 inference, threaded I/O, <30ms latency |
| **Tracking** | ByteTrack / BoTSORT with persistent IDs and trails |
| **Events** | ROI zone enter/exit, line-crossing detection |
| **API** | FastAPI with image upload, stream control, model switch |
| **Output** | Live window, video file, JSON detections log |

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Run (webcam)

```bash
python main.py
```

### 3. Run (video file)

```bash
python main.py --source path/to/video.mp4
```

### 4. Run (RTSP stream)

```bash
python main.py --source "rtsp://user:pass@ip:554/stream"
```

### 5. Multi-stream

```bash
python main.py --source 0 "rtsp://..."
```

## CLI Reference

```bash
# Real-time detection with options
python cli.py run --source 0 --model yolo11s.pt --save-video

# Single image detection
python cli.py detect --image photo.jpg --output result.jpg --json

# Start API server
python cli.py serve --port 8000

# Custom config
python cli.py run --config my_config.yaml --source 0
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Status, model info |
| POST | `/detect` | Upload image → detections JSON |
| POST | `/model` | Switch model (nano/small/medium) |
| POST | `/stream/start` | Start detection on a stream source |
| POST | `/stream/stop` | Stop active stream |

### Example: detect via API

```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@photo.jpg" | python -m json.tool
```

## Configuration

Edit `config/default.yaml`:

```yaml
model:
  name: yolo11n.pt       # yolo11n/s/m/l/x
  confidence: 0.35
  device: "0"             # "0" = GPU, "cpu" = CPU
  half: true              # FP16

tracker:
  enabled: true
  type: bytetrack

visualization:
  show_fps: true
  show_tracks: true
  trail_length: 30
```

## Docker

```bash
docker build -t yolo-detect .
docker run --gpus all -p 8000:8000 yolo-detect
```

## Controls

- Press **`q`** to quit the live window

## Requirements

- Python 3.10+
- NVIDIA GPU + CUDA (recommended) or CPU
- Webcam / video file / RTSP stream
