# YOLO26 Real-Time Object Detection System — Төслийн Дэлгэрэнгүй Тайлан

## Агуулга

1. [Төслийн Тойм](#1-төслийн-тойм)
2. [Технологийн Стек](#2-технологийн-стек)
3. [Системийн Архитектур](#3-системийн-архитектур)
4. [Backend — Python FastAPI](#4-backend--python-fastapi)
5. [Frontend — Next.js Web Application](#5-frontend--nextjs-web-application)
6. [YOLO26 Detection Pipeline](#6-yolo26-detection-pipeline)
7. [Dual-Loop Rendering Architecture](#7-dual-loop-rendering-architecture)
8. [API Reference](#8-api-reference)
9. [Configuration System](#9-configuration-system)
10. [Deployment](#10-deployment)
11. [Хөгжүүлэлтийн Явц](#11-хөгжүүлэлтийн-явц)
12. [Дүгнэлт](#12-дүгнэлт)

---

## 1. Төслийн Тойм

### 1.1 Зорилго

Энэхүү төслийн зорилго нь **YOLO26** (You Only Look Once) загварыг ашиглан бодит цагийн объект илрүүлэлт (real-time object detection), ангилал (classification), болон хөөлт (tracking) хийх бүрэн функционал систем бүтээх юм. Систем нь веб интерфэйсээр дамжуулан хэрэглэгчид видео файл оруулах, камераар шууд илрүүлэлт хийх, REST API-аар програмчлалын хандалт хийх боломж олгоно.

### 1.2 Үндсэн Чадварууд

| Чадвар | Тодорхойлолт |
|--------|-------------|
| **80+ COCO объектын анги** | Хүн, машин, дугуй, нохой гэх мэт олон төрлийн объект таних |
| **Бодит цагийн илрүүлэлт** | Видео файл болон камерын шууд дүрсэнд объект илрүүлэх |
| **60fps Smooth Playback** | Dual-loop архитектурын тусламжтайгаар тасалдалгүй тоглуулалт |
| **Multi-Object Tracking** | ByteTrack / BoTSORT алгоритмаар объектуудад тогтвортой ID оноох |
| **REST API** | FastAPI backend-ээр програмчлалын хандалт |
| **Hot-Swap Models** | Runtime-д YOLO26 nano/small/medium/large загварууд хооронд шилжих |
| **Docker Deployment** | Нэг командаар deploy хийх боломж |

### 1.3 Шийдсэн Асуудал

Уламжлалт объект илрүүлэлтийн системүүд зөвхөн CLI эсвэл desktop application хэлбэрээр ажиллаж, хэрэглэгчдэд хүндрэлтэй байдаг. Энэ төсөл нь:

- **Веб хөтөч дээрээс** ямар ч суулгалтгүйгээр ашиглах боломжтой
- **Video файлыг frame-by-frame** бодит цагт шинжлэх (бүхэлд нь хүлээхгүй)
- **Камерын feed-г шууд** шинжилж, detection overlay харуулах
- **Backend/Frontend тусдаа** ажиллах тул scale хийхэд хялбар

---

## 2. Технологийн Стек

### 2.1 Backend

| Технологи | Хувилбар | Зорилго |
|-----------|----------|---------|
| **Python** | 3.12 | Үндсэн програмчлалын хэл |
| **Ultralytics** | 8.4.30 | YOLO26 загвар ажиллуулах framework |
| **FastAPI** | 0.104+ | REST API framework |
| **Uvicorn** | 0.24+ | ASGI web server |
| **OpenCV** | 4.8+ | Зураг/видео боловсруулалт |
| **PyTorch** | 2.0+ | Deep learning engine (CUDA/CPU) |
| **NumPy** | 1.24+ | Тоон тооцоолол |
| **Pydantic** | 2.0+ | Data validation & serialization |
| **PyYAML** | 6.0+ | Configuration parsing |
| **Pillow** | 10.0+ | Зурган боловсруулалт |
| **lapx** | 0.5+ | Linear Assignment Problem (tracking) |

### 2.2 Frontend

| Технологи | Хувилбар | Зорилго |
|-----------|----------|---------|
| **Next.js** | 14.2 | React-based web framework |
| **React** | 18.3 | UI component library |
| **TypeScript** | 5.x | Type-safe JavaScript |
| **Tailwind CSS** | 3.4 | Utility-first CSS framework |

### 2.3 DevOps

| Технологи | Зорилго |
|-----------|---------|
| **Docker** | Containerization |
| **Git/GitHub** | Version control |
| **Vercel** | Frontend deployment |

---

## 3. Системийн Архитектур

### 3.1 Ерөнхий Бүтэц

```
┌─────────────────────────────────────────────────────────────────┐
│                      CLIENT (Web Browser)                       │
│                                                                 │
│  ┌─────────────┐   ┌─────────────────┐   ┌──────────────────┐  │
│  │  Landing     │   │  Video Upload   │   │  Live Camera     │  │
│  │  Page        │   │  + Detection    │   │  + Detection     │  │
│  │  (page.tsx)  │   │  (demo/page)    │   │  (camera/page)   │  │
│  └─────────────┘   └────────┬────────┘   └────────┬─────────┘  │
│                              │                      │            │
│                     Canvas Display Loop (60fps)                  │
│                     + Background Detection Loop                  │
└──────────────────────────────┼──────────────────────┼────────────┘
                               │ HTTP POST            │
                               │ /detect-frame        │
┌──────────────────────────────▼──────────────────────▼────────────┐
│                      SERVER (FastAPI + Uvicorn)                   │
│                                                                   │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────────────────┐ │
│  │  API Router  │──▶│  YOLO26      │──▶│  Response Builder     │ │
│  │  (server.py) │   │  Detector    │   │  (Annotated JPEG      │ │
│  │              │   │  (detector)  │   │   + JSON headers)     │ │
│  └──────────────┘   └──────────────┘   └───────────────────────┘ │
│                                                                   │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────────────────┐ │
│  │  ByteTrack   │   │  Event       │   │  Visualizer           │ │
│  │  Tracker     │   │  Engine      │   │  (Box drawing)        │ │
│  └──────────────┘   └──────────────┘   └───────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 Өгөгдлийн Урсгал (Data Flow)

```
1. Client → Frame capture (canvas, max 640px) → JPEG encode (60% quality)
2. JPEG → HTTP POST /detect-frame → FastAPI server
3. Server → JPEG decode → OpenCV BGR frame
4. Frame → YOLO26 model.predict() → Detection results
5. Results → Annotated frame + JSON metadata
6. Response → StreamingResponse (JPEG body + X-Detection-Data header)
7. Client → Parse detections → Draw boxes on display canvas
```

### 3.3 Файлын Бүтэц

```
Biydaalt/
├── config/
│   └── default.yaml              # Бүх тохиргооны параметрүүд
├── core/                         # Detection pipeline-ийн цөм
│   ├── detector.py               # YOLO26 inference engine
│   ├── tracker.py                # Object tracking & trail management
│   ├── stream.py                 # Threaded async frame grabber
│   ├── pipeline.py               # Үндсэн orchestrator
│   └── events.py                 # ROI zone & line-crossing events
├── api/                          # REST API давхарга
│   ├── server.py                 # FastAPI endpoints
│   └── schemas.py                # Pydantic models
├── utils/                        # Туслах модулиуд
│   ├── logger.py                 # Logging тохиргоо
│   ├── visualizer.py             # Bounding box, label зурах
│   └── video_writer.py           # Видео файл хадгалах
├── web/                          # Next.js frontend
│   ├── app/
│   │   ├── layout.tsx            # Root layout & metadata
│   │   ├── page.tsx              # Нүүр хуудас
│   │   ├── globals.css           # Global styles
│   │   ├── demo/page.tsx         # Видео upload + detection
│   │   └── camera/page.tsx       # Камер detection
│   ├── .env.local                # API URL тохиргоо
│   ├── next.config.js            # Next.js тохиргоо
│   ├── tailwind.config.js        # Tailwind CSS тохиргоо
│   ├── tsconfig.json             # TypeScript тохиргоо
│   └── package.json              # Dependencies
├── cli.py                        # Command Line Interface
├── main.py                       # Quick-start entry point
├── video.py                      # Video processing utilities
├── Dockerfile                    # Docker container definition
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # Төслийн тайлбар
```

---

## 4. Backend — Python FastAPI

### 4.1 Detector Module (`core/detector.py`)

Энэ нь системийн хамгийн чухал хэсэг — YOLO26 загварыг ачаалж, дүрс бүр дээр inference ажиллуулна.

#### Классууд

**`Detection`** — Нэг объектын илрүүлэлтийн үр дүн:
```python
@dataclass
class Detection:
    bbox: list[float]      # [x1, y1, x2, y2] координат
    confidence: float      # Итгэлцлийн үнэлгээ (0.0 - 1.0)
    class_id: int         # COCO ангийн ID (0-79)
    class_name: str       # Ангийн нэр ("person", "car", ...)
    track_id: int | None  # Tracking ID (ByteTrack)
```

**`FrameResult`** — Нэг frame-ийн бүх илрүүлэлтийн үр дүн:
```python
@dataclass
class FrameResult:
    detections: list[Detection]
    inference_ms: float    # Inference хугацаа (ms)
    frame_id: int         # Frame дугаар
    timestamp: float      # Цагийн тэмдэг
```

**`Detector`** — YOLO загварын wrapper:
```python
class Detector:
    def load(self):
        """Загвар ачаалах, CUDA/CPU автомат илрүүлэх, warm-up"""
        
    def detect(self, frame) -> FrameResult:
        """Зөвхөн detection (tracking-гүй)"""
        
    def detect_and_track(self, frame, tracker_type="bytetrack") -> FrameResult:
        """Detection + ByteTrack/BoTSORT tracking"""
        
    def switch_model(self, model_name: str):
        """Runtime-д загвар солих (hot-swap)"""
```

#### Ажиллах Зарчим

1. **Model Loading**: `YOLO('yolo26s.pt')` — Ultralytics-ийн YOLO класс нь загварыг автоматаар татаж ачаална
2. **Device Detection**: CUDA GPU байгаа эсэхийг шалгаж, байвал GPU, үгүй бол CPU дээр ажиллана
3. **FP16 (Half Precision)**: GPU дээр FP16 inference бол 2x хурдтай
4. **Warm-up**: Эхний dummy inference ажиллуулж, загварыг "халаана" — бодит inference хурдтай эхлэхийн тулд
5. **Inference**: `model.predict(frame, conf=0.25, iou=0.5, imgsz=640)` — confidence threshold, NMS IOU threshold тохируулж болно

### 4.2 Stream Module (`core/stream.py`)

Threaded frame grabber — видео эх сурвалжаас frame-үүдийг тусдаа thread дээр уншина.

```python
class FrameGrabber:
    """Non-blocking, threaded I/O давхарга"""
    
    def start(self):
        """OpenCV VideoCapture нээж, background thread эхлүүлэх"""
        
    def read(self) -> tuple[bool, np.ndarray]:
        """Дараагийн frame авах (FIFO queue-ээс)"""
        
    def stop(self):
        """Capture хаах, thread зогсоох"""
```

**Онцлогууд**:
- **FIFO Queue**: Webcam-д buffer_size=2 (хуучин frame-ийг хаях), video файлд buffer_size=128
- **Auto-reconnect**: RTSP stream тасрахад автоматаар дахин холбогдох
- **FPS Preservation**: Video файлын анхны FPS-ийг хадгалах

### 4.3 Pipeline Module (`core/pipeline.py`)

Бүх модулиудыг нэгтгэж, detection loop ажиллуулна.

```python
class DetectionPipeline:
    """Stream → Detect → Track → Events → Visualize → Output"""
    
    def run(self):
        """Үндсэн detection loop"""
        while running:
            frame = stream.read()
            result = detector.detect_and_track(frame)
            tracks = tracker.update(result.detections)
            events = event_processor.process(result.detections)
            annotated = visualizer.draw(frame, result, tracks)
            video_writer.write(annotated)
            cv2.imshow("Detection", annotated)
```

### 4.4 Tracker Module (`core/tracker.py`)

Объектуудын хөдөлгөөнийг хянаж, тогтвортой ID оноож, motion trail зурна.

- **ByteTrack**: Хурдан, олон объект хянахад тохиромжтой
- **BoTSORT**: Илүү нарийвчлалтай, гэхдээ удаан
- **Trail**: Объект бүрийн сүүлийн 30 frame-ийн байрлалыг хадгалж, мөр зурна

### 4.5 Events Module (`core/events.py`)

ROI (Region of Interest) бүс болон шугам хөндлөн гарах (line-crossing) event илрүүлнэ.

- **ROI Zone**: Polygon хэлбэртэй бүс тодорхойлж, объект түүнд орох/гарахыг мэдэгдэнэ
- **Line Crossing**: Шугам тодорхойлж, объект түүнийг хөндлөн гарахыг мэдэгдэнэ

### 4.6 API Server (`api/server.py`)

FastAPI-д суурилсан REST API сервер.

```python
app = FastAPI(title="YOLO Detection API")

@app.get("/health")
async def health():
    """Серверийн төлөв, загварын мэдээлэл"""

@app.post("/detect")
async def detect_image(file: UploadFile):
    """Нэг зураг дээр detection хийж JSON буцаах"""

@app.post("/detect-frame")
async def detect_frame(file: UploadFile):
    """Frame detection — annotated JPEG + X-Detection-Data header"""

@app.post("/detect-video")
async def detect_video(file: UploadFile):
    """Бүтэн видео боловсруулax"""

@app.post("/model")
async def switch_model(request: ModelSwitchRequest):
    """Runtime-д загвар солих"""
```

**CORS**: Бүх origin-ээс хандалт зөвшөөрсөн (`allow_origins=["*"]`).

**`/detect-frame` endpoint (хамгийн чухал)**:

Энэ endpoint нь web frontend-ээс frame-by-frame detection хийхэд ашиглагддаг:

1. Client JPEG frame илгээнэ
2. Server YOLO26-аар detection хийнэ
3. Annotated JPEG-г body-д, detection metadata-г HTTP header-т буцаана:
   - `X-Inference-Ms`: Inference хугацаа (ms)
   - `X-Detections`: Илрүүлсэн объектын тоо
   - `X-Detection-Data`: JSON — bbox координат, class name, confidence

### 4.7 Schemas (`api/schemas.py`)

Pydantic model-ууд:

```python
class DetectionOut(BaseModel):
    bbox: list[float]          # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    track_id: int | None

class InferenceResponse(BaseModel):
    inference_ms: float
    num_detections: int
    detections: list[DetectionOut]

class StatusResponse(BaseModel):
    status: str
    model: str
    device: str
    num_classes: int
```

### 4.8 Visualizer (`utils/visualizer.py`)

Frame дээр detection үр дүнг дүрслэн зурна:

- **Bounding Box**: Өнгөт хүрээ (20 ялгаатай өнгө, class ID-аар)
- **Label**: Ангийн нэр + confidence хувь
- **Track ID**: `#1`, `#2` гэх мэт тогтвортой ID
- **Trail**: Объектын хөдөлгөөний мөр (fade effect-тэй)
- **HUD**: FPS, inference хугацаа, detection тоо, тоглуулах хурд

### 4.9 CLI Interface (`cli.py`)

```bash
# Webcam дээр бодит цагийн detection
python cli.py run --source 0 --model yolo26s.pt --save-video

# Нэг зураг дээр detection
python cli.py detect --image photo.jpg --output result.jpg --json

# API сервер эхлүүлэх
python cli.py serve --port 8000

# Өөр тохиргоотой
python cli.py run --config my_config.yaml --source video.mp4
```

---

## 5. Frontend — Next.js Web Application

### 5.1 Landing Page (`app/page.tsx`)

Төслийн нүүр хуудас:

- **Navigation Bar**: Лого, Live Camera, Try Demo товчууд
- **Hero Section**: Гарчиг, товч тайлбар, CTA товчууд
- **Stats Grid**: 80+ classes, <30ms inference, 30+ FPS, YOLO26
- **Footer**: Технологийн жагсаалт, GitHub линк

### 5.2 Video Upload Page (`app/demo/page.tsx`)

Хэрэглэгч видео файл сонгоно → видео тоглоно → бодит цагт detection overlay харуулна.

**Гол Онцлогууд**:
- Drag-and-drop файл upload
- `<video>` элементээр тоглуулалт
- Canvas дээр detection boxes overlay
- FPS, inference хугацаа, detection тоо харуулах
- Өнгөт bounding box (10 өнгө, class_id-аар)

**Техник Дэлгэрэнгүй**:
- **Capture Canvas**: Видеоны frame-г max 640px хэмжээнд хураангуйлж capture хийнэ
- **Display Canvas**: Видеоны full resolution дээр detection box-ыг зурна
- **Scale Factor**: Capture координатыг display координат руу хөрвүүлэх (`bbox / scale`)

### 5.3 Camera Page (`app/camera/page.tsx`)

Хөтөчийн камерыг `getUserMedia()` API-аар нээж, бодит цагт detection хийнэ.

**Гол Онцлогууд**:
- Камер эхлүүлэх/зогсоох
- Бодит цагийн detection overlay
- FPS, inference хугацаа, detection тоо
- Permission denied, backend unavailable алдааны зохицуулалт

### 5.4 Layout & Styles

- **`layout.tsx`**: Root layout, meta description, font тохиргоо
- **`globals.css`**: Light theme (цагаан дэвсгэр), gradient-text, glass effect, animation
- **Tailwind CSS**: Utility-first CSS, responsive design

---

## 6. YOLO26 Detection Pipeline

### 6.1 YOLO26 гэж юу вэ?

YOLO (You Only Look Once) нь нэг дамжуулалтаар (single forward pass) зураг дээрх бүх объектыг нэг дор илрүүлдэг deep learning загвар юм. YOLO26 нь энэ цувралын хамгийн сүүлийн хувилбар бөгөөд:

- **Нэг дамжуулалт**: Зургийг нэг удаа neural network-р дамжуулахад бүх объект олдоно
- **Anchor-free**: Урьдчилсан box хэмжээ шаардлагагүй
- **Multi-scale**: Олон хэмжээсийн feature map ашиглана
- **80 COCO анги**: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, TV, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

### 6.2 Загварын Хувилбарууд

| Загвар | Хэмжээ | Хурд (GPU) | Нарийвчлал | Тохиромжтой |
|--------|--------|-----------|-----------|-------------|
| `yolo26n.pt` | ~6MB | ~5ms | Бага | Гар утас, IoT төхөөрөмж |
| `yolo26s.pt` | ~20MB | ~10ms | Дунд | Бодит цагийн хэрэглээ (default) |
| `yolo26m.pt` | ~50MB | ~20ms | Сайн | Тэнцвэртэй хурд/нарийвчлал |
| `yolo26l.pt` | ~80MB | ~30ms | Маш сайн | Өндөр нарийвчлал шаардлагатай |

### 6.3 Inference Pipeline

```
Input Frame (BGR, any size)
    │
    ▼
Pre-processing
    ├── Resize to 640×640 (letterbox padding)
    ├── BGR → RGB
    ├── Normalize (0-255 → 0.0-1.0)
    └── HWC → CHW → Batch (1, 3, 640, 640)
    │
    ▼
YOLO26 Neural Network
    ├── Backbone: Feature extraction (CSPDarknet variant)
    ├── Neck: Feature pyramid network (FPN + PAN)
    └── Head: Multi-scale detection heads
    │
    ▼
Post-processing
    ├── Confidence threshold filter (default: 0.25)
    ├── Non-Maximum Suppression (NMS, IOU: 0.5)
    └── Coordinate rescaling (640×640 → original size)
    │
    ▼
Output: List[Detection(bbox, confidence, class_id, class_name)]
```

### 6.4 Tracking Pipeline

```
Frame N Detections
    │
    ▼
ByteTrack Algorithm
    ├── Hungarian matching (detection ↔ existing tracks)
    ├── High-confidence matches → Update track
    ├── Low-confidence matches → Re-match with unmatched tracks
    ├── Unmatched detections → Create new track
    └── Lost tracks → Mark as inactive after N frames
    │
    ▼
Output: Detection + track_id (persistent across frames)
```

---

## 7. Dual-Loop Rendering Architecture

### 7.1 Асуудал

Хэрэв видео тоглуулалт болон AI inference нэг loop-д ажиллавал:
- Inference хугацаа ~100-500ms (CPU дээр)
- Видео 2-5 FPS-ээр "гацаж" харагдана
- Хэрэглэгчийн туршлага муу

### 7.2 Шийдэл: Хоёр тусдаа Loop

```
┌─────────────────────────────────────┐
│        DISPLAY LOOP (60fps)         │
│  requestAnimationFrame()            │
│                                      │
│  1. video → display canvas зурах    │
│  2. detectionsRef.current авах      │
│  3. drawBoxes() — boxes overlay     │
│  4. Дараагийн frame хүсэх           │
│                                      │
│  Хязгаарлалт: Байхгүй — 60fps     │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│     DETECTION LOOP (background)     │
│  setTimeout() / async               │
│                                      │
│  1. video → capture canvas (640px)  │
│  2. canvas.toBlob() → JPEG          │
│  3. POST /detect-frame              │
│  4. Parse response headers          │
│  5. detectionsRef.current = dets    │
│  6. Stats update (FPS, ms, count)   │
│  7. setTimeout → давтах             │
│                                      │
│  Хурд: ~2-10fps (inference speed)  │
└─────────────────────────────────────┘
```

### 7.3 Coordinate Scaling

Capture canvas нь max 640px хэмжээтэй (YOLO-ийн img_size-тай тааруулсан), гэхдээ display canvas нь видеоны full resolution.

```typescript
// Capture хийхэд:
const scale = Math.min(1, 640 / video.videoWidth);
capture.width = video.videoWidth * scale;
capture.height = video.videoHeight * scale;

// Box зурахад:
const [x1, y1, x2, y2] = det.bbox.map((v) => v / scale);
// → Capture coordinates → Display coordinates
```

### 7.4 Давуу Тал

| Хуучин Арга | Dual-Loop |
|-------------|-----------|
| Video 2-5 FPS | Video 60 FPS |
| Inference блоклоно | Inference background-д |
| Видео гацна | Видео тасалдалгүй |
| UI respond хийхгүй | UI үргэлж мэдрэмжтэй |

---

## 8. API Reference

### 8.1 `GET /health`

Серверийн төлөв болон загварын мэдээллийг буцаана.

**Response:**
```json
{
  "status": "ok",
  "model": "yolo26s.pt",
  "device": "cpu",
  "num_classes": 80
}
```

### 8.2 `POST /detect`

Нэг зураг дээр detection хийнэ.

**Request:** `multipart/form-data` — `file` field (JPEG/PNG)

**Response:**
```json
{
  "inference_ms": 45.2,
  "num_detections": 3,
  "detections": [
    {
      "bbox": [120.5, 80.3, 340.1, 420.7],
      "confidence": 0.92,
      "class_id": 0,
      "class_name": "person",
      "track_id": null
    }
  ]
}
```

### 8.3 `POST /detect-frame`

Frame-by-frame detection — web frontend-д ашиглагдана.

**Request:** `multipart/form-data` — `file` field (JPEG frame)

**Response:**
- **Body**: Annotated JPEG зураг (bounding box зурагдсан)
- **Headers**:
  - `X-Inference-Ms`: `"45.2"`
  - `X-Detections`: `"3"`
  - `X-Detection-Data`: `"[{\"bbox\":[120,80,340,420],\"confidence\":0.92,\"class_id\":0,\"class_name\":\"person\"}]"`

### 8.4 `POST /detect-video`

Бүтэн видео файлыг боловсруулна.

**Request:** `multipart/form-data` — `file` field (MP4/AVI/MOV)

**Response:** Боловсруулсан видео файл + detection metadata

### 8.5 `POST /model`

Runtime-д загвар солих.

**Request:**
```json
{
  "model_name": "yolo26m.pt"
}
```

**Response:**
```json
{
  "status": "ok",
  "model": "yolo26m.pt",
  "device": "cpu",
  "num_classes": 80
}
```

---

## 9. Configuration System

### 9.1 `config/default.yaml`

Бүх тохиргооны параметрүүдийг нэг файлд хадгална:

```yaml
# ── Model ──
model:
  name: yolo26s.pt              # Загварын файл
  confidence: 0.25              # Confidence threshold
  iou_threshold: 0.5            # NMS IOU threshold
  device: "0"                   # "0" = GPU, "cpu" = CPU
  half: true                    # FP16 inference
  img_size: 640                 # Input resolution

# ── Stream ──
stream:
  sources: ["0"]                # Webcam index эсвэл файлын зам
  buffer_size: 2                # Frame buffer хэмжээ
  reconnect_delay: 5.0          # Дахин холбогдох хугацаа (секунд)
  max_retries: 10               # Дээд оролдлогын тоо

# ── Tracker ──
tracker:
  enabled: true
  type: bytetrack               # bytetrack / botsort

# ── Events ──
events:
  enabled: false
  roi_zones:
    - name: "zone1"
      points: [[100,100], [400,100], [400,400], [100,400]]
  line_crossing:
    enabled: false
    lines:
      - name: "line1"
        start: [0, 300]
        end: [640, 300]

# ── Visualization ──
visualization:
  show_window: true
  show_fps: true
  show_boxes: true
  show_labels: true
  show_confidence: true
  show_tracks: true
  box_thickness: 2
  font_scale: 0.6
  trail_length: 30

# ── Output ──
output:
  save_video: false
  video_path: "output/result.mp4"
  video_fps: 30
  codec: "mp4v"
  json_output: false
  json_path: "output/detections.json"

# ── Server ──
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1

# ── Logging ──
logging:
  level: "INFO"
  file: "logs/app.log"
  console: true
```

### 9.2 CLI Override

CLI-г ашиглан config параметрүүдийг дарж бичиж болно:

```bash
python cli.py run --source video.mp4 --model yolo26l.pt --save-video
# → config/default.yaml дахь source, model, save_video параметрүүдийг дарж бичнэ
```

---

## 10. Deployment

### 10.1 Docker

```dockerfile
FROM python:3.12-slim

# OpenCV-ийн system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# YOLO загварыг урьдчилан татах
RUN python -c "from ultralytics import YOLO; YOLO('yolo26s.pt')"

COPY . .
EXPOSE 8000

CMD ["python", "cli.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

**Docker командууд:**
```bash
# Build
docker build -t yolo-detect .

# Run (GPU)
docker run --gpus all -p 8000:8000 yolo-detect

# Run (CPU)
docker run -p 8000:8000 yolo-detect
```

### 10.2 Frontend Deployment (Vercel)

Next.js frontend-г Vercel дээр deploy хийх:

1. `web/` хавтсыг GitHub-аас Vercel-д холбох
2. Environment variable: `NEXT_PUBLIC_API_URL=https://your-backend.com`
3. Build command: `npm run build`
4. Output: Static export

### 10.3 Local Development

```bash
# Terminal 1: Backend
cd Biydaalt
python cli.py serve --port 8000

# Terminal 2: Frontend
cd Biydaalt/web
npm run dev

# Browser: http://localhost:3000
```

---

## 11. Хөгжүүлэлтийн Явц

### 11.1 Үе Шатууд

| Үе Шат | Хугацаа | Хийгдсэн Ажил |
|--------|---------|----------------|
| **1. Core Detection** | 1-р долоо хоног | YOLO11 detector, tracker, pipeline, CLI бүтэц |
| **2. API Layer** | 1-р долоо хоног | FastAPI endpoints, Pydantic schemas, CORS |
| **3. Web Frontend** | 2-р долоо хоног | Next.js landing page, video upload page |
| **4. Camera Feature** | 2-р долоо хоног | getUserMedia API, real-time camera detection |
| **5. UI Polish** | 2-р долоо хоног | Light theme, responsive design, statistics |
| **6. YOLO26 Upgrade** | 3-р долоо хоног | YOLO11 → YOLO26 шилжилт (загвар, config, UI) |
| **7. Performance** | 3-р долоо хоног | Dual-loop architecture, 640px capture cap, coordinate scaling |
| **8. Deployment** | 3-р долоо хоног | Docker, GitHub, Vercel, README |

### 11.2 Тулгарсан Асуудлууд ба Шийдлүүд

#### Асуудал 1: Video гацах (2 FPS)

**Шалтгаан**: Inference (~500ms CPU) бүрийг хүлээж байгаад дараагийн frame зурдаг байсан.

**Шийдэл**: Dual-loop architecture — display loop (60fps) + background detection loop тусад нь ажиллана. Видео тасалдалгүй тоглоно, detection box-ууд background-аас шинэчлэгдэнэ.

#### Асуудал 2: Detection box lag

**Шалтгаан**: `yolo26l.pt` загвар CPU дээр ~500ms inference авдаг → box-ууд объектоос хоцорно.

**Шийдэл**: 
1. `yolo26l.pt` → `yolo26s.pt` солих (~3x хурдан)
2. Capture resolution-г max 640px-ээр хязгаарлах
3. Box координатын scaling зөв хийх

#### Асуудал 3: CUDA байхгүй

**Шалтгаан**: NVIDIA GPU байгаа ч CUDA toolkit суугаагүй, CPU fallback.

**Шийдэл**: CPU дээр ажиллахаар оновчлох — жижиг загвар (yolo26s), бага resolution (640px).

#### Асуудал 4: GitHub 100MB file limit

**Шалтгаан**: `track.mp4` (137MB) git history-д оруулагдсан.

**Шийдэл**: `git rm --cached`, `.gitignore`-д `*.mp4` нэмэх, `git commit --amend`, `git push --force`.

### 11.3 Ашигласан Загварчлалын Хэв Маяг (Design Patterns)

| Pattern | Хэрэглээ |
|---------|----------|
| **Factory** | Detector загвар ачаалах, config-оос объект үүсгэх |
| **Observer** | Event system — ROI zone enter/exit мэдэгдэл |
| **Pipeline** | Stream → Detect → Track → Events → Visualize → Output |
| **Strategy** | Tracker type сонгох (ByteTrack / BoTSORT) |
| **Producer-Consumer** | FrameGrabber (producer) + Pipeline (consumer) FIFO queue-ээр |

---

## 12. Дүгнэлт

### 12.1 Бүтээгдэхүүний Үнэлгээ

Энэхүү төсөл нь **Computer Vision** хичээлийн хүрээнд бодит цагийн объект илрүүлэлтийн бүрэн функционал систем бүтээх зорилготой байсан бөгөөд дараах зорилтуудыг хангасан:

1. **YOLO26 загвар** — Хамгийн сүүлийн үеийн объект илрүүлэлтийн загвар хэрэглэсэн
2. **Full-stack application** — Backend API + Frontend Web Interface
3. **Бодит цагийн боловсруулалт** — Video болон камерын шууд detection
4. **Мэргэжлийн архитектур** — Modular, configurable, Docker-ready
5. **Performance оновчлол** — Dual-loop architecture, coordinate scaling, model tuning

### 12.2 Цаашдын Хөгжүүлэлт

- CUDA суулгаж GPU inference идэвхжүүлэх (~10x хурдасгал)
- Object counting, heatmap, dwell time шинжилгээ нэмэх
- WebSocket ашиглан HTTP overhead бууруулах
- Multi-camera dashboard бүтээх
- Mobile-responsive UI сайжруулах

### 12.3 Технологийн Нөлөө

Энэ төсөл нь Computer Vision, Deep Learning, Web Development, System Design зэрэг олон чиглэлийн мэдлэг, ур чадварыг нэгтгэж, бодит хэрэглээний түвшний систем бүтээсэн туршлага олгосон.

---

**Автор**: TheNasaFx  
**GitHub**: [Computer_vision_assignment](https://github.com/TheNasaFx/Computer_vision_assignment)  
**Хугацаа**: 2026 оны хавар  
**Хичээл**: Компьютер Хараа (FCS322)
