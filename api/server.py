"""
FastAPI server — REST endpoints for image/video inference, stream control, model swap.
"""

from __future__ import annotations

import io
import logging
import os
import threading
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.schemas import (
    InferenceResponse,
    ModelSwitchRequest,
    StatusResponse,
    StreamStartRequest,
)
from core.pipeline import DetectionPipeline

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Module-level pipeline reference — set by create_app()
_pipeline: DetectionPipeline | None = None
_stream_thread: threading.Thread | None = None

# Video processing jobs: job_id → {status, progress, ...}
_jobs: dict[str, dict] = {}


def create_app(cfg: dict) -> FastAPI:
    """Build FastAPI app wired to a DetectionPipeline."""
    global _pipeline

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _pipeline
        _pipeline = DetectionPipeline(cfg)
        _pipeline.detector.load()
        logger.info("API: detector loaded")
        yield
        if _pipeline is not None:
            _pipeline.shutdown()

    app = FastAPI(
        title="YOLO Real-Time Detection API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # ── CORS ───────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Serve processed videos ─────────────────────────────────
    app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

    # ── Health ─────────────────────────────────────────────────

    @app.get("/health", response_model=StatusResponse)
    async def health():
        assert _pipeline is not None
        return StatusResponse(
            status="ok",
            model=_pipeline.detector.model_name,
            device=_pipeline.detector.device,
            num_classes=len(_pipeline.detector.names),
        )

    # ── Image inference ────────────────────────────────────────

    @app.post("/detect", response_model=InferenceResponse)
    async def detect_image(file: UploadFile = File(...)):
        """Run detection on an uploaded image."""
        assert _pipeline is not None
        contents = await file.read()
        arr = np.frombuffer(contents, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(400, "Invalid image file")

        result = _pipeline.infer_image(frame)
        return InferenceResponse(
            inference_ms=round(result.inference_ms, 2),
            num_detections=len(result.detections),
            detections=[d.to_dict() for d in result.detections],
        )

    # ── Model switch ───────────────────────────────────────────

    @app.post("/model", response_model=StatusResponse)
    async def switch_model(req: ModelSwitchRequest):
        assert _pipeline is not None
        _pipeline.detector.switch_model(req.model_name)
        return StatusResponse(
            status="model_switched",
            model=_pipeline.detector.model_name,
            device=_pipeline.detector.device,
            num_classes=len(_pipeline.detector.names),
        )

    # ── Stream control ─────────────────────────────────────────

    @app.post("/stream/start")
    async def stream_start(req: StreamStartRequest):
        global _stream_thread
        assert _pipeline is not None
        if _stream_thread is not None and _stream_thread.is_alive():
            return JSONResponse({"status": "already_running"})

        _pipeline.setup(sources=[req.source])
        _stream_thread = threading.Thread(target=_pipeline.run, daemon=True)
        _stream_thread.start()
        return JSONResponse({"status": "started", "source": req.source})

    @app.post("/stream/stop")
    async def stream_stop():
        assert _pipeline is not None
        _pipeline.shutdown()
        return JSONResponse({"status": "stopped"})

    # ── Video file detection ───────────────────────────────────

    @app.post("/detect-video")
    async def detect_video(file: UploadFile = File(...)):
        """Process uploaded video with YOLO and return annotated video + stats."""
        assert _pipeline is not None

        # Validate file type
        if file.content_type and not file.content_type.startswith("video/"):
            raise HTTPException(400, "File must be a video")

        job_id = uuid.uuid4().hex[:10]
        input_path = OUTPUT_DIR / f"input_{job_id}.mp4"
        output_path = OUTPUT_DIR / f"result_{job_id}.mp4"

        # Save uploaded file (limit: 200MB)
        content = await file.read()
        if len(content) > 200 * 1024 * 1024:
            raise HTTPException(413, "Video file too large (max 200MB)")
        input_path.write_bytes(content)

        _jobs[job_id] = {"status": "processing", "progress": 0}

        try:
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise HTTPException(400, "Could not read video file")

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

            all_detections: list[dict] = []
            class_counts: dict[str, int] = {}
            total_inf_ms = 0.0
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                result = _pipeline.detector.detect(frame)
                vis = _pipeline.visualizer.draw(frame, result, {}, 0.0,
                                                playback_speed=1.0, paused=False)
                writer.write(vis)

                for d in result.detections:
                    dd = d.to_dict()
                    dd["frame"] = frame_count
                    all_detections.append(dd)
                    name = d.class_name
                    class_counts[name] = class_counts.get(name, 0) + 1

                total_inf_ms += result.inference_ms
                frame_count += 1

                if total_frames > 0:
                    _jobs[job_id]["progress"] = int(frame_count / total_frames * 100)

            cap.release()
            writer.release()
            input_path.unlink(missing_ok=True)

            avg_inf = total_inf_ms / max(frame_count, 1)
            _jobs[job_id] = {"status": "done", "progress": 100}

            return JSONResponse({
                "job_id": job_id,
                "video_url": f"/output/result_{job_id}.mp4",
                "stats": {
                    "total_frames": frame_count,
                    "total_detections": len(all_detections),
                    "avg_inference_ms": round(avg_inf, 1),
                    "fps": round(fps, 1),
                    "resolution": f"{w}x{h}",
                    "class_counts": dict(sorted(class_counts.items(),
                                                key=lambda x: x[1], reverse=True)),
                },
            })
        except HTTPException:
            raise
        except Exception as e:
            _jobs[job_id] = {"status": "error", "error": str(e)}
            input_path.unlink(missing_ok=True)
            logger.exception("Video processing failed")
            raise HTTPException(500, f"Processing failed: {e}")

    @app.get("/detect-video/{job_id}/status")
    async def video_job_status(job_id: str):
        """Poll processing progress."""
        if job_id not in _jobs:
            raise HTTPException(404, "Job not found")
        return JSONResponse(_jobs[job_id])

    # ── Single frame detection (for live camera) ───────────────

    @app.post("/detect-frame")
    async def detect_frame(file: UploadFile = File(...)):
        """Detect objects on a single camera frame.
        Returns annotated JPEG image with detection metadata in headers."""
        assert _pipeline is not None
        contents = await file.read()
        arr = np.frombuffer(contents, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(400, "Invalid image")

        result = _pipeline.detector.detect(frame)
        vis = _pipeline.visualizer.draw(frame, result, {}, 0.0,
                                        playback_speed=1.0, paused=False)

        _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])

        import json as _json
        det_summary = _json.dumps({
            "inference_ms": round(result.inference_ms, 1),
            "num_detections": len(result.detections),
            "detections": [d.to_dict() for d in result.detections],
        })

        return StreamingResponse(
            io.BytesIO(buf.tobytes()),
            media_type="image/jpeg",
            headers={
                "X-Inference-Ms": str(round(result.inference_ms, 1)),
                "X-Detections": str(len(result.detections)),
                "X-Detection-Data": det_summary,
                "Access-Control-Expose-Headers": "X-Inference-Ms, X-Detections, X-Detection-Data",
            },
        )

    return app
