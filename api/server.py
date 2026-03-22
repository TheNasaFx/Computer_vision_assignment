"""
FastAPI server — REST endpoints for image inference, stream control, model swap.
"""

from __future__ import annotations

import io
import logging
import threading
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from api.schemas import (
    InferenceResponse,
    ModelSwitchRequest,
    StatusResponse,
    StreamStartRequest,
)
from core.pipeline import DetectionPipeline

logger = logging.getLogger(__name__)

# Module-level pipeline reference — set by create_app()
_pipeline: DetectionPipeline | None = None
_stream_thread: threading.Thread | None = None


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

    return app
