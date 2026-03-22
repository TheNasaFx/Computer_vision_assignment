"""
YOLO Inference Engine — GPU-accelerated, FP16-capable detector.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detection result."""

    bbox: list[float]          # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    track_id: int | None = None
    mask: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": [round(v, 1) for v in self.bbox],
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "track_id": self.track_id,
        }


@dataclass
class FrameResult:
    """Result for a single frame."""

    detections: list[Detection]
    inference_ms: float
    frame_id: int = 0
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp": round(self.timestamp, 4),
            "inference_ms": round(self.inference_ms, 2),
            "num_detections": len(self.detections),
            "detections": [d.to_dict() for d in self.detections],
        }


class Detector:
    """YOLO-based object detector with GPU/FP16 support."""

    def __init__(self, cfg: dict):
        model_cfg = cfg.get("model", {})
        self.model_name: str = model_cfg.get("name", "yolo11n.pt")
        self.confidence: float = model_cfg.get("confidence", 0.35)
        self.iou_threshold: float = model_cfg.get("iou_threshold", 0.45)
        self.max_det: int = model_cfg.get("max_detections", 100)
        requested_device: str = model_cfg.get("device", "0")
        # Auto-detect: use GPU only if CUDA is actually available
        import torch
        if requested_device != "cpu" and not torch.cuda.is_available():
            logger.warning("CUDA not available — falling back to CPU (half disabled)")
            self.device = "cpu"
            self.half = False
        else:
            self.device = requested_device
            self.half = model_cfg.get("half", True)
        self.img_size: int = model_cfg.get("img_size", 640)
        self.classes: list[int] | None = model_cfg.get("classes", None)

        self._model: YOLO | None = None
        self._names: dict[int, str] = {}

    # ── lifecycle ──────────────────────────────────────────────

    def load(self) -> None:
        """Load model onto device with optional FP16."""
        logger.info("Loading model: %s  device=%s  half=%s",
                     self.model_name, self.device, self.half)
        self._model = YOLO(self.model_name)
        # Warm-up: single dummy inference to allocate CUDA memory
        dummy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        self._model.predict(
            dummy,
            device=self.device,
            half=self.half,
            imgsz=self.img_size,
            verbose=False,
        )
        self._names = self._model.names
        logger.info("Model loaded — %d classes, warm-up done.", len(self._names))

    def switch_model(self, model_name: str) -> None:
        """Hot-swap to a different model variant."""
        logger.info("Switching model to %s", model_name)
        self.model_name = model_name
        self.load()

    @property
    def names(self) -> dict[int, str]:
        return self._names

    # ── inference ──────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> FrameResult:
        """Run detection on a single BGR frame."""
        assert self._model is not None, "Call load() first."

        t0 = time.perf_counter()
        results = self._model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            max_det=self.max_det,
            device=self.device,
            half=self.half,
            imgsz=self.img_size,
            classes=self.classes,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        detections = self._parse_results(results[0])
        return FrameResult(detections=detections, inference_ms=elapsed_ms)

    def detect_and_track(self, frame: np.ndarray,
                         tracker_type: str = "bytetrack",
                         persist: bool = True) -> FrameResult:
        """Run detection + built-in tracking."""
        assert self._model is not None, "Call load() first."

        t0 = time.perf_counter()
        results = self._model.track(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            max_det=self.max_det,
            device=self.device,
            half=self.half,
            imgsz=self.img_size,
            classes=self.classes,
            tracker=f"{tracker_type}.yaml",
            persist=persist,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        detections = self._parse_results(results[0], with_tracks=True)
        return FrameResult(detections=detections, inference_ms=elapsed_ms)

    # ── helpers ────────────────────────────────────────────────

    def _parse_results(self, result, with_tracks: bool = False) -> list[Detection]:
        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxys = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if with_tracks and boxes.id is not None else [None] * len(xyxys)

        for xyxy, conf, cls, tid in zip(xyxys, confs, clss, ids):
            detections.append(Detection(
                bbox=xyxy.tolist(),
                confidence=float(conf),
                class_id=int(cls),
                class_name=self._names.get(int(cls), str(cls)),
                track_id=int(tid) if tid is not None else None,
            ))
        return detections
