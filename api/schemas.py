"""
Pydantic schemas for the FastAPI layer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DetectionOut(BaseModel):
    bbox: list[float]
    confidence: float
    class_id: int
    class_name: str
    track_id: int | None = None


class InferenceResponse(BaseModel):
    inference_ms: float
    num_detections: int
    detections: list[DetectionOut]


class ModelSwitchRequest(BaseModel):
    model_name: str = Field(..., examples=["yolo26n.pt", "yolo26s.pt", "yolo26l.pt"])


class StatusResponse(BaseModel):
    status: str
    model: str
    device: str
    num_classes: int


class StreamStartRequest(BaseModel):
    source: str = Field(..., examples=["0", "rtsp://..."])
