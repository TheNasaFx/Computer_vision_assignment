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


# ── Study Space schemas ───────────────────────────────────────

class OwnershipItem(BaseModel):
    class_name: str
    confidence: float
    distance: float
    bbox: list[float]


class PersonInventoryOut(BaseModel):
    track_id: int | None
    person_bbox: list[float]
    num_items: int
    items: list[OwnershipItem]


class AlertOut(BaseModel):
    alert_id: str
    alert_type: str
    severity: str
    message: str
    timestamp: float
    bbox: list[float] | None = None
    class_name: str | None = None


class ZoneOut(BaseModel):
    zone_id: str
    name: str
    occupied: bool
    occupant_id: int | None = None
    occupied_seconds: float = 0.0


class ZoneStatusOut(BaseModel):
    total: int
    occupied: int
    available: int
    zones: list[ZoneOut]


class StudySpaceResponse(BaseModel):
    inference_ms: float
    num_persons: int
    num_objects: int
    zones: ZoneStatusOut
    inventories: list[PersonInventoryOut]
    unowned_objects: list[DetectionOut]
    new_alerts: list[AlertOut]
    active_alerts: list[AlertOut]


class ZoneDefinition(BaseModel):
    id: str = Field(..., examples=["desk_1"])
    name: str = Field(..., examples=["Desk 1"])
    points: list[list[int]] = Field(..., examples=[[[100, 200], [400, 200], [400, 500], [100, 500]]])


class SetZonesRequest(BaseModel):
    zones: list[ZoneDefinition]
