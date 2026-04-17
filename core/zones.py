"""
Zone Manager — defines desk / seat zones and tracks their occupancy state.

Each zone is a polygon region in the frame.  When a detected person's centre
falls inside a zone polygon, that zone is marked *occupied*.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from core.detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class DeskZone:
    """A single desk / seat zone."""
    zone_id: str
    name: str
    points: list[list[int]]        # [[x1,y1], [x2,y2], ...]
    occupied: bool = False
    occupant_id: int | None = None
    occupied_since: float | None = None
    total_occupied_seconds: float = 0.0

    # ── helpers ────────────────────────────────────────────────

    def polygon(self) -> np.ndarray:
        return np.array(self.points, dtype=np.int32)

    def contains(self, cx: float, cy: float) -> bool:
        return cv2.pointPolygonTest(self.polygon(), (cx, cy), False) >= 0

    def to_dict(self) -> dict:
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "occupied": self.occupied,
            "occupant_id": self.occupant_id,
            "occupied_seconds": round(self.total_occupied_seconds, 1),
        }


@dataclass
class ZoneStatus:
    """Snapshot of all zones after a single update."""
    zones: list[DeskZone]
    total: int = 0
    occupied: int = 0
    available: int = 0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "occupied": self.occupied,
            "available": self.available,
            "zones": [z.to_dict() for z in self.zones],
        }


class ZoneManager:
    """Manages a set of desk / seat zones and their occupancy."""

    def __init__(self, cfg: dict):
        scfg = cfg.get("study_space", {})
        raw_zones = scfg.get("zones", [])
        self._zones: list[DeskZone] = []
        for i, z in enumerate(raw_zones):
            self._zones.append(DeskZone(
                zone_id=z.get("id", f"zone_{i}"),
                name=z.get("name", f"Desk {i + 1}"),
                points=z.get("points", []),
            ))
        self._last_update: float = time.time()

    @property
    def zones(self) -> list[DeskZone]:
        return self._zones

    def update(self, persons: list[Detection]) -> ZoneStatus:
        """Update zone occupancy based on detected persons."""
        now = time.time()
        dt = now - self._last_update
        self._last_update = now

        for zone in self._zones:
            prev_occupied = zone.occupied
            zone.occupied = False
            zone.occupant_id = None

            for p in persons:
                cx = (p.bbox[0] + p.bbox[2]) / 2
                cy = (p.bbox[1] + p.bbox[3]) / 2
                if zone.contains(cx, cy):
                    zone.occupied = True
                    zone.occupant_id = p.track_id
                    break  # one person per zone

            if zone.occupied:
                if not prev_occupied:
                    zone.occupied_since = now
                zone.total_occupied_seconds += dt
            else:
                zone.occupied_since = None

        occ = sum(1 for z in self._zones if z.occupied)
        return ZoneStatus(
            zones=list(self._zones),
            total=len(self._zones),
            occupied=occ,
            available=len(self._zones) - occ,
        )

    def set_zones(self, zone_defs: list[dict]) -> None:
        """Replace zone definitions at runtime (e.g. from API)."""
        self._zones.clear()
        for i, z in enumerate(zone_defs):
            self._zones.append(DeskZone(
                zone_id=z.get("id", f"zone_{i}"),
                name=z.get("name", f"Desk {i + 1}"),
                points=z.get("points", []),
            ))
        logger.info("Zones updated: %d zones", len(self._zones))

    def reset(self) -> None:
        for z in self._zones:
            z.occupied = False
            z.occupant_id = None
            z.occupied_since = None
            z.total_occupied_seconds = 0.0
