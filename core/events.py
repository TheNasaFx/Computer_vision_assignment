"""
Event Processor — ROI zone monitoring and line-crossing detection.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import cv2

from core.detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """A triggered event."""
    event_type: str           # "zone_enter", "zone_exit", "line_cross"
    track_id: int
    class_name: str
    zone_or_line: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "type": self.event_type,
            "track_id": self.track_id,
            "class_name": self.class_name,
            "zone_or_line": self.zone_or_line,
            "timestamp": round(self.timestamp, 4),
        }


class EventProcessor:
    """Region-of-interest and line-crossing event detection."""

    def __init__(self, cfg: dict):
        ecfg = cfg.get("events", {})
        self.enabled: bool = ecfg.get("enabled", False)
        self._zones: list[dict] = ecfg.get("roi_zones", [])
        lc = ecfg.get("line_crossing", {})
        self._line_crossing_enabled: bool = lc.get("enabled", False)
        self._lines: list[dict] = lc.get("lines", [])
        # track_id -> last known center
        self._prev_centers: dict[int, tuple[int, int]] = {}
        # track_id -> set of zones currently inside
        self._inside_zones: dict[int, set[str]] = {}

    def process(self, detections: list[Detection]) -> list[Event]:
        """Check all detections against zones and lines."""
        if not self.enabled:
            return []

        events: list[Event] = []
        current_centers: dict[int, tuple[int, int]] = {}

        for det in detections:
            if det.track_id is None:
                continue
            cx = int((det.bbox[0] + det.bbox[2]) / 2)
            cy = int((det.bbox[1] + det.bbox[3]) / 2)
            current_centers[det.track_id] = (cx, cy)

            # ── Zone checks ──
            for zone in self._zones:
                zone_name = zone.get("name", "zone")
                polygon = np.array(zone.get("points", []), dtype=np.int32)
                if len(polygon) < 3:
                    continue
                inside = cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0
                was_inside = zone_name in self._inside_zones.get(det.track_id, set())

                if inside and not was_inside:
                    events.append(Event("zone_enter", det.track_id,
                                        det.class_name, zone_name))
                    self._inside_zones.setdefault(det.track_id, set()).add(zone_name)
                elif not inside and was_inside:
                    events.append(Event("zone_exit", det.track_id,
                                        det.class_name, zone_name))
                    self._inside_zones.get(det.track_id, set()).discard(zone_name)

            # ── Line crossing checks ──
            if self._line_crossing_enabled and det.track_id in self._prev_centers:
                prev = self._prev_centers[det.track_id]
                curr = (cx, cy)
                for line in self._lines:
                    ls = tuple(line.get("start", [0, 0]))
                    le = tuple(line.get("end", [0, 0]))
                    if self._segments_intersect(prev, curr, ls, le):
                        events.append(Event("line_cross", det.track_id,
                                            det.class_name,
                                            line.get("name", "line")))

        self._prev_centers = current_centers

        for ev in events:
            logger.debug("Event: %s", ev)

        return events

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw ROI zones and lines on frame."""
        overlay = frame.copy()
        for zone in self._zones:
            pts = np.array(zone.get("points", []), dtype=np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(overlay, [pts], (0, 255, 0))
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)

        for line in self._lines:
            s = tuple(line.get("start", [0, 0]))
            e = tuple(line.get("end", [0, 0]))
            cv2.line(frame, s, e, (0, 0, 255), 2)
            name = line.get("name", "")
            if name:
                cv2.putText(frame, name, s, cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 1)
        return frame

    # ── geometry ───────────────────────────────────────────────

    @staticmethod
    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    @classmethod
    def _segments_intersect(cls, p1, p2, p3, p4) -> bool:
        d1 = cls._cross(p3, p4, p1)
        d2 = cls._cross(p3, p4, p2)
        d3 = cls._cross(p1, p2, p3)
        d4 = cls._cross(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False
