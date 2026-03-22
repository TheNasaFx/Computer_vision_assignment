"""
Object Tracker — wraps Ultralytics built-in ByteTrack / BoTSORT.
Maintains persistent IDs and track history (trails).
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from core.detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class Track:
    track_id: int
    class_name: str
    history: deque  # deque of (cx, cy) center points


class ObjectTracker:
    """Lightweight wrapper that stores per-ID history for trail drawing."""

    def __init__(self, cfg: dict):
        tcfg = cfg.get("tracker", {})
        self.enabled: bool = tcfg.get("enabled", True)
        self.trail_length: int = cfg.get("visualization", {}).get("trail_length", 30)
        self._tracks: dict[int, Track] = {}

    def update(self, detections: list[Detection]) -> dict[int, Track]:
        """Update track history from current frame detections."""
        seen: set[int] = set()
        for det in detections:
            tid = det.track_id
            if tid is None:
                continue
            seen.add(tid)
            cx = (det.bbox[0] + det.bbox[2]) / 2
            cy = (det.bbox[1] + det.bbox[3]) / 2
            if tid not in self._tracks:
                self._tracks[tid] = Track(
                    track_id=tid,
                    class_name=det.class_name,
                    history=deque(maxlen=self.trail_length),
                )
            self._tracks[tid].history.append((int(cx), int(cy)))
            self._tracks[tid].class_name = det.class_name

        # Prune stale tracks
        stale = [tid for tid in self._tracks if tid not in seen]
        for tid in stale:
            if len(self._tracks[tid].history) > 0:
                self._tracks[tid].history.popleft()
            if len(self._tracks[tid].history) == 0:
                del self._tracks[tid]

        return self._tracks

    @property
    def tracks(self) -> dict[int, Track]:
        return self._tracks

    def reset(self) -> None:
        self._tracks.clear()
