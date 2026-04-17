"""
Proximity Engine — assigns ownership of detected objects to the nearest person.

Uses Euclidean distance between bounding-box centres to decide which objects
"belong to" which person (e.g. the laptop on a desk is owned by the person
sitting closest to it).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from core.detector import Detection

logger = logging.getLogger(__name__)

# COCO classes that are typical personal belongings in a study space
STUDY_OBJECT_CLASSES: set[str] = {
    "laptop", "cell phone", "book", "backpack", "handbag",
    "bottle", "cup", "mouse", "keyboard", "suitcase",
    "umbrella", "remote", "scissors",
}


def _centre(det: Detection) -> tuple[float, float]:
    """Return (cx, cy) of a detection's bounding box."""
    return (det.bbox[0] + det.bbox[2]) / 2, (det.bbox[1] + det.bbox[3]) / 2


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class OwnershipLink:
    """A single object→person ownership link."""
    object_det: Detection
    person_det: Detection
    distance: float

    def to_dict(self) -> dict:
        return {
            "object": self.object_det.to_dict(),
            "person_track_id": self.person_det.track_id,
            "distance": round(self.distance, 1),
        }


@dataclass
class ProximityResult:
    """Result of a single-frame proximity analysis."""
    persons: list[Detection] = field(default_factory=list)
    ownership: dict[int, list[OwnershipLink]] = field(default_factory=dict)
    unowned_objects: list[Detection] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "num_persons": len(self.persons),
            "ownership": {
                pid: [link.to_dict() for link in links]
                for pid, links in self.ownership.items()
            },
            "unowned_objects": [d.to_dict() for d in self.unowned_objects],
        }


class ProximityEngine:
    """Assigns detected objects to the nearest person within a distance threshold."""

    def __init__(self, cfg: dict):
        scfg = cfg.get("study_space", {})
        self.max_distance: float = scfg.get("ownership_max_distance", 200.0)
        self.object_classes: set[str] = set(
            scfg.get("object_classes", list(STUDY_OBJECT_CLASSES))
        )

    def analyze(self, detections: list[Detection]) -> ProximityResult:
        """Split detections into persons / objects and assign ownership."""
        persons = [d for d in detections if d.class_name == "person"]
        objects = [d for d in detections if d.class_name in self.object_classes]

        ownership: dict[int, list[OwnershipLink]] = {}
        unowned: list[Detection] = []

        for obj in objects:
            nearest, dist = self._nearest_person(obj, persons)
            if nearest is not None and dist <= self.max_distance:
                pid = nearest.track_id if nearest.track_id is not None else id(nearest)
                link = OwnershipLink(object_det=obj, person_det=nearest, distance=dist)
                ownership.setdefault(pid, []).append(link)
            else:
                unowned.append(obj)

        return ProximityResult(
            persons=persons,
            ownership=ownership,
            unowned_objects=unowned,
        )

    # ── helpers ────────────────────────────────────────────────

    @staticmethod
    def _nearest_person(
        obj: Detection, persons: list[Detection]
    ) -> tuple[Detection | None, float]:
        """Find the person closest to *obj*. Returns (person, distance)."""
        if not persons:
            return None, float("inf")
        obj_c = _centre(obj)
        best, best_dist = None, float("inf")
        for p in persons:
            d = _distance(obj_c, _centre(p))
            if d < best_dist:
                best, best_dist = p, d
        return best, best_dist
