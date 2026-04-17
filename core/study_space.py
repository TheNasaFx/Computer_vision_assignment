"""
Study Space Analyzer — main orchestrator that combines detection with
proximity ownership, zone occupancy, and alert management.

Usage (API / pipeline):
    analyzer = StudySpaceAnalyzer(cfg)
    result   = analyzer.analyze(frame_result)   # → StudySpaceResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.detector import FrameResult, Detection
from core.proximity import ProximityEngine, ProximityResult, OwnershipLink
from core.zones import ZoneManager, ZoneStatus
from core.alerts import AlertManager, Alert

logger = logging.getLogger(__name__)


@dataclass
class PersonInventory:
    """Summary of a single person and their belongings."""
    track_id: int | None
    person: Detection
    items: list[OwnershipLink] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "person_bbox": [round(v, 1) for v in self.person.bbox],
            "num_items": len(self.items),
            "items": [
                {
                    "class_name": link.object_det.class_name,
                    "confidence": round(link.object_det.confidence, 2),
                    "distance": round(link.distance, 1),
                    "bbox": [round(v, 1) for v in link.object_det.bbox],
                }
                for link in self.items
            ],
        }


@dataclass
class StudySpaceResult:
    """Complete result of a single-frame study-space analysis."""
    frame_result: FrameResult
    proximity: ProximityResult
    zone_status: ZoneStatus
    new_alerts: list[Alert]
    all_alerts: list[Alert]
    inventories: list[PersonInventory]

    def to_dict(self) -> dict:
        return {
            "inference_ms": round(self.frame_result.inference_ms, 2),
            "num_persons": len(self.proximity.persons),
            "num_objects": sum(
                len(links) for links in self.proximity.ownership.values()
            ) + len(self.proximity.unowned_objects),
            "zones": self.zone_status.to_dict(),
            "inventories": [inv.to_dict() for inv in self.inventories],
            "unowned_objects": [d.to_dict() for d in self.proximity.unowned_objects],
            "new_alerts": [a.to_dict() for a in self.new_alerts],
            "active_alerts": [a.to_dict() for a in self.all_alerts],
        }


class StudySpaceAnalyzer:
    """Combines YOLO detections with proximity, zones, and alerts."""

    def __init__(self, cfg: dict):
        self.proximity_engine = ProximityEngine(cfg)
        self.zone_manager = ZoneManager(cfg)
        self.alert_manager = AlertManager(cfg)
        logger.info("StudySpaceAnalyzer initialized")

    def analyze(self, frame_result: FrameResult) -> StudySpaceResult:
        """Run full study-space analysis on a single frame's detections."""

        # 1. Proximity — assign objects to nearest person
        prox = self.proximity_engine.analyze(frame_result.detections)

        # 2. Zone occupancy
        zone_status = self.zone_manager.update(prox.persons)

        # 3. Alerts — check unattended items
        new_alerts = self.alert_manager.update(prox.unowned_objects)

        # 4. Build per-person inventory
        inventories: list[PersonInventory] = []
        for person in prox.persons:
            pid = person.track_id if person.track_id is not None else id(person)
            links = prox.ownership.get(pid, [])
            inventories.append(PersonInventory(
                track_id=person.track_id,
                person=person,
                items=links,
            ))

        return StudySpaceResult(
            frame_result=frame_result,
            proximity=prox,
            zone_status=zone_status,
            new_alerts=new_alerts,
            all_alerts=self.alert_manager.active_alerts,
            inventories=inventories,
        )

    def set_zones(self, zone_defs: list[dict]) -> None:
        """Update desk zone definitions at runtime."""
        self.zone_manager.set_zones(zone_defs)

    def reset(self) -> None:
        """Reset all state."""
        self.zone_manager.reset()
        self.alert_manager.reset()
