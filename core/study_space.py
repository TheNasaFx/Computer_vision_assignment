"""
Study Space Analyzer — main orchestrator.

Pipeline:
    Frame → (Detector + Tracker)        →  FrameResult (detections w/ track_id)
          → PoseEstimator                → wrist keypoints per person track_id
          → OwnershipTracker             → stateful per-object owner inference
          → AbandonmentMonitor           → PRESENT/AWAY/ABANDONED state machine
          → ZoneManager                  → desk/seat occupancy
          ─────────────────────────────────────────────────────────
                         → StudySpaceResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.detector import FrameResult, Detection
from core.pose import PoseEstimator, PoseResult
from core.ownership_tracker import (
    OwnershipTracker,
    OwnershipFrameResult,
    OwnershipMemory,
)
from core.abandonment import (
    AbandonmentMonitor,
    AbandonmentFrameResult,
    ItemState,
    ItemStatus,
)
from core.zones import ZoneManager, ZoneStatus

logger = logging.getLogger(__name__)


@dataclass
class PersonInventory:
    """Aggregated view: the items currently confirmed-owned by one person."""
    track_id: int | None
    person: Detection
    items: list[OwnershipMemory] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "person_bbox": [round(v, 1) for v in self.person.bbox],
            "num_items": len(self.items),
            "items": [
                {
                    "class_name": m.object_class,
                    "object_track_id": m.object_track_id,
                    "confidence": round(m.confidence, 2),
                    "bbox": [round(v, 1) for v in m.last_object_bbox],
                }
                for m in self.items
            ],
        }


@dataclass
class StudySpaceResult:
    """Single-frame analysis result."""
    frame_result: FrameResult
    pose_result: PoseResult
    ownership: OwnershipFrameResult
    abandonment: AbandonmentFrameResult
    zone_status: ZoneStatus
    inventories: list[PersonInventory]

    # Derived helpers (so the visualizer / frontend don't recompute)
    @property
    def confirmed_memories(self) -> list[OwnershipMemory]:
        return list(self.ownership.confirmed().values())

    @property
    def unclaimed_objects(self) -> list[OwnershipMemory]:
        """Tracked study items with no confirmed owner."""
        return [
            m for m in self.ownership.memories.values()
            if m.confirmed_owner_id is None and m.last_object_bbox
        ]

    def to_dict(self) -> dict:
        item_statuses = [
            s.to_dict() for s in self.abandonment.statuses.values()
        ]

        # Frontend convenience: frequently-used summary fields
        return {
            "inference_ms": round(self.frame_result.inference_ms, 2),
            "pose_inference_ms": round(self.pose_result.inference_ms, 2),
            "num_persons": len(self.ownership.persons_present),
            "num_objects": len(self.ownership.objects_present),
            "num_owned": len(self.confirmed_memories),
            "num_unclaimed": len(self.unclaimed_objects),
            "num_abandoned": len(self.abandonment.abandoned),
            "num_away": len(self.abandonment.away),
            "zones": self.zone_status.to_dict(),
            "inventories": [inv.to_dict() for inv in self.inventories],
            "unowned_objects": [
                {
                    "class_name": m.object_class,
                    "object_track_id": m.object_track_id,
                    "bbox": [round(v, 1) for v in m.last_object_bbox],
                }
                for m in self.unclaimed_objects
            ],
            "item_states": item_statuses,
            "new_alerts": [a.to_dict() for a in self.abandonment.new_alerts],
            "active_alerts": [a.to_dict() for a in self.abandonment.active_alerts],
        }


class StudySpaceAnalyzer:
    """Combines detection, pose, ownership, abandonment, and zones."""

    def __init__(self, cfg: dict, *, load_pose: bool = True):
        self.pose = PoseEstimator(cfg)
        if load_pose:
            self.pose.load()
        self.ownership_tracker = OwnershipTracker(cfg)
        self.abandonment_monitor = AbandonmentMonitor(cfg)
        self.zone_manager = ZoneManager(cfg)
        logger.info(
            "StudySpaceAnalyzer ready (pose=%s)",
            "on" if self.pose.available else "off",
        )

    def analyze(
        self, frame, frame_result: FrameResult
    ) -> StudySpaceResult:
        """Run full analysis. `frame` is the raw BGR ndarray (needed for pose)."""

        # 1. Pose — wrist positions per person track_id
        persons = [
            d for d in frame_result.detections if d.class_name == "person"
        ]
        pose_result = self.pose.estimate(frame, persons)

        # 2. Ownership update
        ownership = self.ownership_tracker.update(
            frame_result.detections,
            wrist_positions=pose_result.wrists,
        )

        # 3. Abandonment state machine
        abandonment = self.abandonment_monitor.update(ownership)

        # 4. Zone occupancy
        zone_status = self.zone_manager.update(list(ownership.persons_present.values()))

        # 5. Per-person inventories (only confirmed ownerships)
        inventories: list[PersonInventory] = []
        for pid, person in ownership.persons_present.items():
            owned: list[OwnershipMemory] = [
                m for m in ownership.memories.values()
                if m.confirmed_owner_id == pid
            ]
            inventories.append(PersonInventory(
                track_id=pid,
                person=person,
                items=owned,
            ))

        return StudySpaceResult(
            frame_result=frame_result,
            pose_result=pose_result,
            ownership=ownership,
            abandonment=abandonment,
            zone_status=zone_status,
            inventories=inventories,
        )

    def set_zones(self, zone_defs: list[dict]) -> None:
        self.zone_manager.set_zones(zone_defs)

    def reset(self) -> None:
        self.ownership_tracker.reset()
        self.abandonment_monitor.reset()
        self.zone_manager.reset()
