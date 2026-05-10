"""
Ownership Tracker — temporal, track-aware ownership inference.

Why this exists
---------------
The original ProximityEngine recomputes "nearest person" every frame
independently, which causes ownership flicker, flips between adjacent
persons, and a complete reset on a single missed track.

This module fixes that by accumulating per-frame proximity evidence into a
score per (object_track_id, person_track_id) pair across many frames. An
owner is *confirmed* only after enough cumulative evidence (and a clear
margin over the runner-up). Once confirmed, ownership is sticky — short
occlusions, brief flicker, or someone walking close don't steal it.

Optionally takes a per-person wrist position (pose keypoint) — a wrist
near an object is a much stronger ownership signal than the torso center.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

from core.detector import Detection

logger = logging.getLogger(__name__)


# COCO classes treated as personal study items by default
DEFAULT_OBJECT_CLASSES: set[str] = {
    "laptop", "cell phone", "book", "backpack", "handbag",
    "bottle", "cup", "mouse", "keyboard", "suitcase",
    "umbrella", "remote", "scissors",
}


def _centre(det: Detection) -> tuple[float, float]:
    return (det.bbox[0] + det.bbox[2]) / 2, (det.bbox[1] + det.bbox[3]) / 2


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class OwnershipMemory:
    """Per-object accumulated evidence."""
    object_track_id: int
    object_class: str
    candidate_scores: dict[int, float] = field(default_factory=dict)
    confirmed_owner_id: int | None = None
    confidence: float = 0.0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_owner_present: float = field(default_factory=time.time)
    last_owner_close: float = field(default_factory=time.time)
    last_object_bbox: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "object_track_id": self.object_track_id,
            "object_class": self.object_class,
            "owner_track_id": self.confirmed_owner_id,
            "confidence": round(self.confidence, 2),
            "candidate_scores": {
                pid: round(s, 1) for pid, s in self.candidate_scores.items()
            },
        }


@dataclass
class OwnershipFrameResult:
    """Result of one frame's ownership update."""
    memories: dict[int, OwnershipMemory]   # object_track_id → memory
    persons_present: dict[int, Detection]  # person_track_id → detection
    objects_present: dict[int, Detection]  # object_track_id → detection
    wrist_positions: dict[int, list[tuple[float, float]]] = field(default_factory=dict)

    def confirmed(self) -> dict[int, OwnershipMemory]:
        return {
            oid: m for oid, m in self.memories.items()
            if m.confirmed_owner_id is not None
        }

    def to_dict(self) -> dict:
        return {
            "objects": [m.to_dict() for m in self.memories.values()],
            "num_persons": len(self.persons_present),
            "num_tracked_objects": len(self.objects_present),
        }


class OwnershipTracker:
    """Stateful, track-aware ownership accumulator.

    Score model
    -----------
    Each frame, for every (object, person) pair within proximity range,
    the person's score for that object grows by:

        Δ = base_gain * max(0, 1 - center_dist / max_distance)

    If a wrist keypoint of the person lies within `wrist_distance` of the
    object centre, an additional `wrist_bonus` is added — this captures
    "the hand actually on/near the item" much better than torso position.

    All scores then decay multiplicatively (× decay) so old evidence fades.
    Scores are capped at `max_score`.

    Confirmation
    ------------
    The candidate with the highest score is confirmed as owner when both:
      - score ≥ confirmation_score
      - score ≥ margin_factor × second_best_score (clear leader)

    Once confirmed, the owner stays sticky until either:
      - their score drops below `min_keep_score` (e.g. they left long ago), OR
      - another candidate overtakes them by a clear margin.

    This is robust to: per-frame flicker, brief occlusion, two people
    standing equally close for a moment.
    """

    def __init__(self, cfg: dict):
        scfg = cfg.get("study_space", {})
        ocfg = scfg.get("ownership", {})

        self.max_distance: float = ocfg.get("max_distance", 240.0)
        self.wrist_distance: float = ocfg.get("wrist_distance", 90.0)
        self.base_gain: float = ocfg.get("base_gain", 1.0)
        self.wrist_bonus: float = ocfg.get("wrist_bonus", 2.5)
        self.decay: float = ocfg.get("decay", 0.94)
        self.max_score: float = ocfg.get("max_score", 30.0)
        self.confirmation_score: float = ocfg.get("confirmation_score", 6.0)
        self.margin_factor: float = ocfg.get("margin_factor", 1.6)
        self.min_keep_score: float = ocfg.get("min_keep_score", 2.0)
        self.memory_ttl_seconds: float = ocfg.get("memory_ttl_seconds", 12.0)

        self.object_classes: set[str] = set(
            scfg.get("object_classes", list(DEFAULT_OBJECT_CLASSES))
        )

        self._memories: dict[int, OwnershipMemory] = {}

    # ── public api ─────────────────────────────────────────────

    def update(
        self,
        detections: list[Detection],
        wrist_positions: dict[int, list[tuple[float, float]]] | None = None,
    ) -> OwnershipFrameResult:
        """Run one frame's update.

        Parameters
        ----------
        detections : detections from the current frame (must include track_id
                     for both persons and objects — i.e. came from
                     detect_and_track).
        wrist_positions : optional map of person_track_id → list of wrist
                          (x, y) positions, from a pose model. Empty/None
                          means "no pose info, fall back to torso centre".
        """
        wrist_positions = wrist_positions or {}
        now = time.time()

        # 1. Split detections by role
        persons: dict[int, Detection] = {}
        objects: dict[int, Detection] = {}
        for det in detections:
            if det.track_id is None:
                continue
            if det.class_name == "person":
                persons[det.track_id] = det
            elif det.class_name in self.object_classes:
                objects[det.track_id] = det

        # 2. Decay all scores (whether or not the object is visible — old
        #    evidence fades with time)
        for mem in self._memories.values():
            for pid in list(mem.candidate_scores.keys()):
                mem.candidate_scores[pid] *= self.decay
                if mem.candidate_scores[pid] < 0.05:
                    del mem.candidate_scores[pid]

        # 3. For each visible object, accumulate evidence from each person
        for oid, obj in objects.items():
            mem = self._memories.get(oid)
            if mem is None:
                mem = OwnershipMemory(
                    object_track_id=oid,
                    object_class=obj.class_name,
                    first_seen=now,
                )
                self._memories[oid] = mem

            mem.last_seen = now
            mem.last_object_bbox = list(obj.bbox)
            obj_c = _centre(obj)

            for pid, person in persons.items():
                # Centre-to-centre score component
                centre_d = _dist(obj_c, _centre(person))
                centre_gain = 0.0
                if centre_d <= self.max_distance:
                    centre_gain = self.base_gain * (1.0 - centre_d / self.max_distance)

                # Wrist proximity bonus (if pose available)
                wrist_gain = 0.0
                wrists = wrist_positions.get(pid, [])
                for w in wrists:
                    wd = _dist(obj_c, w)
                    if wd <= self.wrist_distance:
                        wrist_gain = max(
                            wrist_gain,
                            self.wrist_bonus * (1.0 - wd / self.wrist_distance),
                        )

                gain = centre_gain + wrist_gain
                if gain > 0:
                    mem.candidate_scores[pid] = min(
                        self.max_score,
                        mem.candidate_scores.get(pid, 0.0) + gain,
                    )

            self._reconfirm_owner(mem, persons, obj_c, now)

        # 4. For memories whose object is missing this frame, still
        #    re-evaluate ownership presence (in case owner is around the
        #    last known object location).
        for oid, mem in self._memories.items():
            if oid in objects:
                continue
            if mem.last_object_bbox:
                last_c = (
                    (mem.last_object_bbox[0] + mem.last_object_bbox[2]) / 2,
                    (mem.last_object_bbox[1] + mem.last_object_bbox[3]) / 2,
                )
                self._reconfirm_owner(mem, persons, last_c, now)

        # 5. Prune stale memories (object hasn't been seen for a while)
        stale = [
            oid for oid, mem in self._memories.items()
            if now - mem.last_seen > self.memory_ttl_seconds
        ]
        for oid in stale:
            del self._memories[oid]

        return OwnershipFrameResult(
            memories=dict(self._memories),
            persons_present=persons,
            objects_present=objects,
            wrist_positions=wrist_positions,
        )

    @property
    def memories(self) -> dict[int, OwnershipMemory]:
        return self._memories

    def reset(self) -> None:
        self._memories.clear()

    # ── internals ──────────────────────────────────────────────

    def _reconfirm_owner(
        self,
        mem: OwnershipMemory,
        persons: dict[int, Detection],
        ref_pos: tuple[float, float],
        now: float,
    ) -> None:
        """Re-pick the confirmed owner using current candidate_scores."""
        if not mem.candidate_scores:
            mem.confirmed_owner_id = None
            mem.confidence = 0.0
            return

        # Sort candidates by score, descending
        ranked = sorted(
            mem.candidate_scores.items(), key=lambda x: x[1], reverse=True
        )
        best_pid, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        prior_owner = mem.confirmed_owner_id

        # Sticky rule: if we had an owner and they still have decent score,
        # only switch when a challenger has a clear margin
        if prior_owner is not None:
            prior_score = mem.candidate_scores.get(prior_owner, 0.0)
            if prior_score >= self.min_keep_score:
                # Keep prior unless a challenger clearly beats them
                if best_pid != prior_owner:
                    if best_score < self.margin_factor * prior_score:
                        best_pid = prior_owner
                        best_score = prior_score
                        # Recompute second-best as the actual top non-prior
                        second_score = ranked[0][1]

        # Confirmation conditions
        confirmed = (
            best_score >= self.confirmation_score
            and best_score >= self.margin_factor * max(second_score, 1e-6)
        )

        if confirmed:
            mem.confirmed_owner_id = best_pid
            mem.confidence = min(1.0, best_score / self.max_score)
        elif prior_owner is not None and mem.candidate_scores.get(
            prior_owner, 0.0
        ) >= self.min_keep_score:
            # Hold previous owner under uncertainty
            mem.confidence = min(
                1.0, mem.candidate_scores[prior_owner] / self.max_score
            )
        else:
            mem.confirmed_owner_id = None
            mem.confidence = 0.0

        # Track owner-presence timestamps (used by AbandonmentMonitor)
        owner = mem.confirmed_owner_id
        if owner is not None and owner in persons:
            mem.last_owner_present = now
            person_c = _centre(persons[owner])
            if _dist(ref_pos, person_c) <= self.max_distance:
                mem.last_owner_close = now
