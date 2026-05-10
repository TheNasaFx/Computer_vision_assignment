"""
Abandonment Monitor — turns OwnershipTracker memories into a per-object
state machine and emits alerts on transitions.

States
------
    PRESENT    : owner is in frame and within proximity of the object.
    AWAY       : owner not visible OR has moved away from the object,
                 but not long enough to count as abandonment.
    ABANDONED  : owner has been AWAY for more than the configured
                 abandonment threshold.
    RECOVERED  : object was previously ABANDONED and the owner has
                 returned (transient state, lasts one update).
    UNCLAIMED  : the object has appeared but no owner has been confirmed
                 (also covers items that lost their owner via score decay).

Why a state machine
-------------------
The original AlertManager keyed by a fragile spatial bucket and flipped
state on noise. Here, transitions are derived from clean per-object
timestamps maintained by OwnershipTracker, and each transition emits a
single alert with full provenance ("Person #5's backpack abandoned 32s
ago"). RECOVERED gives the operator a positive signal too.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

from core.ownership_tracker import OwnershipMemory, OwnershipFrameResult

logger = logging.getLogger(__name__)


class ItemState(str, Enum):
    PRESENT = "present"
    AWAY = "away"
    ABANDONED = "abandoned"
    RECOVERED = "recovered"
    UNCLAIMED = "unclaimed"


@dataclass
class ItemStatus:
    """Snapshot of one object's abandonment state for one frame."""
    object_track_id: int
    object_class: str
    owner_track_id: int | None
    confidence: float
    state: ItemState
    seconds_owner_away: float
    seconds_owner_far: float
    countdown_seconds: float        # seconds until ABANDONED triggers
    last_known_bbox: list[float]
    owner_present_in_frame: bool

    def to_dict(self) -> dict:
        return {
            "object_track_id": self.object_track_id,
            "object_class": self.object_class,
            "owner_track_id": self.owner_track_id,
            "confidence": round(self.confidence, 2),
            "state": self.state.value,
            "seconds_owner_away": round(self.seconds_owner_away, 1),
            "seconds_owner_far": round(self.seconds_owner_far, 1),
            "countdown_seconds": round(self.countdown_seconds, 1),
            "last_known_bbox": [round(v, 1) for v in self.last_known_bbox],
            "owner_present_in_frame": self.owner_present_in_frame,
        }


@dataclass
class AbandonmentAlert:
    """Emitted once on a meaningful state transition."""
    alert_id: str
    transition: str        # e.g. "AWAY→ABANDONED"
    severity: str          # info | warning | critical
    object_track_id: int
    object_class: str
    owner_track_id: int | None
    seconds_owner_away: float
    timestamp: float
    bbox: list[float]
    message: str

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "transition": self.transition,
            "severity": self.severity,
            "object_track_id": self.object_track_id,
            "object_class": self.object_class,
            "owner_track_id": self.owner_track_id,
            "seconds_owner_away": round(self.seconds_owner_away, 1),
            "timestamp": round(self.timestamp, 2),
            "bbox": [round(v, 1) for v in self.bbox],
            "message": self.message,
        }


@dataclass
class AbandonmentFrameResult:
    statuses: dict[int, ItemStatus] = field(default_factory=dict)
    new_alerts: list[AbandonmentAlert] = field(default_factory=list)
    active_alerts: list[AbandonmentAlert] = field(default_factory=list)

    @property
    def abandoned(self) -> list[ItemStatus]:
        return [s for s in self.statuses.values() if s.state == ItemState.ABANDONED]

    @property
    def away(self) -> list[ItemStatus]:
        return [s for s in self.statuses.values() if s.state == ItemState.AWAY]

    def to_dict(self) -> dict:
        return {
            "statuses": [s.to_dict() for s in self.statuses.values()],
            "new_alerts": [a.to_dict() for a in self.new_alerts],
            "active_alerts": [a.to_dict() for a in self.active_alerts],
            "num_abandoned": len(self.abandoned),
            "num_away": len(self.away),
        }


class AbandonmentMonitor:
    """Track each object's PRESENT/AWAY/ABANDONED/RECOVERED state."""

    def __init__(self, cfg: dict):
        scfg = cfg.get("study_space", {}).get("abandonment", {})
        # Time owner can be missing/far before we say AWAY (vs. PRESENT)
        self.presence_grace_seconds: float = scfg.get("presence_grace_seconds", 1.5)
        # Time AWAY before we promote to ABANDONED
        self.abandoned_after_seconds: float = scfg.get("abandoned_after_seconds", 20.0)
        # Time ABANDONED before we escalate to "critical"
        self.critical_after_seconds: float = scfg.get("critical_after_seconds", 60.0)
        # How many alerts to keep
        self.max_active_alerts: int = scfg.get("max_active_alerts", 20)

        self._prev_state: dict[int, ItemState] = {}
        self._active_alerts: list[AbandonmentAlert] = []
        self._counter: int = 0

    def update(self, ownership: OwnershipFrameResult) -> AbandonmentFrameResult:
        now = time.time()
        result = AbandonmentFrameResult()

        for oid, mem in ownership.memories.items():
            status = self._compute_status(mem, ownership.persons_present, now)
            result.statuses[oid] = status

            prev = self._prev_state.get(oid)
            if prev != status.state:
                alert = self._maybe_emit_alert(prev, status, now)
                if alert is not None:
                    result.new_alerts.append(alert)
                    self._active_alerts.append(alert)
                self._prev_state[oid] = status.state

        # Drop transition memory for objects no longer tracked
        gone = [oid for oid in self._prev_state if oid not in ownership.memories]
        for oid in gone:
            del self._prev_state[oid]

        # Keep alert ring buffer trimmed
        if len(self._active_alerts) > self.max_active_alerts:
            self._active_alerts = self._active_alerts[-self.max_active_alerts:]
        result.active_alerts = list(self._active_alerts)

        return result

    @property
    def active_alerts(self) -> list[AbandonmentAlert]:
        return list(self._active_alerts)

    def reset(self) -> None:
        self._prev_state.clear()
        self._active_alerts.clear()
        self._counter = 0

    # ── internals ──────────────────────────────────────────────

    def _compute_status(
        self,
        mem: OwnershipMemory,
        persons_present: dict[int, "..."],  # noqa: F821
        now: float,
    ) -> ItemStatus:
        if mem.confirmed_owner_id is None:
            return ItemStatus(
                object_track_id=mem.object_track_id,
                object_class=mem.object_class,
                owner_track_id=None,
                confidence=mem.confidence,
                state=ItemState.UNCLAIMED,
                seconds_owner_away=0.0,
                seconds_owner_far=0.0,
                countdown_seconds=self.abandoned_after_seconds,
                last_known_bbox=mem.last_object_bbox,
                owner_present_in_frame=False,
            )

        seconds_away = max(0.0, now - mem.last_owner_present)
        seconds_far = max(0.0, now - mem.last_owner_close)
        owner_in_frame = mem.confirmed_owner_id in persons_present

        if seconds_far <= self.presence_grace_seconds:
            state = ItemState.PRESENT
            countdown = self.abandoned_after_seconds
        elif seconds_far >= self.abandoned_after_seconds:
            state = ItemState.ABANDONED
            countdown = 0.0
        else:
            state = ItemState.AWAY
            countdown = max(0.0, self.abandoned_after_seconds - seconds_far)

        prev = self._prev_state.get(mem.object_track_id)
        # Special: if previously ABANDONED and owner is back close, mark RECOVERED
        # for one update (transient, will fall back to PRESENT next frame).
        if prev == ItemState.ABANDONED and state == ItemState.PRESENT:
            state = ItemState.RECOVERED

        return ItemStatus(
            object_track_id=mem.object_track_id,
            object_class=mem.object_class,
            owner_track_id=mem.confirmed_owner_id,
            confidence=mem.confidence,
            state=state,
            seconds_owner_away=seconds_away,
            seconds_owner_far=seconds_far,
            countdown_seconds=countdown,
            last_known_bbox=mem.last_object_bbox,
            owner_present_in_frame=owner_in_frame,
        )

    def _maybe_emit_alert(
        self,
        prev: ItemState | None,
        status: ItemStatus,
        now: float,
    ) -> AbandonmentAlert | None:
        """Emit alerts on meaningful transitions only."""
        prev_v = prev.value if prev else "init"
        transition = f"{prev_v}→{status.state.value}"

        # AWAY → ABANDONED : new abandonment
        if status.state == ItemState.ABANDONED:
            severity = "critical" if status.seconds_owner_far >= self.critical_after_seconds else "warning"
            self._counter += 1
            owner_label = (
                f"Person #{status.owner_track_id}"
                if status.owner_track_id is not None
                else "an unknown owner"
            )
            return AbandonmentAlert(
                alert_id=f"abandon_{self._counter}",
                transition=transition,
                severity=severity,
                object_track_id=status.object_track_id,
                object_class=status.object_class,
                owner_track_id=status.owner_track_id,
                seconds_owner_away=status.seconds_owner_away,
                timestamp=now,
                bbox=status.last_known_bbox,
                message=(
                    f"{owner_label}'s {status.object_class} has been "
                    f"unattended for {int(status.seconds_owner_far)}s"
                ),
            )

        # ABANDONED → RECOVERED : owner returned (positive event)
        if status.state == ItemState.RECOVERED:
            self._counter += 1
            owner_label = (
                f"Person #{status.owner_track_id}"
                if status.owner_track_id is not None
                else "Owner"
            )
            return AbandonmentAlert(
                alert_id=f"recover_{self._counter}",
                transition=transition,
                severity="info",
                object_track_id=status.object_track_id,
                object_class=status.object_class,
                owner_track_id=status.owner_track_id,
                seconds_owner_away=status.seconds_owner_away,
                timestamp=now,
                bbox=status.last_known_bbox,
                message=(
                    f"{owner_label} returned to their {status.object_class}"
                ),
            )

        return None
