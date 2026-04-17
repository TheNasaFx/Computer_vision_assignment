"""
Alert Manager — generates alerts for unattended items and other anomalies.

Tracks objects that have no nearby person for a configurable duration and
raises alerts so the frontend / pipeline can notify the user.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from core.detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """A single alert event."""
    alert_id: str
    alert_type: str          # "unattended_item" | "zone_overcrowded"
    severity: str            # "info" | "warning" | "critical"
    message: str
    timestamp: float
    bbox: list[float] | None = None
    class_name: str | None = None

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "timestamp": round(self.timestamp, 2),
            "bbox": self.bbox,
            "class_name": self.class_name,
        }


def _bbox_key(det: Detection) -> str:
    """Rough spatial key so we can track the 'same' static object across frames."""
    cx = int((det.bbox[0] + det.bbox[2]) / 2) // 40
    cy = int((det.bbox[1] + det.bbox[3]) / 2) // 40
    return f"{det.class_name}_{cx}_{cy}"


class AlertManager:
    """Monitors unowned objects and raises alerts after a timeout."""

    def __init__(self, cfg: dict):
        scfg = cfg.get("study_space", {}).get("alerts", {})
        self.unattended_timeout: float = scfg.get("unattended_timeout", 30.0)
        self.cooldown: float = scfg.get("cooldown", 60.0)

        # bbox_key → first-seen timestamp
        self._first_seen: dict[str, float] = {}
        # bbox_key → last alert timestamp (for cooldown)
        self._last_alert: dict[str, float] = {}
        # currently active alerts
        self._active: list[Alert] = []
        self._counter: int = 0

    def update(self, unowned_objects: list[Detection]) -> list[Alert]:
        """Check unowned objects and return new alerts."""
        now = time.time()
        new_alerts: list[Alert] = []
        current_keys: set[str] = set()

        for obj in unowned_objects:
            key = _bbox_key(obj)
            current_keys.add(key)

            if key not in self._first_seen:
                self._first_seen[key] = now

            elapsed = now - self._first_seen[key]
            if elapsed >= self.unattended_timeout:
                last = self._last_alert.get(key, 0.0)
                if now - last >= self.cooldown:
                    self._counter += 1
                    severity = "critical" if elapsed > self.unattended_timeout * 3 else "warning"
                    alert = Alert(
                        alert_id=f"alert_{self._counter}",
                        alert_type="unattended_item",
                        severity=severity,
                        message=f"Unattended {obj.class_name} detected for {int(elapsed)}s",
                        timestamp=now,
                        bbox=obj.bbox,
                        class_name=obj.class_name,
                    )
                    new_alerts.append(alert)
                    self._last_alert[key] = now
                    logger.info("Alert: %s", alert.message)

        # Prune keys no longer visible
        stale = [k for k in self._first_seen if k not in current_keys]
        for k in stale:
            del self._first_seen[k]
            self._last_alert.pop(k, None)

        # Maintain active alerts list (keep last 20)
        self._active.extend(new_alerts)
        self._active = self._active[-20:]

        return new_alerts

    @property
    def active_alerts(self) -> list[Alert]:
        return list(self._active)

    def get_unattended_durations(self) -> dict[str, float]:
        """Return how long each currently-unattended object has been seen."""
        now = time.time()
        return {k: round(now - v, 1) for k, v in self._first_seen.items()}

    def reset(self) -> None:
        self._first_seen.clear()
        self._last_alert.clear()
        self._active.clear()
