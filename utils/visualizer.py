"""
Visualizer — draws boxes, labels, tracks, FPS overlay on frames.
Also provides study-space overlays: per-object state, ownership lines,
zone fills, abandonment alerts.
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

from core.detector import FrameResult, Detection
from core.tracker import Track
from core.abandonment import ItemState, ItemStatus


# 20 distinct colors for track IDs
_PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]

# Per-state colour codes (BGR) used when drawing study-space overlay
_STATE_COLOR = {
    ItemState.PRESENT: (80, 200, 80),       # green
    ItemState.AWAY: (0, 200, 240),          # amber
    ItemState.ABANDONED: (40, 40, 230),     # red
    ItemState.RECOVERED: (220, 200, 60),    # cyan-ish
    ItemState.UNCLAIMED: (170, 170, 170),   # grey
}


class Visualizer:
    """Draws detections, tracks, and HUD info on frames."""

    def __init__(self, cfg: dict):
        vcfg = cfg.get("visualization", {})
        self.show_fps = vcfg.get("show_fps", True)
        self.show_boxes = vcfg.get("show_boxes", True)
        self.show_labels = vcfg.get("show_labels", True)
        self.show_confidence = vcfg.get("show_confidence", True)
        self.show_tracks = vcfg.get("show_tracks", True)
        self.box_thickness = vcfg.get("box_thickness", 2)
        self.font_scale = vcfg.get("font_scale", 0.6)

    def draw(self, frame: np.ndarray, result: FrameResult,
             tracks: dict[int, Track] | None = None,
             fps: float = 0.0, *,
             playback_speed: float = 1.0,
             paused: bool = False) -> np.ndarray:
        """Render all overlays and return annotated frame."""
        out = frame.copy()

        # Draw tracks (trails)
        if self.show_tracks and tracks:
            self._draw_trails(out, tracks)

        # Draw detections
        if self.show_boxes:
            for det in result.detections:
                self._draw_detection(out, det)

        # HUD
        if self.show_fps:
            self._draw_hud(out, fps, result.inference_ms, len(result.detections),
                           playback_speed=playback_speed, paused=paused)

        return out

    def _draw_detection(self, frame: np.ndarray, det: Detection,
                        color: tuple | None = None) -> None:
        x1, y1, x2, y2 = map(int, det.bbox)
        if color is None:
            color = self._color_for(det.track_id or det.class_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)

        if self.show_labels or self.show_confidence:
            parts = []
            if det.track_id is not None:
                parts.append(f"#{det.track_id}")
            if self.show_labels:
                parts.append(det.class_name)
            if self.show_confidence:
                parts.append(f"{det.confidence:.0%}")
            label = " ".join(parts)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                           self.font_scale, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                        (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_trails(self, frame: np.ndarray, tracks: dict[int, Track]) -> None:
        for tid, track in tracks.items():
            if len(track.history) < 2:
                continue
            color = self._color_for(tid)
            pts = list(track.history)
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                thickness = max(1, int(3 * alpha))
                cv2.line(frame, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray, fps: float,
                  inf_ms: float, n_det: int, *,
                  playback_speed: float = 1.0,
                  paused: bool = False) -> None:
        h, w = frame.shape[:2]

        speed_label = f"Speed: {playback_speed:.2g}x"
        if paused:
            speed_label = "PAUSED"

        lines = [
            f"FPS: {fps:.1f}",
            f"Inference: {inf_ms:.1f} ms",
            f"Detections: {n_det}",
            speed_label,
        ]
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (250, 12 + 28 * len(lines)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, line in enumerate(lines):
            color = (0, 255, 255) if i == 3 and paused else (0, 255, 0)
            cv2.putText(frame, line, (14, 32 + 28 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        color, 2, cv2.LINE_AA)

        # Controls hint (bottom-left)
        hint = "+/-: speed | Space: pause | R: reset | Q: quit"
        cv2.putText(frame, hint, (14, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (180, 180, 180), 1, cv2.LINE_AA)

    @staticmethod
    def _color_for(idx: int) -> tuple[int, int, int]:
        return _PALETTE[idx % len(_PALETTE)]

    # ── Study-Space Drawing ────────────────────────────────────

    def draw_study_space(self, frame: np.ndarray, ss_result) -> np.ndarray:
        """Draw full study-space overlay using the new state-machine result."""
        out = frame.copy()

        # 1. Desk zones (bottom layer, semi-transparent)
        self._draw_zones(out, ss_result.zone_status.zones)

        # 2. Pose wrists (small dots — confirms pose is firing)
        self._draw_wrists(out, ss_result.pose_result.wrists)

        # 3. Persons — coloured by their track_id
        for pid, person in ss_result.ownership.persons_present.items():
            self._draw_detection(out, person, color=self._color_for(pid))

        # 4. Per-object state-aware boxes
        statuses = ss_result.abandonment.statuses
        for status in statuses.values():
            self._draw_item_status(out, status)

        # 5. Ownership lines (person ↔ confirmed object)
        self._draw_ownership_links(
            out,
            ss_result.ownership.memories,
            ss_result.ownership.persons_present,
        )

        # 6. Other (non-study, non-person) detections in muted colour
        owned_or_tracked = {
            d.track_id for d in ss_result.frame_result.detections
            if d.class_name == "person" and d.track_id is not None
        }
        item_tids = set(statuses.keys())
        for det in ss_result.frame_result.detections:
            if det.class_name == "person":
                continue
            if det.track_id is not None and det.track_id in item_tids:
                continue  # already drawn by _draw_item_status
            self._draw_detection(out, det, color=(140, 140, 140))

        # 7. Alert banner
        self._draw_alerts(out, ss_result.abandonment.active_alerts)

        # 8. Study-space HUD
        self._draw_study_hud(out, ss_result)

        return out

    def _draw_zones(self, frame: np.ndarray, zones) -> None:
        overlay = frame.copy()
        for zone in zones:
            pts = np.array(zone.points, dtype=np.int32)
            if len(pts) < 3:
                continue
            color = (0, 0, 200) if zone.occupied else (0, 180, 0)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(frame, [pts], True, color, 2)
            cx = int(np.mean([p[0] for p in zone.points]))
            cy = int(np.mean([p[1] for p in zone.points]))
            status = f"{zone.name}: OCCUPIED" if zone.occupied else f"{zone.name}: VACANT"
            (tw, th), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (cx - tw // 2 - 4, cy - th - 6),
                          (cx + tw // 2 + 4, cy + 4), (0, 0, 0), -1)
            cv2.putText(frame, status, (cx - tw // 2, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    def _draw_wrists(self, frame: np.ndarray,
                     wrists: dict[int, list[tuple[float, float]]]) -> None:
        for pid, pts in wrists.items():
            color = self._color_for(pid)
            for (x, y) in pts:
                cv2.circle(frame, (int(x), int(y)), 5, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (int(x), int(y)), 7, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_item_status(self, frame: np.ndarray, status: ItemStatus) -> None:
        if not status.last_known_bbox:
            return
        x1, y1, x2, y2 = map(int, status.last_known_bbox)
        base_color = _STATE_COLOR.get(status.state, (170, 170, 170))

        # Pulsing for ABANDONED, otherwise steady
        if status.state == ItemState.ABANDONED:
            pulse = abs(math.sin(time.time() * 4.0)) * 0.5 + 0.5
            thickness = int(2 + 4 * pulse)
            color = base_color
        elif status.state == ItemState.AWAY:
            thickness = 3
            color = base_color
        elif status.state == ItemState.RECOVERED:
            thickness = 4
            color = base_color
        else:
            thickness = 2
            color = base_color

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Build status label
        owner_txt = (
            f"#{status.owner_track_id}"
            if status.owner_track_id is not None else "?"
        )
        if status.state == ItemState.PRESENT:
            label = f"{status.object_class}  owner {owner_txt}  OK"
        elif status.state == ItemState.AWAY:
            label = (
                f"{status.object_class}  owner {owner_txt} away  "
                f"abandon in {int(status.countdown_seconds)}s"
            )
        elif status.state == ItemState.ABANDONED:
            label = (
                f"!! ABANDONED {status.object_class}  "
                f"(owner {owner_txt}, {int(status.seconds_owner_far)}s)"
            )
        elif status.state == ItemState.RECOVERED:
            label = f"OK Recovered: {status.object_class} (owner {owner_txt})"
        else:
            label = f"{status.object_class}  (no owner yet)"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_ownership_links(self, frame: np.ndarray, memories,
                              persons_present) -> None:
        for mem in memories.values():
            if mem.confirmed_owner_id is None:
                continue
            owner = persons_present.get(mem.confirmed_owner_id)
            if owner is None or not mem.last_object_bbox:
                continue
            color = self._color_for(mem.confirmed_owner_id)
            ocx = int((mem.last_object_bbox[0] + mem.last_object_bbox[2]) / 2)
            ocy = int((mem.last_object_bbox[1] + mem.last_object_bbox[3]) / 2)
            pcx = int((owner.bbox[0] + owner.bbox[2]) / 2)
            pcy = int((owner.bbox[1] + owner.bbox[3]) / 2)
            self._draw_dashed_line(frame, (pcx, pcy), (ocx, ocy), color, 2, 10)

    def _draw_alerts(self, frame: np.ndarray, alerts) -> None:
        if not alerts:
            return
        recent = alerts[-3:]
        h, w = frame.shape[:2]
        overlay = frame.copy()
        banner_h = 30 * len(recent) + 10
        cv2.rectangle(overlay, (w - 480, 8), (w - 8, 8 + banner_h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        for i, a in enumerate(recent):
            icon = "!!" if a.severity == "critical" else ("OK" if a.severity == "info" else "!")
            text = f"{icon} {a.message}"
            cv2.putText(frame, text, (w - 470, 32 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_study_hud(self, frame: np.ndarray, ss_result) -> None:
        h, w = frame.shape[:2]
        zs = ss_result.zone_status
        ab = ss_result.abandonment
        lines = [
            f"Persons: {len(ss_result.ownership.persons_present)}",
            f"Owned items: {len(ss_result.confirmed_memories)}",
            f"Away: {len(ab.away)}",
            f"Abandoned: {len(ab.abandoned)}",
            f"Inference: {ss_result.frame_result.inference_ms:.0f}ms"
            f" + pose {ss_result.pose_result.inference_ms:.0f}ms",
        ]
        if zs.total > 0:
            lines.insert(2, f"Seats: {zs.occupied}/{zs.total}")

        box_w, box_h = 300, 12 + 26 * len(lines)
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - box_w - 12, h - box_h - 12),
                      (w - 8, h - 8), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (w - box_w - 4, h - box_h + 18 + 26 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_dashed_line(frame: np.ndarray, pt1: tuple, pt2: tuple,
                          color: tuple, thickness: int = 1,
                          gap: int = 10) -> None:
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        dist = max(1, int(math.hypot(dx, dy)))
        for i in range(0, dist, gap * 2):
            start_ratio = i / dist
            end_ratio = min((i + gap) / dist, 1.0)
            sx = int(pt1[0] + dx * start_ratio)
            sy = int(pt1[1] + dy * start_ratio)
            ex = int(pt1[0] + dx * end_ratio)
            ey = int(pt1[1] + dy * end_ratio)
            cv2.line(frame, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
