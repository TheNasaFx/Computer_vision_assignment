"""
Visualizer — draws boxes, labels, tracks, FPS overlay on frames.
Also provides study-space overlays: ownership lines, zone fills, alerts.
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

from core.detector import FrameResult, Detection
from core.tracker import Track


# 20 distinct colors for track IDs
_PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


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

    def _draw_detection(self, frame: np.ndarray, det: Detection) -> None:
        x1, y1, x2, y2 = map(int, det.bbox)
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
        """Draw full study-space overlay: ownership lines, zones, alerts, HUD."""
        out = frame.copy()

        # 1. Draw desk zones
        self._draw_zones(out, ss_result.zone_status.zones)

        # 2. Draw ownership lines (person ↔ object)
        self._draw_ownership(out, ss_result.inventories)

        # 3. Draw all detections
        for det in ss_result.frame_result.detections:
            self._draw_detection(out, det)

        # 4. Highlight unattended items
        self._draw_unattended(out, ss_result.proximity.unowned_objects)

        # 5. Draw alert indicators
        self._draw_alerts(out, ss_result.all_alerts)

        # 6. Study-space HUD
        self._draw_study_hud(out, ss_result)

        return out

    def _draw_zones(self, frame: np.ndarray, zones) -> None:
        """Draw desk zone polygons with occupancy color."""
        overlay = frame.copy()
        for zone in zones:
            pts = np.array(zone.points, dtype=np.int32)
            if len(pts) < 3:
                continue
            color = (0, 0, 200) if zone.occupied else (0, 180, 0)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(frame, [pts], True, color, 2)
            # Zone label
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

    def _draw_ownership(self, frame: np.ndarray, inventories) -> None:
        """Draw dashed colored lines from person to their objects."""
        for inv in inventories:
            if not inv.items:
                continue
            pid = inv.track_id if inv.track_id is not None else 0
            color = self._color_for(pid)
            pcx = int((inv.person.bbox[0] + inv.person.bbox[2]) / 2)
            pcy = int((inv.person.bbox[1] + inv.person.bbox[3]) / 2)

            for link in inv.items:
                ocx = int((link.object_det.bbox[0] + link.object_det.bbox[2]) / 2)
                ocy = int((link.object_det.bbox[1] + link.object_det.bbox[3]) / 2)
                self._draw_dashed_line(frame, (pcx, pcy), (ocx, ocy), color, 2, 10)

            # Person label with item count
            label = f"Person #{pid}: {len(inv.items)} items"
            x1, y1 = int(inv.person.bbox[0]), int(inv.person.bbox[1])
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 24), (x1 + tw + 8, y1 - 12), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_unattended(self, frame: np.ndarray, unowned: list[Detection]) -> None:
        """Draw pulsing red border around unattended items."""
        pulse = abs(math.sin(time.time() * 3)) * 0.5 + 0.5
        thickness = int(2 + 3 * pulse)
        for obj in unowned:
            x1, y1, x2, y2 = map(int, obj.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness)
            label = f"UNATTENDED {obj.class_name}"
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2, cv2.LINE_AA)

    def _draw_alerts(self, frame: np.ndarray, alerts) -> None:
        """Draw alert banner at top of frame if there are active alerts."""
        if not alerts:
            return
        recent = alerts[-3:]  # Show last 3
        h, w = frame.shape[:2]
        overlay = frame.copy()
        banner_h = 30 * len(recent) + 10
        cv2.rectangle(overlay, (w - 420, 8), (w - 8, 8 + banner_h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        for i, a in enumerate(recent):
            icon = "!!" if a.severity == "critical" else "!"
            text = f"{icon} {a.message}"
            cv2.putText(frame, text, (w - 410, 32 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_study_hud(self, frame: np.ndarray, ss_result) -> None:
        """Draw study-space specific HUD (bottom-right)."""
        h, w = frame.shape[:2]
        zs = ss_result.zone_status
        lines = [
            f"Persons: {len(ss_result.proximity.persons)}",
            f"Owned items: {sum(len(inv.items) for inv in ss_result.inventories)}",
            f"Unattended: {len(ss_result.proximity.unowned_objects)}",
            f"Inference: {ss_result.frame_result.inference_ms:.1f}ms",
        ]
        if zs.total > 0:
            lines.insert(2, f"Seats: {zs.occupied}/{zs.total} occupied")

        box_w, box_h = 260, 12 + 26 * len(lines)
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
        """Draw a dashed line between two points."""
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
