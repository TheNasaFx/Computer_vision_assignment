"""
Visualizer — draws boxes, labels, tracks, FPS overlay on frames.
"""

from __future__ import annotations

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
