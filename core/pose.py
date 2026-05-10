"""
Pose Estimator — wraps a YOLO pose model and extracts wrist keypoints
for use as ownership cues.

The pose model runs on the full frame, detects every person, and produces
17 COCO keypoints per person. We match each pose to a tracked person from
the main detector (by bbox-IoU, since both saw the same frame) so that
the wrists can be referenced by the same `track_id` the OwnershipTracker
uses.

Wrist proximity is a far stronger ownership signal than the torso centre:
"hand on the laptop" is what real ownership looks like.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.detector import Detection

logger = logging.getLogger(__name__)


# COCO keypoint indices we care about
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6

# Full COCO keypoint names in the canonical order returned by the model
COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


@dataclass
class PoseResult:
    """Per-frame pose output, keyed by person track_id."""
    wrists: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    keypoints: dict[int, np.ndarray] = field(default_factory=dict)  # tid -> (17, 3)
    inference_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "num_persons_with_pose": len(self.wrists),
            "inference_ms": round(self.inference_ms, 2),
        }


class PoseEstimator:
    """Lightweight wrapper around a YOLO pose model."""

    def __init__(self, cfg: dict):
        pcfg = cfg.get("pose", {})
        self.enabled: bool = pcfg.get("enabled", True)
        self.model_name: str = pcfg.get("model", "yolo11l-pose.pt")
        self.confidence: float = pcfg.get("confidence", 0.4)
        self.kpt_confidence: float = pcfg.get("kpt_confidence", 0.35)
        self.img_size: int = pcfg.get("img_size", 960)
        self.iou_match_threshold: float = pcfg.get("iou_match_threshold", 0.3)

        # Device matches main detector
        self.device: str = cfg.get("model", {}).get("device", "0")
        self.half: bool = cfg.get("model", {}).get("half", True)

        self._model: Any | None = None
        self._available: bool = False

    def load(self) -> None:
        if not self.enabled:
            logger.info("Pose: disabled in config")
            return
        try:
            import torch
            if self.device != "cpu" and not torch.cuda.is_available():
                logger.warning("Pose: CUDA unavailable, falling back to CPU")
                self.device = "cpu"
                self.half = False

            from ultralytics import YOLO
            self._model = YOLO(self.model_name)
            # warm-up
            dummy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            self._model.predict(
                dummy,
                device=self.device,
                half=self.half,
                imgsz=self.img_size,
                verbose=False,
            )
            self._available = True
            logger.info("Pose: loaded %s (device=%s)", self.model_name, self.device)
        except Exception as e:
            logger.warning("Pose: failed to load (%s) — running without pose cues", e)
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def estimate_raw(self, frame: np.ndarray) -> dict:
        """Run pose on `frame` and return raw, identity-free keypoints.

        Returned shape (suited for the /pose-frame endpoint):
            {
              "inference_ms": float,
              "frame_width": int,
              "frame_height": int,
              "persons": [
                  {
                    "bbox": [x1, y1, x2, y2],
                    "keypoints": {
                        "<name>": {"x": float, "y": float, "conf": float}, ...
                    }
                  }, ...
              ]
            }
        """
        h, w = frame.shape[:2]
        out_payload: dict = {
            "inference_ms": 0.0,
            "frame_width": int(w),
            "frame_height": int(h),
            "persons": [],
        }

        if not self._available or self._model is None:
            return out_payload

        t0 = time.perf_counter()
        try:
            yolo_out = self._model.predict(
                frame,
                conf=self.confidence,
                device=self.device,
                half=self.half,
                imgsz=self.img_size,
                verbose=False,
            )
        except Exception as e:
            logger.debug("Pose inference error: %s", e)
            return out_payload
        out_payload["inference_ms"] = round(
            (time.perf_counter() - t0) * 1000.0, 2
        )

        if not yolo_out:
            return out_payload
        result = yolo_out[0]
        if result.keypoints is None or result.boxes is None:
            return out_payload

        try:
            kpts_xy = result.keypoints.xy.cpu().numpy()       # (N, 17, 2)
            kpts_conf = (
                result.keypoints.conf.cpu().numpy()
                if result.keypoints.conf is not None
                else None
            )
            boxes = result.boxes.xyxy.cpu().numpy()           # (N, 4)
        except Exception as e:
            logger.debug("Pose tensor extraction failed: %s", e)
            return out_payload

        names = COCO_KEYPOINT_NAMES
        for i in range(len(boxes)):
            kp_dict: dict = {}
            for k_idx, name in enumerate(names):
                x, y = kpts_xy[i][k_idx]
                c = (
                    float(kpts_conf[i][k_idx])
                    if kpts_conf is not None else 1.0
                )
                kp_dict[name] = {
                    "x": round(float(x), 1),
                    "y": round(float(y), 1),
                    "conf": round(c, 3),
                }
            out_payload["persons"].append({
                "bbox": [round(float(v), 1) for v in boxes[i].tolist()],
                "keypoints": kp_dict,
            })

        return out_payload

    def estimate(
        self, frame: np.ndarray, persons: list[Detection]
    ) -> PoseResult:
        """Run pose on `frame` and align results to the tracked persons.

        Returns a PoseResult whose `wrists[track_id]` gives a list of (x, y)
        wrist positions for that person (one or two entries, depending on
        which wrists were visible above kpt_confidence).
        """
        result = PoseResult()
        if not self._available or self._model is None or not persons:
            return result

        t0 = time.perf_counter()
        try:
            yolo_out = self._model.predict(
                frame,
                conf=self.confidence,
                device=self.device,
                half=self.half,
                imgsz=self.img_size,
                verbose=False,
            )
        except Exception as e:
            logger.debug("Pose inference error: %s", e)
            return result
        result.inference_ms = (time.perf_counter() - t0) * 1000.0

        if not yolo_out:
            return result
        out = yolo_out[0]
        if out.keypoints is None or out.boxes is None:
            return result

        try:
            kpts_xy = out.keypoints.xy.cpu().numpy()       # (N, 17, 2)
            kpts_conf = out.keypoints.conf.cpu().numpy() if out.keypoints.conf is not None else None  # (N, 17)
            pose_boxes = out.boxes.xyxy.cpu().numpy()       # (N, 4)
        except Exception as e:
            logger.debug("Pose tensor extraction failed: %s", e)
            return result

        if len(pose_boxes) == 0:
            return result

        # Match each pose detection to one tracked person by max-IoU
        for pose_idx, pbox in enumerate(pose_boxes):
            best_tid: int | None = None
            best_iou = self.iou_match_threshold
            for person in persons:
                if person.track_id is None:
                    continue
                iou = _bbox_iou(pbox.tolist(), person.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = person.track_id
            if best_tid is None:
                continue

            kpts = kpts_xy[pose_idx]               # (17, 2)
            confs = kpts_conf[pose_idx] if kpts_conf is not None else None

            # Combine xy + conf into (17, 3) for downstream
            if confs is not None:
                full = np.concatenate([kpts, confs[:, None]], axis=1)
            else:
                full = np.concatenate(
                    [kpts, np.ones((kpts.shape[0], 1))], axis=1
                )
            result.keypoints[best_tid] = full

            wrist_pts: list[tuple[float, float]] = []
            for wi in (KP_LEFT_WRIST, KP_RIGHT_WRIST):
                x, y = kpts[wi]
                c = confs[wi] if confs is not None else 1.0
                if c >= self.kpt_confidence and (x > 0 or y > 0):
                    wrist_pts.append((float(x), float(y)))
            if wrist_pts:
                result.wrists[best_tid] = wrist_pts

        return result
