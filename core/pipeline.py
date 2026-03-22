"""
Detection Pipeline — orchestrates stream → detector → tracker → events → output.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from core.detector import Detector, FrameResult
from core.tracker import ObjectTracker
from core.stream import FrameGrabber
from core.events import EventProcessor
from utils.visualizer import Visualizer
from utils.video_writer import VideoWriter

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """Main real-time detection pipeline."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        vcfg = cfg.get("visualization", {})
        ocfg = cfg.get("output", {})

        self.detector = Detector(cfg)
        self.tracker = ObjectTracker(cfg)
        self.event_processor = EventProcessor(cfg)
        self.visualizer = Visualizer(cfg)

        self._tracking_enabled: bool = cfg.get("tracker", {}).get("enabled", True)
        self._tracker_type: str = cfg.get("tracker", {}).get("type", "bytetrack")
        self._show_window: bool = vcfg.get("show_window", True)
        self._save_video: bool = ocfg.get("save_video", False)
        self._json_output: bool = ocfg.get("json_output", False)
        self._json_path: str = ocfg.get("json_path", "output/detections.json")

        self._grabbers: list[FrameGrabber] = []
        self._writers: list[VideoWriter | None] = []
        self._running = False

    # ── lifecycle ──────────────────────────────────────────────

    def setup(self, sources: list[str] | None = None) -> None:
        """Initialize detector and frame grabbers."""
        self.detector.load()

        if sources is None:
            sources = self.cfg.get("stream", {}).get("sources", ["0"])

        scfg = self.cfg.get("stream", {})
        for src in sources:
            g = FrameGrabber(
                source=src,
                buffer_size=scfg.get("buffer_size", 2),
                reconnect_delay=scfg.get("reconnect_delay", 5.0),
                max_retries=scfg.get("max_retries", 10),
            )
            self._grabbers.append(g)
            self._writers.append(None)

        logger.info("Pipeline ready — %d stream(s)", len(self._grabbers))

    def run(self) -> None:
        """Start all streams and enter main loop."""
        for g in self._grabbers:
            if not g.start():
                logger.error("Failed to start grabber for %s", g.source)

        self._running = True
        fps_counter = _FPSCounter()
        json_records: list[dict] = []

        logger.info("Pipeline running — press 'q' to quit")

        try:
            while self._running:
                for idx, grabber in enumerate(self._grabbers):
                    ok, frame, fid = grabber.read()
                    if not ok or frame is None:
                        continue

                    # ── Inference ──
                    if self._tracking_enabled:
                        result = self.detector.detect_and_track(
                            frame, tracker_type=self._tracker_type)
                    else:
                        result = self.detector.detect(frame)
                    result.frame_id = fid
                    result.timestamp = time.time()

                    # ── Tracking state ──
                    if self._tracking_enabled:
                        self.tracker.update(result.detections)

                    # ── Events ──
                    events = self.event_processor.process(result.detections)

                    # ── Visualization ──
                    fps_counter.tick()
                    vis_frame = self.visualizer.draw(
                        frame, result, self.tracker.tracks, fps_counter.fps)

                    if self.event_processor.enabled:
                        vis_frame = self.event_processor.draw_zones(vis_frame)

                    # ── Video writer ──
                    if self._save_video:
                        if self._writers[idx] is None:
                            self._writers[idx] = VideoWriter(self.cfg)
                            h, w = frame.shape[:2]
                            self._writers[idx].open(w, h)
                        self._writers[idx].write(vis_frame)

                    # ── JSON ──
                    if self._json_output:
                        json_records.append(result.to_dict())

                    # ── Display ──
                    if self._show_window:
                        title = f"Stream {idx}" if len(self._grabbers) > 1 else "YOLO Detection"
                        cv2.imshow(title, vis_frame)

                # Handle key events
                if self._show_window:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("Quit requested")
                        break

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.shutdown()
            if self._json_output and json_records:
                self._save_json(json_records)

    def shutdown(self) -> None:
        """Clean up all resources."""
        self._running = False
        for g in self._grabbers:
            g.stop()
        for w in self._writers:
            if w is not None:
                w.release()
        if self._show_window:
            cv2.destroyAllWindows()
        logger.info("Pipeline shut down")

    # ── single-frame inference (for API) ───────────────────────

    def infer_image(self, frame: np.ndarray) -> FrameResult:
        """Run detection on a single image (no tracking)."""
        return self.detector.detect(frame)

    # ── helpers ────────────────────────────────────────────────

    def _save_json(self, records: list[dict]) -> None:
        p = Path(self._json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        logger.info("JSON output saved: %s (%d frames)", p, len(records))


class _FPSCounter:
    """Exponential moving average FPS counter."""

    def __init__(self, alpha: float = 0.1):
        self._alpha = alpha
        self._fps: float = 0.0
        self._prev = time.perf_counter()

    def tick(self) -> None:
        now = time.perf_counter()
        dt = now - self._prev
        if dt > 0:
            instant = 1.0 / dt
            self._fps = self._alpha * instant + (1 - self._alpha) * self._fps
        self._prev = now

    @property
    def fps(self) -> float:
        return self._fps
