"""
Async Frame Grabber — threaded capture for webcam / video / RTSP.
Non-blocking I/O with configurable buffer.
"""

from __future__ import annotations

import logging
import time
import threading
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameGrabber:
    """Threaded frame grabber for a single video source."""

    def __init__(self, source: str | int, buffer_size: int = 2,
                 reconnect_delay: float = 5.0, max_retries: int = 10):
        self._source_raw = source
        self.source: int | str = int(source) if str(source).isdigit() else source
        self.buffer_size = buffer_size
        self.reconnect_delay = reconnect_delay
        self.max_retries = max_retries

        self._cap: cv2.VideoCapture | None = None
        self._buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._frame_id = 0
        self._fps: float = 0.0

    # ── public API ─────────────────────────────────────────────

    def start(self) -> bool:
        """Open capture and start background thread."""
        if not self._open():
            return False
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("FrameGrabber started: %s", self.source)
        return True

    def read(self) -> tuple[bool, np.ndarray | None, int]:
        """Get latest frame (non-blocking). Returns (ok, frame, frame_id)."""
        with self._lock:
            if len(self._buffer) == 0:
                return False, None, self._frame_id
            frame = self._buffer[-1]  # always latest
            return True, frame, self._frame_id

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._release()
        logger.info("FrameGrabber stopped: %s", self.source)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def resolution(self) -> tuple[int, int]:
        if self._cap and self._cap.isOpened():
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        return 0, 0

    # ── internals ──────────────────────────────────────────────

    def _open(self) -> bool:
        self._release()
        for attempt in range(1, self.max_retries + 1):
            self._cap = cv2.VideoCapture(self.source)
            if self._cap.isOpened():
                # Optimize capture
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                logger.info("Capture opened: %s (%dx%d)",
                            self.source, *self.resolution)
                return True
            logger.warning("Open attempt %d/%d failed for %s",
                           attempt, self.max_retries, self.source)
            time.sleep(self.reconnect_delay)
        logger.error("Failed to open source: %s", self.source)
        return False

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _capture_loop(self) -> None:
        t_prev = time.perf_counter()
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                if not self._open():
                    break
                continue

            ret, frame = self._cap.read()
            if not ret:
                # End of file or stream drop
                if isinstance(self.source, int):
                    logger.warning("Frame drop on webcam — reconnecting")
                    time.sleep(0.1)
                    continue
                else:
                    # Video file ended
                    logger.info("End of video file: %s", self.source)
                    self._running = False
                    break

            with self._lock:
                self._buffer.append(frame)
                self._frame_id += 1

            t_now = time.perf_counter()
            dt = t_now - t_prev
            if dt > 0:
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
            t_prev = t_now

        self._release()
