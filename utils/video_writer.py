"""
Video Writer — saves annotated frames to an output video file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoWriter:
    """OpenCV-based video writer."""

    def __init__(self, cfg: dict):
        ocfg = cfg.get("output", {})
        self._path: str = ocfg.get("video_path", "output/result.mp4")
        self._fps: int = ocfg.get("video_fps", 30)
        self._codec: str = ocfg.get("codec", "mp4v")
        self._writer: cv2.VideoWriter | None = None

    def open(self, width: int, height: int) -> None:
        p = Path(self._path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self._codec)
        self._writer = cv2.VideoWriter(str(p), fourcc, self._fps, (width, height))
        logger.info("VideoWriter opened: %s (%dx%d @ %d fps)",
                     self._path, width, height, self._fps)

    def write(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.write(frame)

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info("VideoWriter released: %s", self._path)
