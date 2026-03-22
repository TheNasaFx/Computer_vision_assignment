"""
Logging configuration.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(cfg: dict) -> logging.Logger:
    """Configure root logger from config dict."""
    lcfg = cfg.get("logging", {})
    level_str = lcfg.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    log_file = lcfg.get("file", None)
    console = lcfg.get("console", True)

    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)-25s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Remove existing handlers
    root.handlers.clear()

    if console:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    if log_file:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(p), encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    return root
