"""
Quick-start entry point.

    python main.py                    # webcam with defaults
    python main.py --source video.mp4 # video file
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logger
from core.pipeline import DetectionPipeline


def load_config(path: str | None = None) -> dict:
    cfg_path = Path(path) if path else Path(__file__).parent / "config" / "default.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="YOLO Real-Time Detection")
    parser.add_argument("--source", "-s", nargs="+", default=None)
    parser.add_argument("--model", "-m", type=str, default=None)
    parser.add_argument("--config", "-c", type=str, default=None)
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.source:
        cfg.setdefault("stream", {})["sources"] = args.source
    if args.model:
        cfg.setdefault("model", {})["name"] = args.model
    if args.save_video:
        cfg.setdefault("output", {})["save_video"] = True

    setup_logger(cfg)

    pipeline = DetectionPipeline(cfg)
    pipeline.setup()
    pipeline.run()


if __name__ == "__main__":
    main()
