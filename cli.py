"""
CLI interface for the Real-Time Object Detection System.

Usage:
    python cli.py run --source 0
    python cli.py run --source video.mp4 --save-video
    python cli.py serve --port 8000
    python cli.py detect --image photo.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def load_config(path: str | None = None) -> dict:
    """Load YAML config, fallback to defaults."""
    cfg_path = Path(path) if path else Path(__file__).parent / "config" / "default.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def cmd_run(args, cfg: dict) -> None:
    """Run real-time detection pipeline."""
    from utils.logger import setup_logger
    from core.pipeline import DetectionPipeline

    # Override config with CLI args
    if args.source:
        cfg.setdefault("stream", {})["sources"] = args.source
    if args.model:
        cfg.setdefault("model", {})["name"] = args.model
    if args.confidence:
        cfg.setdefault("model", {})["confidence"] = args.confidence
    if args.device:
        cfg.setdefault("model", {})["device"] = args.device
    if args.save_video:
        cfg.setdefault("output", {})["save_video"] = True
    if args.no_display:
        cfg.setdefault("visualization", {})["show_window"] = False
    if args.no_track:
        cfg.setdefault("tracker", {})["enabled"] = False

    setup_logger(cfg)
    pipeline = DetectionPipeline(cfg)
    pipeline.setup()
    pipeline.run()


def cmd_serve(args, cfg: dict) -> None:
    """Start FastAPI server."""
    from utils.logger import setup_logger
    setup_logger(cfg)

    if args.port:
        cfg.setdefault("server", {})["port"] = args.port
    if args.host:
        cfg.setdefault("server", {})["host"] = args.host
    if args.model:
        cfg.setdefault("model", {})["name"] = args.model

    import uvicorn
    from api.server import create_app

    app = create_app(cfg)
    scfg = cfg.get("server", {})
    uvicorn.run(app,
                host=scfg.get("host", "0.0.0.0"),
                port=scfg.get("port", 8000),
                log_level="info")


def cmd_detect(args, cfg: dict) -> None:
    """Single-image detection."""
    from utils.logger import setup_logger
    from core.detector import Detector

    if args.model:
        cfg.setdefault("model", {})["name"] = args.model
    if args.confidence:
        cfg.setdefault("model", {})["confidence"] = args.confidence
    if args.device:
        cfg.setdefault("model", {})["device"] = args.device

    setup_logger(cfg)

    import cv2
    detector = Detector(cfg)
    detector.load()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: cannot read image '{args.image}'", file=sys.stderr)
        sys.exit(1)

    result = detector.detect(frame)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Inference: {result.inference_ms:.1f} ms")
        print(f"Detections: {len(result.detections)}")
        for d in result.detections:
            print(f"  {d.class_name}: {d.confidence:.1%}  bbox={[round(v) for v in d.bbox]}")

    if args.output:
        from utils.visualizer import Visualizer
        vis = Visualizer(cfg)
        out = vis.draw(frame, result)
        cv2.imwrite(args.output, out)
        print(f"Saved: {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="yolo-detect",
        description="Real-Time Object Detection System (YOLO-based)",
    )
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to YAML config file")

    sub = parser.add_subparsers(dest="command")

    # ── run ──
    p_run = sub.add_parser("run", help="Run real-time detection")
    p_run.add_argument("--source", "-s", nargs="+", default=None,
                       help="Input source(s): webcam index, video path, RTSP URL")
    p_run.add_argument("--model", "-m", type=str, default=None)
    p_run.add_argument("--confidence", type=float, default=None)
    p_run.add_argument("--device", type=str, default=None)
    p_run.add_argument("--save-video", action="store_true")
    p_run.add_argument("--no-display", action="store_true")
    p_run.add_argument("--no-track", action="store_true")

    # ── serve ──
    p_serve = sub.add_parser("serve", help="Start API server")
    p_serve.add_argument("--host", type=str, default=None)
    p_serve.add_argument("--port", "-p", type=int, default=None)
    p_serve.add_argument("--model", "-m", type=str, default=None)

    # ── detect ──
    p_detect = sub.add_parser("detect", help="Single image detection")
    p_detect.add_argument("--image", "-i", type=str, required=True)
    p_detect.add_argument("--model", "-m", type=str, default=None)
    p_detect.add_argument("--confidence", type=float, default=None)
    p_detect.add_argument("--device", type=str, default=None)
    p_detect.add_argument("--output", "-o", type=str, default=None,
                          help="Save annotated image")
    p_detect.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    cfg = load_config(args.config)

    {"run": cmd_run, "serve": cmd_serve, "detect": cmd_detect}[args.command](args, cfg)


if __name__ == "__main__":
    main()
