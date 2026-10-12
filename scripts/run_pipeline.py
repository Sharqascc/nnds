#!/usr/bin/env python3
"""
Main entry point for the NNDS Intersection Safety Pipeline.
Automatically ensures required models are downloaded and OpenVINO-exported.

Usage:
    python scripts/run_pipeline.py --video path/to/video.mp4 [options]
    python scripts/run_pipeline.py --video path/to/video.mp4 --skip-ensure

Or, after `pip install -e .`, as a console script:

    nnds-run --video path/to/video.mp4 [options]

The pipeline is dispatched via runpy into src/pipeline/traffic_analyzer.py so
that argparse help and error messages use the analyzer's own usage block.
"""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def ensure_models_if_needed() -> None:
    """Run scripts/ensure_models.py unless --skip-ensure or --help is present.

    Reads sys.argv directly because the final dispatch to the pipeline uses
    runpy and does not go through argparse in this module.
    """
    if "--skip-ensure" in sys.argv:
        sys.argv.remove("--skip-ensure")
        return

    if "--help" in sys.argv or "-h" in sys.argv:
        return

    ensure_script = REPO_ROOT / "scripts" / "ensure_models.py"
    if not ensure_script.exists():
        return

    print("\n[run_pipeline] Checking models...")
    result = subprocess.run([sys.executable, str(ensure_script), "--imgsz", "640"])
    if result.returncode != 0:
        print("Model preparation failed. Exiting.")
        sys.exit(result.returncode)


def _maybe_estimate_time_of_day() -> None:
    """If --auto-time-of-day is set and no explicit label, call the VLM helper.

    Mutates sys.argv so the downstream pipeline receives --time-of-day LABEL.
    """
    if "--auto-time-of-day" not in sys.argv or "--time-of-day" in sys.argv:
        return

    video_arg_idx = None
    for i, arg in enumerate(sys.argv):
        if arg == "--video" and i + 1 < len(sys.argv):
            video_arg_idx = i + 1
            break

    if video_arg_idx is None:
        print("[run_pipeline] --video not found; cannot auto-estimate time-of-day.")
        return

    video_path = sys.argv[video_arg_idx]
    print("\n[run_pipeline] Estimating time-of-day via VLM...")
    result = subprocess.run(
        [sys.executable, "scripts/estimate_time_of_day.py", "--video", video_path],
        capture_output=True,
        text=True,
    )
    label = result.stdout.strip() or "unknown"
    print(f"[run_pipeline] Estimated time-of-day: {label}")
    sys.argv.extend(["--time-of-day", label])


def main() -> None:
    """Prepare models, apply seeding, and dispatch to the pipeline module.

    Invoked by the `nnds-run` console script and by `python scripts/run_pipeline.py`.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from src.utils.seed import set_seed

    _seed = int(os.environ.get("NNDS_SEED", "42"))
    set_seed(_seed)

    ensure_models_if_needed()
    _maybe_estimate_time_of_day()

    pipeline_path = REPO_ROOT / "src" / "pipeline" / "traffic_analyzer.py"
    if not pipeline_path.exists():
        print(f"ERROR: Pipeline not found at {pipeline_path}")
        sys.exit(1)

    sys.argv[0] = str(pipeline_path)
    runpy.run_path(str(pipeline_path), run_name="__main__")


if __name__ == "__main__":
    main()
