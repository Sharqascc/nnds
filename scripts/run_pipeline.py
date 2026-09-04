#!/usr/bin/env python3
"""
Main entry point for the NNDS Intersection Safety Pipeline.
Automatically ensures required models are downloaded and OpenVINO-exported.

Usage:
    python scripts/run_pipeline.py --video path/to/video.mp4 [options]
    python scripts/run_pipeline.py --video path/to/video.mp4 --skip-ensure
"""

import runpy
import subprocess
import sys
from pathlib import Path


def ensure_models_if_needed():
    """Run ensure_models.py unless --skip-ensure or --help is present."""
    if "--skip-ensure" in sys.argv:
        sys.argv.remove("--skip-ensure")
        return

    if "--help" in sys.argv or "-h" in sys.argv:
        return

    root_dir = Path(__file__).parent.parent
    ensure_script = root_dir / "scripts" / "ensure_models.py"

    if not ensure_script.exists():
        return

    print("\n[run_pipeline] Checking models...")
    result = subprocess.run([sys.executable, str(ensure_script), "--imgsz", "640"])
    if result.returncode != 0:
        print("❌ Model preparation failed. Exiting.")
        sys.exit(result.returncode)


if __name__ == "__main__":
    # Add project root to Python path so imports like 'src.analysis' work
    root_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(root_dir))
    from src.utils.seed import set_seed

    set_seed()
    set_seed()

    ensure_models_if_needed()

    # Auto time-of-day using VLM if requested and no explicit label
    if "--auto-time-of-day" in sys.argv and "--time-of-day" not in sys.argv:
        video_arg_idx = None
        for i, arg in enumerate(sys.argv):
            if arg == "--video" and i + 1 < len(sys.argv):
                video_arg_idx = i + 1
                break
        if video_arg_idx:
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
        else:
            print("[run_pipeline] --video not found; cannot auto-estimate time-of-day.")

    # Run the actual pipeline script
    pipeline_path = root_dir / "src" / "pipeline" / "traffic_analyzer.py"
    if not pipeline_path.exists():
        print(f"ERROR: Pipeline not found at {pipeline_path}")
        sys.exit(1)

    # Pass command-line arguments through
    sys.argv[0] = str(pipeline_path)  # Makes help messages look correct
    runpy.run_path(str(pipeline_path), run_name="__main__")
