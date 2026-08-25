#!/usr/bin/env python3
"""
Unified validation entry point for the NNDS repository.

Runs in sequence:
  1. pytest (unless --skip-pytest)
  2. BEV homography validation
  3. Detection/tracking/PET output validation

Usage examples:
  python scripts/validate_all.py
  python scripts/validate_all.py --skip-pytest \
      --detections tests/fixtures/sample_detections.csv \
      --detections-split tests/fixtures/sample_split_detections.csv \
      --pet tests/fixtures/sample_pet.csv \
      --video-frames 100
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

def run_cmd(cmd, description):
    print("\n" + "=" * 60)
    print(f"▶ {description}")
    print("=" * 60)
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"❌ {description} failed with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"✅ {description} passed")

def main():
    parser = argparse.ArgumentParser(description="Run all NNDS validations")
    parser.add_argument("--skip-pytest", action="store_true", help="Skip pytest (use if already run separately)")
    parser.add_argument("--detections", default="outputs/petevents_bev_final_detections.csv")
    parser.add_argument("--detections-split", default="outputs/petevents_bev_final_split_detections.csv")
    parser.add_argument("--pet", default="outputs/petevents_bev_final.csv")
    parser.add_argument("--video-frames", type=int, default=1838)
    parser.add_argument("--max-gap", type=int, default=10)
    parser.add_argument("--max-jump", type=float, default=50.0)
    args = parser.parse_args()

    # 1. pytest
    if not args.skip_pytest:
        run_cmd(["pytest", "-q"], "Running unit tests")

    # 2. BEV validation
    run_cmd(["python", "scripts/validate_bev.py"], "BEV homography validation")

    # 3. Detection / tracking / PET output validation
    cmd = [
        "python", "scripts/validate_outputs.py",
        "--detections", args.detections,
        "--detections-split", args.detections_split,
        "--pet", args.pet,
        "--video-frames", str(args.video_frames),
        "--max-gap", str(args.max_gap),
        "--max-jump", str(args.max_jump),
    ]
    run_cmd(cmd, "Detection/tracking/PET output validation")

    print("\n🎉 All validations passed!")

if __name__ == "__main__":
    main()
