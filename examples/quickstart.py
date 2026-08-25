#!/usr/bin/env python3
"""
Quickstart example: run the full pipeline on the sample video and validate outputs.

Usage:
    python examples/quickstart.py
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

def run(cmd, desc):
    print(f"\n=== {desc} ===")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

def main():
    # 1. Run pipeline on 100 frames
    run([
        sys.executable, "scripts/run_pipeline.py",
        "--video", "data/sample_data/traffic_video.mp4",
        "--max-frames", "100",
        "--imgsz", "640",
        "--out-csv", "outputs/quickstart_pet.csv"
    ], "Step 1: Run pipeline (100 frames)")

    # 2. Split detections for validation
    run([
        sys.executable, "scripts/split_detections.py",
        "--input", "outputs/quickstart_pet_detections.csv",
        "--output", "outputs/quickstart_pet_split_detections.csv",
        "--max-gap", "5", "--max-jump", "30"
    ], "Step 2: Split detections")

    # 3. Validate outputs
    run([
        sys.executable, "scripts/validate_outputs.py",
        "--detections", "outputs/quickstart_pet_detections.csv",
        "--detections-split", "outputs/quickstart_pet_split_detections.csv",
        "--pet", "outputs/quickstart_pet.csv",
        "--video-frames", "100"
    ], "Step 3: Validate outputs")

    print("\n✅ Quickstart complete! Check outputs/quickstart_pet.csv for results.")

if __name__ == "__main__":
    main()
