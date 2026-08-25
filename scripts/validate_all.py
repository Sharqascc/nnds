#!/usr/bin/env python3
"""
Full end-to-end validation for NNDS repository.

Runs in order:
  1. Optional model/sample-data presence check
  2. Unit tests (pytest)
  3. BEV homography validation
  4. Optional end-to-end mini-pipeline run (default: 10 frames)
  5. Detection/tracking/PET output validation on generated or provided fixtures

Usage examples:
  # Full validation (models required, runs mini-pipeline)
  python scripts/validate_all.py --run-e2e

  # Skip e2e, use existing fixtures
  python scripts/validate_all.py

  # Skip pytest (already run)
  python scripts/validate_all.py --skip-pytest
"""

import argparse
import json
import subprocess
import sys
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run_cmd(cmd, description, **kwargs):
    print("\n" + "=" * 60)
    print(f"▶ {description}")
    print("=" * 60)
    result = subprocess.run(cmd, cwd=REPO, **kwargs)
    if result.returncode != 0:
        print(f"❌ {description} failed with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"✅ {description} passed")
    return result


def check_file_exists(path, label):
    p = REPO / path
    if p.exists():
        print(f"✅ {label} exists: {path}")
        return True
    print(f"❌ {label} missing: {path}")
    return False


def check_model_files():
    print("\n--- Model & sample data check ---")
    ok = True
    ok &= check_file_exists("data/models/uvh26.pt", "UVH model")
    ok &= check_file_exists("data/models/yolo11n.pt", "YOLO model")
    ok &= check_file_exists("data/sample_data/traffic_video.mp4", "Sample video")
    if not ok:
        print("⚠️ Missing models/data. Run `bash scripts/download_models.sh` first.")
    return ok


def run_e2e(frames=10):
    print("\n--- End-to-end mini-pipeline ---")
    out_csv = "outputs/e2e_validation_pet.csv"
    det_csv = out_csv.replace(".csv", "_detections.csv")
    split_csv = out_csv.replace(".csv", "_split_detections.csv")

    # Remove old files if exist
    for f in [out_csv, det_csv, split_csv]:
        if Path(f).exists():
            Path(f).unlink()

    # Run pipeline
    run_cmd([
        sys.executable, "scripts/run_pipeline.py",
        "--video", "data/sample_data/traffic_video.mp4",
        "--max-frames", str(frames),
        "--imgsz", "640",
        "--out-csv", out_csv,
    ], f"Running pipeline on {frames} frames")

    # Split detections
    run_cmd([
        sys.executable, "scripts/split_detections.py",
        "--input", det_csv,
        "--output", split_csv,
        "--max-gap", "5", "--max-jump", "30",
    ], "Splitting detections")

    # Validate outputs
    run_cmd([
        sys.executable, "scripts/validate_outputs.py",
        "--detections", det_csv,
        "--detections-split", split_csv,
        "--pet", out_csv,
        "--video-frames", str(frames),
    ], "Validating generated outputs")

    return out_csv


def run_full_validation(args):
    # 1. Model/sample check (optional)
    if args.check_models:
        check_model_files()
    else:
        print("\nℹ️ Skipping model/data presence check")

    # 2. Unit tests
    if not args.skip_pytest:
        run_cmd(["pytest", "-q"], "Running unit tests")
    else:
        print("\nℹ️ Skipping pytest (--skip-pytest)")

    # 3. BEV validation
    run_cmd([sys.executable, "scripts/validate_bev.py"], "BEV homography validation")

    # 4. Optional end-to-end run
    if args.run_e2e:
        run_e2e(args.frames)
    else:
        print("\nℹ️ Skipping end-to-end pipeline run (use --run-e2e to include)")

    # 5. If not e2e, validate provided fixtures or default
    if not args.run_e2e:
        det = args.detections
        det_split = args.detections_split
        pet = args.pet
        run_cmd([
            sys.executable, "scripts/validate_outputs.py",
            "--detections", det,
            "--detections-split", det_split,
            "--pet", pet,
            "--video-frames", str(args.video_frames),
        ], "Detection/tracking/PET output validation")

    # Generate scientific validation report (if detections/pet files exist)
    det_for_report = args.detections
    pet_for_report = args.pet
    if args.run_e2e:
        # Use generated files from e2e
        det_for_report = f"outputs/e2e_validation_pet_detections.csv"
        pet_for_report = f"outputs/e2e_validation_pet.csv"
    if Path(det_for_report).exists() and Path(pet_for_report).exists():
        run_cmd([
            sys.executable, "scripts/validation_report.py",
            "--detections", det_for_report,
            "--pet", pet_for_report,
            "--bev-config", "configs/bev_config.json",
            "--calib", "configs/giti_calibration_points.json",
            "--output", "outputs/validation_report.md",
        ], "Generating quantitative validation report")
    else:
        print("\nℹ️ Skipping validation report due to missing detections/pet files")

    print("\n🎉 All validations passed!")


def main():
    parser = argparse.ArgumentParser(description="Full end-to-end NNDS validation")
    parser.add_argument("--skip-pytest", action="store_true", help="Skip unit tests")
    parser.add_argument("--check-models", action="store_true", help="Check model and sample data presence")
    parser.add_argument("--run-e2e", action="store_true", help="Run mini-pipeline (requires models)")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames for e2e")
    parser.add_argument("--detections", default="tests/fixtures/sample_detections.csv")
    parser.add_argument("--detections-split", default="tests/fixtures/sample_split_detections.csv")
    parser.add_argument("--pet", default="tests/fixtures/sample_pet.csv")
    parser.add_argument("--video-frames", type=int, default=100)
    args = parser.parse_args()
    run_full_validation(args)


if __name__ == "__main__":
    main()
