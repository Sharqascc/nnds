#!/usr/bin/env python3
"""
Ensure required models are present and exported to OpenVINO (if possible).

Automatically downloads PyTorch weights if missing, then exports OpenVINO
directories for faster CPU inference.

Usage:
    python scripts/ensure_models.py [--imgsz 640] [--skip-openvino]
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, description):
    print(f"\n[setup] {description}...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[setup] ❌ Failed: {description}")
        sys.exit(result.returncode)


def download_models():
    script = Path(__file__).parent / "download_models.sh"
    if script.exists():
        run_cmd(["bash", str(script)], "Downloading base models")
    else:
        print("⚠️ download_models.sh not found; skipping download")


def export_openvino(imgsz=640):
    script = Path(__file__).parent / "export_openvino.py"
    if script.exists():
        run_cmd([sys.executable, str(script), "--imgsz", str(imgsz)], "Exporting OpenVINO IR models")
    else:
        print("⚠️ export_openvino.py not found; skipping OpenVINO export")


def dirs_exist(paths):
    return all(Path(p).exists() for p in paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--skip-openvino", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    models_dir = root / "data" / "models"
    uvh_pt = models_dir / "uvh26.pt"
    yolo_pt = models_dir / "yolo11n.pt"
    uvh_ov = models_dir / "uvh26_openvino_model"
    yolo_ov = models_dir / "yolo11n_openvino_model"

    # 1. Download if missing
    if not (uvh_pt.exists() and yolo_pt.exists()):
        download_models()

    # 2. Export OpenVINO if directories missing
    if not args.skip_openvino:
        if not (uvh_ov.exists() and yolo_ov.exists()):
            export_openvino(args.imgsz)
        else:
            print("[setup] OpenVINO models already exist; skipping export")
    else:
        print("[setup] Skipping OpenVINO export due to --skip-openvino")

    print("[setup] ✅ Model preparation complete")


if __name__ == "__main__":
    main()
