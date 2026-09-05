#!/usr/bin/env python3
"""
Export UVH and YOLO models to OpenVINO IR for faster CPU inference.
Usage:
    python scripts/export_openvino.py [--uvh path] [--yolo path] [--imgsz 640]
"""

import argparse
import os
import sys

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uvh", default="data/models/uvh26.pt")
    parser.add_argument("--yolo", default="data/models/yolo11n.pt")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Export image size (larger = more accuracy, slower)",
    )
    args = parser.parse_args()

    # Verify that the model files exist before proceeding
    for path in (args.uvh, args.yolo):
        if not os.path.isfile(path):
            print(f"❌ Error: Model file not found: '{path}'")
            sys.exit(1)

    # Load each model and export to OpenVINO, handling any errors gracefully
    for path in (args.uvh, args.yolo):
        try:
            model = YOLO(path)
            print(f"Exporting {path} to OpenVINO...")
            model.export(format="openvino", imgsz=args.imgsz, dynamic=True)
        except Exception as e:
            print(f"❌ Error processing '{path}': {e}")
            sys.exit(1)

    print("✅ OpenVINO export complete.")


if __name__ == "__main__":
    main()
