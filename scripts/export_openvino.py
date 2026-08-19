#!/usr/bin/env python3
"""
Export UVH and YOLO models to OpenVINO IR for faster CPU inference.
Usage:
    python scripts/export_openvino.py [--uvh path] [--yolo path] [--imgsz 640]
"""
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uvh", default="data/models/uvh26.pt")
    parser.add_argument("--yolo", default="data/models/yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size (larger = more accuracy, slower)")
    args = parser.parse_args()

    for path in [args.uvh, args.yolo]:
        model = YOLO(path)
        print(f"Exporting {path} to OpenVINO...")
        model.export(format="openvino", imgsz=args.imgsz, dynamic=True)
    print("✅ OpenVINO export complete.")

if __name__ == "__main__":
    main()
