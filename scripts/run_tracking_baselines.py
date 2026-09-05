#!/usr/bin/env python3
"""
Run tracking baselines using Ultralytics built-in trackers.

Usage:
    python scripts/run_tracking_baselines.py --video data/sample_data/traffic_video.mp4 --max-frames 100 --imgsz 640
"""

import argparse
import subprocess
import sys


def run_tracker(tracker_name, args):
    cmd = [
        sys.executable,
        "scripts/run_pipeline.py",
        "--video",
        args.video,
        "--max-frames",
        str(args.max_frames),
        "--imgsz",
        str(args.imgsz),
        "--out-csv",
        f"outputs/baseline_{tracker_name}_pet.csv",
        "--tracker",
        tracker_name if tracker_name in ["bytetrack", "botsort"] else "bytetrack",
    ]
    # Execute the pipeline
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/sample_data/traffic_video.mp4")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    if args.max_frames <= 0:
        parser.error("--max-frames must be a positive integer")
    if args.imgsz <= 0:
        parser.error("--imgsz must be a positive integer")

    for tracker in ["bytetrack", "botsort"]:
        run_tracker(tracker, args)


if __name__ == "__main__":
    main()
