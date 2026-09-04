#!/usr/bin/env python3
"""
Run tracking baselines using Ultralytics built-in trackers.

Usage:
    python scripts/run_tracking_baselines.py --video data/sample_data/traffic_video.mp4 --max-frames 100 --imgsz 640
"""

import argparse
import sys


def run_tracker(tracker_name, args):
    [
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
    # Note: run_pipeline currently doesn't accept --tracker; we'll rely on default TrackTrack for main, but this script is placeholder.
    # For now, just print command.
    print(f"Running {tracker_name} baseline (placeholder)")
    # subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/sample_data/traffic_video.mp4")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    for tracker in ["bytetrack", "botsort"]:
        run_tracker(tracker, args)


if __name__ == "__main__":
    main()
