#!/usr/bin/env python3
"""
Diagnose tracking instability by analyzing detections CSV.

For each track, computes:
  - frame span and number of detections
  - max frame gap between consecutive detections
  - max spatial jump (in pixels) between consecutive detections
  - average jump

Flags suspicious tracks: max_gap > 10 frames OR max_jump > 50 pixels.

Usage:
    python scripts/diagnose_tracking.py --csv outputs/petevents_bev_300_split_detections.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def analyze_track(group):
    """Analyze a single track group."""
    group = group.sort_values("frame")
    frames = group["frame"].values
    x = group["cx"].values
    y = group["cy"].values

    if len(group) < 2:
        return {
            "num_detections": len(group),
            "start_frame": frames[0] if len(group) else None,
            "end_frame": frames[-1] if len(group) else None,
            "max_gap": 0,
            "max_jump": 0.0,
            "avg_jump": 0.0,
            "flag": False,
        }

    gaps = np.diff(frames)
    max_gap = int(gaps.max()) if len(gaps) > 0 else 0

    dx = np.diff(x)
    dy = np.diff(y)
    jumps = np.sqrt(dx**2 + dy**2)
    max_jump = float(jumps.max()) if len(jumps) > 0 else 0.0
    avg_jump = float(jumps.mean()) if len(jumps) > 0 else 0.0

    flag = bool(max_gap > 10 or max_jump > 50.0)

    return {
        "num_detections": len(group),
        "start_frame": int(frames[0]),
        "end_frame": int(frames[-1]),
        "max_gap": max_gap,
        "max_jump": max_jump,
        "avg_jump": avg_jump,
        "flag": flag,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="outputs/petevents_bev_300_split_detections.csv")
    parser.add_argument("--report", default="outputs/tracking_diagnosis.csv")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "track_id" not in df.columns or "frame" not in df.columns:
        raise ValueError("CSV must contain 'track_id' and 'frame' columns")

    rows = []
    for track_id, group in df.groupby("track_id"):
        info = analyze_track(group)
        info["track_id"] = track_id
        rows.append(info)

    report_df = pd.DataFrame(rows).sort_values("track_id")
    report_df.to_csv(args.report, index=False)

    flagged = report_df[report_df["flag"]]
    print(f"Total tracks analyzed: {len(report_df)}")
    print(f"Suspicious tracks flagged: {len(flagged)}")
    print(f"Report saved to {args.report}")

    if not flagged.empty:
        print("\nSuspicious track list (top 20):")
        print(flagged.head(20).to_string(index=False))
    else:
        print("No suspicious tracks found.")


if __name__ == "__main__":
    main()
