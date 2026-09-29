#!/usr/bin/env python3
"""
Run deconfounded sensitivity analysis:
  - Vary max_gap while holding max_jump fixed (default 30)
  - Vary max_jump while holding max_gap fixed (default 5)

Usage:
    python scripts/sensitivity_deconfounded.py --video data/sample_data/traffic_video.mp4 --max-frames 300
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_pipeline(video, frames, out_csv, max_gap, max_jump):
    subprocess.run(
        [
            sys.executable,
            "scripts/run_pipeline.py",
            "--video",
            video,
            "--max-frames",
            str(frames),
            "--imgsz",
            "640",
            "--out-csv",
            out_csv,
            "--max-gap",
            str(max_gap),
            "--max-jump",
            str(max_jump),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def parse_pet(csv_path):
    if not Path(csv_path).exists():
        return {"events": 0, "median": None, "mean": None, "std": None}
    df = pd.read_csv(csv_path)
    if df.empty:
        return {"events": 0, "median": None, "mean": None, "std": None}
    return {
        "events": len(df),
        "median": float(df["pet"].median()),
        "mean": float(df["pet"].mean()),
        "std": float(df["pet"].std()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/sample_data/traffic_video.mp4")
    parser.add_argument("--max-frames", type=int, default=300)
    args = parser.parse_args()

    gap_values = [5, 10, 15, 20]
    jump_fixed = 30.0
    jump_values = [30, 50, 80, 100]
    gap_fixed = 5

    gap_rows = []
    jump_rows = []

    for g in gap_values:
        out_csv = f"outputs/sens_gap_{g}_jump{jump_fixed}.csv"
        run_pipeline(args.video, args.max_frames, out_csv, g, jump_fixed)
        m = parse_pet(out_csv)
        m.update({"variable": "gap", "max_gap": g, "max_jump": jump_fixed})
        gap_rows.append(m)

    for j in jump_values:
        out_csv = f"outputs/sens_gap_{gap_fixed}_jump{j}.csv"
        run_pipeline(args.video, args.max_frames, out_csv, gap_fixed, j)
        m = parse_pet(out_csv)
        m.update({"variable": "jump", "max_gap": gap_fixed, "max_jump": j})
        jump_rows.append(m)

    gap_df = pd.DataFrame(gap_rows)
    jump_df = pd.DataFrame(jump_rows)
    combined = pd.concat([gap_df, jump_df], ignore_index=True)
    combined.to_csv("outputs/sensitivity_deconfounded.csv", index=False)
    print("Deconfounded sensitivity results:")
    print(combined.to_string(index=False))
    print("Saved to outputs/sensitivity_deconfounded.csv")


if __name__ == "__main__":
    main()
