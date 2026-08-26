#!/usr/bin/env python3
"""
Run sensitivity analysis for PET metrics vs. tracking fragmentation thresholds.

Usage:
    python scripts/sensitivity_pet_fragmentation.py --video data/sample_data/traffic_video.mp4 --max-frames 100
"""
import argparse
import subprocess
import sys
import pandas as pd
from pathlib import Path
import json

def run_pipeline(video, frames, out_csv, max_gap, max_jump):
    subprocess.run([
        sys.executable, "scripts/run_pipeline.py",
        "--video", video,
        "--max-frames", str(frames),
        "--imgsz", "640",
        "--out-csv", out_csv,
        "--max-gap", str(max_gap),
        "--max-jump", str(max_jump),
    ], check=True, capture_output=True, text=True)

def parse_pet(csv_path):
    if not Path(csv_path).exists():
        return {"events":0, "median":None, "mean":None}
    df = pd.read_csv(csv_path)
    if df.empty:
        return {"events":0, "median":None, "mean":None}
    return {
        "events": len(df),
        "median": float(df['pet'].median()),
        "mean": float(df['pet'].mean()),
        "std": float(df['pet'].std()),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/sample_data/traffic_video.mp4")
    parser.add_argument("--max-frames", type=int, default=100)
    args = parser.parse_args()

    thresholds = [
        {"max_gap": 5, "max_jump": 30},
        {"max_gap": 10, "max_jump": 50},
        {"max_gap": 15, "max_jump": 80},
        {"max_gap": 20, "max_jump": 100},
    ]
    results = []
    for th in thresholds:
        out_csv = f"outputs/sensitivity_pet_gap{th['max_gap']}_jump{th['max_jump']}.csv"
        run_pipeline(args.video, args.max_frames, out_csv, th['max_gap'], th['max_jump'])
        metrics = parse_pet(out_csv)
        metrics.update(th)
        results.append(metrics)

    df = pd.DataFrame(results)
    df.to_csv("outputs/sensitivity_analysis.csv", index=False)
    print("Sensitivity Analysis Results:")
    print(df.to_string(index=False))
    print("Saved to outputs/sensitivity_analysis.csv")

if __name__ == "__main__":
    main()
