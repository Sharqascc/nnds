#!/usr/bin/env python3
"""
Prepare split detections CSV for validation.
Applies the same gap/jump track splitting as the pipeline.

Usage:
    python scripts/split_detections.py \
        --input outputs/petevents_bev_final_detections.csv \
        --output outputs/petevents_bev_final_split_detections.csv \
        --max-gap 5 --max-jump 30
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def split_track(group, max_gap=5, max_jump=30.0):
    group = group.sort_values("frame")
    if group.empty:
        return group
    new_ids = []
    segment = 0
    base_id = group.iloc[0]["track_id"]
    new_ids.append(base_id * 1000 + segment)
    frames = group["frame"].values
    x = group["cx"].values
    y = group["cy"].values
    for i in range(1, len(group)):
        gap = frames[i] - frames[i - 1]
        dx = x[i] - x[i - 1]
        dy = y[i] - y[i - 1]
        dist = np.sqrt(dx**2 + dy**2)
        if gap > max_gap or dist > max_jump:
            segment += 1
        new_ids.append(base_id * 1000 + segment)
    group = group.copy()
    group["track_id"] = new_ids
    return group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-gap", type=int, default=5)
    parser.add_argument("--max-jump", type=float, default=30.0)
    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    df = pd.read_csv(input_path)
    if "track_id" not in df.columns:
        raise ValueError("CSV must contain 'track_id'")
    split_groups = [
        split_track(g, args.max_gap, args.max_jump) for _, g in df.groupby("track_id")
    ]
    result = (
        pd.concat(split_groups)
        .sort_values(["frame", "track_id"])
        .reset_index(drop=True)
    )
    result.to_csv(args.output, index=False)
    print(f"Original tracks: {df['track_id'].nunique()}")
    print(f"Split tracks:    {result['track_id'].nunique()}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
