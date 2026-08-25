#!/usr/bin/env python3
"""
Inspect PET events from a detailed PET CSV.

Usage:
    python scripts/inspect_pet.py --csv outputs/petevents_bev.csv --top 5
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect detailed PET events")
    parser.add_argument(
        "--csv", default="outputs/petevents_bev.csv", help="Path to PET CSV"
    )
    parser.add_argument("--top", type=int, default=10, help="Number of events to show")
    return parser.parse_args()


def summarize_traj(traj_json: str) -> str:
    """Return a compact summary of a trajectory JSON string."""
    if not traj_json or traj_json == "[]":
        return "empty"
    try:
        pts = json.loads(traj_json)
    except Exception:
        return "invalid_json"
    if not pts:
        return "empty"
    first = pts[0]
    last = pts[-1]
    return (
        f"{len(pts)} pts | first(frame={first.get('frame')}, "
        f"px=({first.get('x_pixel'):.1f},{first.get('y_pixel'):.1f}), "
        f"world=({first.get('world_x'):.1f},{first.get('world_y'):.1f})) | "
        f"last(frame={last.get('frame')}, px=({last.get('x_pixel'):.1f},{last.get('y_pixel'):.1f}), "
        f"world=({last.get('world_x'):.1f},{last.get('world_y'):.1f}))"
    )


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print("File not found:", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if df.empty:
        print("No PET events found in", csv_path)
        return

    print(f"PET Events: {len(df)} rows")

    if "grid_cell" in df.columns:
        print("Grid cell counts:")
        counts = df["grid_cell"].value_counts()
        for cell, count in counts.items():
            print(f"  {cell}: {count}")
        print()

    show_df = df.head(args.top)
    for _, row in show_df.iterrows():
        print("=" * 80)
        print(f"Event ID:        {row.get('event_id', '-')}")
        print(f"PET (s):         {row.get('pet', '-')}")
        print(f"Frame:           {row.get('frame', '-')}")
        print(f"Track A / B:     {row.get('track_a', '-')} / {row.get('track_b', '-')}")
        print(f"Conflict type:   {row.get('conflict_type', '-')}")
        print(f"Grid cell:       {row.get('grid_cell', '-')}")
        print(
            f"Entry/Exit A:    {row.get('track_a_entry_frame', '-')} / {row.get('track_a_exit_frame', '-')}"
        )
        print(
            f"Entry/Exit B:    {row.get('track_b_entry_frame', '-')} / {row.get('track_b_exit_frame', '-')}"
        )
        print(f"Traj A:          {summarize_traj(row.get('traj_a_json', '[]'))}")
        print(f"Traj B:          {summarize_traj(row.get('traj_b_json', '[]'))}")
    print("=" * 80)


if __name__ == "__main__":
    main()
