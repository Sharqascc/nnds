#!/usr/bin/env python3
"""
Convert detailed PET CSV (with traj_a_json/traj_b_json) to diffusion training CSV.

Output columns:
    event_id, frame, x_i, y_i, x_j, y_j
"""
import argparse
import json
import pandas as pd
from pathlib import Path

def parse_traj(traj_json):
    """Return list of dicts with frame, world_x, world_y."""
    if not traj_json:
        return []
    return json.loads(traj_json)

def align_tracks(traj_a, traj_b):
    """Align tracks by common frames; return list of rows."""
    # Build frame -> point maps
    map_a = {p["frame"]: p for p in traj_a}
    map_b = {p["frame"]: p for p in traj_b}
    frames = sorted(set(map_a.keys()) & set(map_b.keys()))
    rows = []
    for f in frames:
        pa = map_a[f]
        pb = map_b[f]
        rows.append({
            "frame": f,
            "x_i": pa.get("world_x", pa.get("x_pixel", 0)),
            "y_i": pa.get("world_y", pa.get("y_pixel", 0)),
            "x_j": pb.get("world_x", pb.get("x_pixel", 0)),
            "y_j": pb.get("world_y", pb.get("y_pixel", 0)),
        })
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Detailed PET CSV")
    parser.add_argument("--output", default="outputs/petevents_diffusion_train.csv")
    parser.add_argument("--min-frames", type=int, default=5, help="Minimum common frames per event")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    all_rows = []
    for idx, row in df.iterrows():
        traj_a = parse_traj(row.get("traj_a_json", "[]"))
        traj_b = parse_traj(row.get("traj_b_json", "[]"))
        if not traj_a or not traj_b:
            continue
        rows = align_tracks(traj_a, traj_b)
        if len(rows) >= args.min_frames:
            for r in rows:
                r["event_id"] = idx  # use CSV row index as event id
                all_rows.append(r)

    if not all_rows:
        print("No aligned events found.")
        return

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(args.output, index=False)
    print(f"✅ Wrote {len(out_df)} rows for {out_df['event_id'].nunique()} events to {args.output}")

if __name__ == "__main__":
    main()
