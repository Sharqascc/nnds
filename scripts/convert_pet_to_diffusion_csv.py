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

def align_tracks(traj_a, traj_b, pet_value=None, scale=0.001):
    """Align tracks by common frames; return list of rows centered at each track's start."""
    map_a = {p["frame"]: p for p in traj_a}
    map_b = {p["frame"]: p for p in traj_b}
    frames = sorted(set(map_a.keys()) & set(map_b.keys()))
    if not frames:
        return []

    start_a = map_a[frames[0]]
    start_b = map_b[frames[0]]

    def get_world_x(p):
        return p.get("world_x", p.get("x_pixel", 0))
    def get_world_y(p):
        return p.get("world_y", p.get("y_pixel", 0))

    start_a_x = get_world_x(start_a)
    start_a_y = get_world_y(start_a)
    start_b_x = get_world_x(start_b)
    start_b_y = get_world_y(start_b)

    rows = []
    for f in frames:
        pa = map_a[f]
        pb = map_b[f]
        rows.append({
            "frame": f,
            "x_i": (get_world_x(pa) - start_a_x) * scale,
            "y_i": (get_world_y(pa) - start_a_y) * scale,
            "x_j": (get_world_x(pb) - start_b_x) * scale,
            "y_j": (get_world_y(pb) - start_b_y) * scale,
            "pet": pet_value,
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
        rows = align_tracks(traj_a, traj_b, pet_value=row.get('pet', None), scale=0.001)
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
