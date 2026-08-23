#!/usr/bin/env python3
"""
Convert DEL_4.csv (Indian trajectory dataset) to diffusion training CSV.

Output columns: event_id, frame, x_i, y_i, x_j, y_j, pet
- x_i/y_i and x_j/y_j are in metres
- frame is a 0-based integer time step
- pet is the real PET (seconds) between two road users in a shared conflict cell

Usage:
  python scripts/convert_del4_to_diffusion.py --csv data/raw/DEL_4.csv --output outputs/diffusion_del4.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to DEL_4.csv")
    parser.add_argument("--output", default="outputs/diffusion_del4.csv")
    parser.add_argument("--min-points", type=int, default=20, help="Minimum points per track")
    parser.add_argument("--max-distance", type=float, default=8.0, help="Max distance (m) to consider a pair as interacting")
    parser.add_argument("--min-frames", type=int, default=16, help="Minimum common frames per event")
    parser.add_argument("--max-events", type=int, default=5000, help="Maximum events to output")
    parser.add_argument("--cell-size", type=float, default=25.0, help="Spatial cell size for conflict detection")
    parser.add_argument("--conflict-radius", type=float, default=5.0, help="Conflict radius around cell center (m)")
    return parser.parse_args()


def compute_pet(ta, tb, shared_cells, cell_size=25.0, radius=5.0):
    """Compute PET for two tracks around any shared spatial cell center."""
    for cell in shared_cells:
        # Cell is (cx_idx, cy_idx) -> center in metres
        cx = cell[0] * cell_size + cell_size / 2.0
        cy = cell[1] * cell_size + cell_size / 2.0

        def entry_exit(track):
            d = np.sqrt((track["x"] - cx) ** 2 + (track["y"] - cy) ** 2)
            inside = np.where(d < radius)[0]
            if len(inside) == 0:
                return None
            first = inside[0]
            last = inside[-1]
            # time is rounded to 0.1s; divide by 10 for seconds
            return (float(track["time"][first]) / 10.0, float(track["time"][last]) / 10.0)

        a_interval = entry_exit(ta)
        b_interval = entry_exit(tb)
        if a_interval is None or b_interval is None:
            continue
        a_entry, a_exit = a_interval
        b_entry, b_exit = b_interval
        if a_exit <= b_entry:
            pet = b_entry - a_exit
        elif b_exit <= a_entry:
            pet = a_entry - b_exit
        else:
            pet = 0.0
        if 0.0 <= pet <= 5.0:
            return max(0.0, pet)
    return None


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    df.columns = [c.strip() for c in df.columns]
    df["Type"] = df["Type"].str.strip()
    df["Time [s]"] = pd.to_numeric(df["Time [s]"])
    df["x [m]"] = pd.to_numeric(df["x [m]"])
    df["y [m]"] = pd.to_numeric(df["y [m]"])

    # Round time to 0.1s integer units
    df["time_round"] = (df["Time [s]"] * 10).round().astype(int)

    # Keep tracks with enough points
    track_counts = df.groupby("Track ID").size()
    valid_ids = set(track_counts[track_counts >= args.min_points].index)
    df = df[df["Track ID"].isin(valid_ids)]

    # Build track dictionary
    tracks = {}
    for tid, grp in tqdm(df.groupby("Track ID"), desc="Building tracks"):
        grp = grp.sort_values("Time [s]")
        tracks[tid] = {
            "time": grp["time_round"].values,
            "x": grp["x [m]"].values,
            "y": grp["y [m]"].values,
        }

    # Spatial grid to reduce pairs
    grid = defaultdict(set)
    cell_size = args.cell_size
    for tid, tr in tracks.items():
        cells = set()
        for x, y in zip(tr["x"], tr["y"]):
            cx = int(x // cell_size)
            cy = int(y // cell_size)
            cells.add((cx, cy))
        grid[tid] = cells

    track_ids = list(tracks.keys())
    events = []
    event_id = 1

    print("Searching for interacting pairs...")
    for i in tqdm(range(len(track_ids)), desc="Pair search", unit="track"):
        tid_a = track_ids[i]
        cells_a = grid[tid_a]
        ta = tracks[tid_a]
        # Check only tracks sharing any cell
        for j in range(i+1, len(track_ids)):
            if event_id > args.max_events:
                break
            tid_b = track_ids[j]
            shared_cells = cells_a & grid[tid_b]
            if not shared_cells:
                continue
            tb = tracks[tid_b]

            # Compute common time frames
            # Find overlapping time interval
            t_start = max(ta["time"][0], tb["time"][0])
            t_end = min(ta["time"][-1], tb["time"][-1])
            if t_end - t_start < (args.min_frames - 1):
                continue

            # Create common time grid (0.1s units)
            common = np.arange(t_start, t_end + 1)  # inclusive integer steps
            # Interpolate positions to common grid
            xa = np.interp(common, ta["time"], ta["x"])
            ya = np.interp(common, ta["time"], ta["y"])
            xb = np.interp(common, tb["time"], tb["x"])
            yb = np.interp(common, tb["time"], tb["y"])

            dist = np.sqrt((xa - xb)**2 + (ya - yb)**2)
            if dist.min() >= args.max_distance:
                continue

            # Compute real PET
            pet = compute_pet(ta, tb, shared_cells, cell_size=cell_size, radius=args.conflict_radius)
            if pet is None:
                continue

            for k, f in enumerate(common):
                events.append({
                    "event_id": event_id,
                    "frame": k,
                    "x_i": float(xa[k]),
                    "y_i": float(ya[k]),
                    "x_j": float(xb[k]),
                    "y_j": float(yb[k]),
                    "pet": pet,
                })
            event_id += 1
        if event_id > args.max_events:
            break

    if not events:
        print("No events found.")
        return

    out = pd.DataFrame(events)
    out.to_csv(args.output, index=False)
    print(f"✅ Wrote {len(out)} rows for {out['event_id'].nunique()} events to {args.output}")


if __name__ == "__main__":
    main()
