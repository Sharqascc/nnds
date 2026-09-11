#!/usr/bin/env python3
"""Velocity, acceleration, and position error vs GT trajectories.

CSV format: frame, track_id, x, y (world coordinates in meters).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.traj_error import Point, trajectory_metrics


def _group_by_track(df: pd.DataFrame) -> dict[int, list[Point]]:
    frames = pd.to_numeric(df["frame"], errors="coerce").astype(int).to_numpy()
    tids = pd.to_numeric(df["track_id"], errors="coerce").astype(int).to_numpy()
    xs = pd.to_numeric(df["x"], errors="coerce").astype(float).to_numpy()
    ys = pd.to_numeric(df["y"], errors="coerce").astype(float).to_numpy()
    out: dict[int, list[Point]] = {}
    for i in range(len(df)):
        out.setdefault(int(tids[i]), []).append((int(frames[i]), float(xs[i]), float(ys[i])))
    return out


def _mean(xs):
    return float(sum(xs) / len(xs)) if xs else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    pred = _group_by_track(pd.read_csv(args.predicted))
    gt = _group_by_track(pd.read_csv(args.ground_truth))

    v_maes, v_rmses, a_maes, a_rmses, pos_rmses = [], [], [], [], []
    for tid in sorted(set(pred) & set(gt)):
        m = trajectory_metrics(pred[tid], gt[tid], args.fps)
        if m["velocity"]["n"] > 0:
            v_maes.append(m["velocity"]["mae"])
            v_rmses.append(m["velocity"]["rmse"])
        if m["acceleration"]["n"] > 0:
            a_maes.append(m["acceleration"]["mae"])
            a_rmses.append(m["acceleration"]["rmse"])
        # Position RMSE between paired (frame, x, y) points
        gd = {f: (x, y) for f, x, y in gt[tid]}
        errs = []
        for f, x, y in pred[tid]:
            if f in gd:
                gx, gy = gd[f]
                errs.append((x - gx) ** 2 + (y - gy) ** 2)
        if errs:
            pos_rmses.append(float(np.sqrt(np.mean(errs))))

    print("Trajectory Metrics:")
    print(f"  Position RMSE (m):        {_mean(pos_rmses):.4f}")
    print(f"  Velocity MAE (m/s):       {_mean(v_maes):.4f}")
    print(f"  Velocity RMSE (m/s):      {_mean(v_rmses):.4f}")
    print(f"  Acceleration MAE (m/s^2): {_mean(a_maes):.4f}")
    print(f"  Acceleration RMSE (m/s^2):{_mean(a_rmses):.4f}")
    print(f"  Tracks evaluated:         {len(set(pred) & set(gt))}")

    out_json = {
        "position_rmse_m": _mean(pos_rmses),
        "velocity_mae_mps": _mean(v_maes),
        "accel_mae_mps2": _mean(a_maes),
    }
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(out_json, indent=2, default=float))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
