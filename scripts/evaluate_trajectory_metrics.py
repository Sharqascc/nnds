#!/usr/bin/env python3
"""Velocity and acceleration error vs GT trajectories.

CSV format: frame, track_id, x, y (world coordinates in meters).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    pred = _group_by_track(pd.read_csv(args.predicted))
    gt = _group_by_track(pd.read_csv(args.ground_truth))

    v_maes: list[float] = []
    v_rmses: list[float] = []
    a_maes: list[float] = []
    a_rmses: list[float] = []
    for tid in sorted(set(pred) & set(gt)):
        m = trajectory_metrics(pred[tid], gt[tid], args.fps)
        if m["velocity"]["n"] > 0:
            v_maes.append(m["velocity"]["mae"])
            v_rmses.append(m["velocity"]["rmse"])
        if m["acceleration"]["n"] > 0:
            a_maes.append(m["acceleration"]["mae"])
            a_rmses.append(m["acceleration"]["rmse"])

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    print("Trajectory Metrics:")
    print(f"  Velocity MAE (m/s):     {_mean(v_maes):.4f}")
    print(f"  Velocity RMSE (m/s):    {_mean(v_rmses):.4f}")
    print(f"  Acceleration MAE (m/s^2): {_mean(a_maes):.4f}")
    print(f"  Acceleration RMSE (m/s^2): {_mean(a_rmses):.4f}")
    print(f"  Tracks evaluated: {len(set(pred) & set(gt))}")


if __name__ == "__main__":
    main()
