#!/usr/bin/env python3
"""End-to-end smoke test for the metric pipeline using synthetic data.

Generates fake ground truth and fake predictions, runs every metric script
with --out-json, and assembles the 15-row gold-standard table.

This is a plumbing test: it proves the pipeline runs end-to-end. It does NOT
say anything about NNDS accuracy -- the "predictions" here are synthetic.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def _write_gt_detection(path: Path, n_frames: int = 10) -> None:
    rows = []
    for f in range(n_frames):
        rows.append(
            {"frame": f, "x1": 10 + f, "y1": 20, "x2": 60 + f, "y2": 80, "class_name": "car"}
        )
        rows.append(
            {
                "frame": f,
                "x1": 100,
                "y1": 100 + f,
                "x2": 120,
                "y2": 140 + f,
                "class_name": "pedestrian",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_pred_detection(path: Path, gt: Path, jitter: float = 2.0) -> None:
    gt_df = pd.read_csv(gt)
    rng = np.random.default_rng(0)
    gt_df["x1"] += rng.normal(0, jitter, len(gt_df))
    gt_df["y1"] += rng.normal(0, jitter, len(gt_df))
    gt_df["x2"] += rng.normal(0, jitter, len(gt_df))
    gt_df["y2"] += rng.normal(0, jitter, len(gt_df))
    gt_df["conf"] = 0.9
    gt_df.to_csv(path, index=False)


def _write_gt_tracking(path: Path, n_frames: int = 10) -> None:
    rows = []
    for tid in (1, 2):
        for f in range(n_frames):
            rows.append(
                {"frame": f, "track_id": tid, "x": 10 + f + tid * 50, "y": 20, "w": 40, "h": 60}
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_gt_trajectory(path: Path, n_frames: int = 10, fps: float = 30.0) -> None:
    rows = []
    for tid in (1, 2):
        vx = 10.0 if tid == 1 else -8.0
        for f in range(n_frames):
            rows.append({"frame": f, "track_id": tid, "x": vx * f / fps, "y": 0.0})
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_pred_trajectory(path: Path, gt: Path, jitter: float = 0.05) -> None:
    gt_df = pd.read_csv(gt)
    rng = np.random.default_rng(1)
    gt_df["x"] += rng.normal(0, jitter, len(gt_df))
    gt_df["y"] += rng.normal(0, jitter, len(gt_df))
    gt_df.to_csv(path, index=False)


def _write_gt_ssm(path: Path) -> None:
    rows = [
        {"track_a": 1, "track_b": 2, "pet": 1.2, "ttc": 0.8},
        {"track_a": 3, "track_b": 4, "pet": 2.5, "ttc": 1.2},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_pred_ssm(path: Path, gt: Path, jitter: float = 0.1) -> None:
    gt_df = pd.read_csv(gt)
    rng = np.random.default_rng(2)
    gt_df["pet"] += rng.normal(0, jitter, len(gt_df))
    gt_df["ttc"] += rng.normal(0, jitter, len(gt_df))
    gt_df.to_csv(path, index=False)


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, cwd=REPO)
    if result.returncode != 0:
        raise SystemExit(f"command failed with code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/e2e_demo")
    args = parser.parse_args()

    out = Path(args.out_dir)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)

    gt_det = out / "gt_detection.csv"
    pred_det = out / "pred_detection.csv"
    gt_trk = out / "gt_tracking.csv"
    gt_traj = out / "gt_trajectory.csv"
    pred_traj = out / "pred_trajectory.csv"
    gt_ssm = out / "gt_ssm.csv"
    pred_ssm = out / "pred_ssm.csv"

    _write_gt_detection(gt_det)
    _write_pred_detection(pred_det, gt_det)
    _write_gt_tracking(gt_trk)
    _write_gt_trajectory(gt_traj)
    _write_pred_trajectory(pred_traj, gt_traj)
    _write_gt_ssm(gt_ssm)
    _write_pred_ssm(pred_ssm, gt_ssm)

    det_json = out / "detection.json"
    trk_json = out / "tracking.json"
    traj_json = out / "trajectory.json"
    ssm_json = out / "ssm.json"

    py = sys.executable

    _run(
        [
            py,
            "scripts/validate_gt.py",
            "--detection",
            str(gt_det),
            "--tracking",
            str(gt_trk),
            "--trajectory",
            str(gt_traj),
            "--ssm",
            str(gt_ssm),
        ]
    )

    _run(
        [
            py,
            "scripts/evaluate_detection_metrics.py",
            "--detections",
            str(pred_det),
            "--ground-truth",
            str(gt_det),
            "--out-json",
            str(det_json),
        ]
    )

    # Tracking metrics need ID-labelled predictions. Reuse GT as a perfect
    # prediction here -- the point is to prove the plumbing runs.
    _run(
        [
            py,
            "scripts/evaluate_tracking_metrics.py",
            "--tracked",
            str(gt_trk),
            "--ground-truth",
            str(gt_trk),
            "--out-json",
            str(trk_json),
        ]
    )

    _run(
        [
            py,
            "scripts/evaluate_trajectory_metrics.py",
            "--predicted",
            str(pred_traj),
            "--ground-truth",
            str(gt_traj),
            "--fps",
            "30",
            "--out-json",
            str(traj_json),
        ]
    )

    _run(
        [
            py,
            "scripts/evaluate_ssm_metrics.py",
            "--predicted",
            str(pred_ssm),
            "--ground-truth",
            str(gt_ssm),
            "--out-json",
            str(ssm_json),
        ]
    )

    report_md = out / "validation_report.md"
    _run(
        [
            py,
            "scripts/gold_standard_report.py",
            "--detection-metrics",
            str(det_json),
            "--tracking-metrics",
            str(trk_json),
            "--trajectory-metrics",
            str(traj_json),
            "--ssm-metrics",
            str(ssm_json),
            "--out-md",
            str(report_md),
        ]
    )

    print(f"\nDemo complete. Report: {report_md}")

    table_lines = report_md.read_text().splitlines()
    data_rows = [
        ln
        for ln in table_lines
        if ln.startswith("| ") and "---" not in ln and not ln.startswith("| Metric")
    ]
    print(f"Gold-standard rows: {len(data_rows)} (expected 15)")
    if len(data_rows) != 15:
        raise SystemExit(f"gold-standard table is {len(data_rows)} rows, expected 15")


if __name__ == "__main__":
    main()
