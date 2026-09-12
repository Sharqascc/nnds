#!/usr/bin/env python3
"""End-to-end smoke test for the metric pipeline using synthetic data.

Generates fake ground truth and fake pipeline output, runs the converter to
produce PRED CSVs in the metric schemas, runs every metric script with
--out-json, and assembles the 15-row gold-standard table.

This is a plumbing test: it proves the pipeline runs end-to-end, including
the run_pipeline.py -> metric-schema conversion step. It does NOT say
anything about NNDS accuracy -- the "predictions" here are synthetic.
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


def _write_pipeline_detections(path: Path, n_frames: int = 10, jitter: float = 2.0) -> None:
    """Write a CSV in run_pipeline.py's detection schema."""
    rng = np.random.default_rng(0)
    rows = []
    for f in range(n_frames):
        for tid, base in ((1, (10 + f, 20)), (2, (100, 100 + f))):
            x1 = base[0] + float(rng.normal(0, jitter))
            y1 = base[1] + float(rng.normal(0, jitter))
            x2 = x1 + (50 if tid == 1 else 20)
            y2 = y1 + (60 if tid == 1 else 40)
            cls = "car" if tid == 1 else "pedestrian"
            rows.append(
                {
                    "frame": f,
                    "track_id": tid,
                    "class_id": 0 if tid == 1 else 1,
                    "class_name": cls,
                    "conf": 0.9,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": (x1 + x2) / 2.0,
                    "cy": (y1 + y2) / 2.0,
                    "source": "uvh26",
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


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


def _write_pipeline_pet(path: Path) -> None:
    rows = [
        {
            "event_id": 0,
            "site": "GITI",
            "pet": 1.2,
            "conflict_type": "crossing",
            "grid_cell": "A1",
            "orig_track_a": 1,
            "orig_track_b": 2,
        },
        {
            "event_id": 1,
            "site": "GITI",
            "pet": 2.5,
            "conflict_type": "rear_end",
            "grid_cell": "B2",
            "orig_track_a": 3,
            "orig_track_b": 4,
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_gt_ssm(path: Path) -> None:
    rows = [
        {"track_a": 1, "track_b": 2, "pet": 1.2, "ttc": 0.8},
        {"track_a": 3, "track_b": 4, "pet": 2.5, "ttc": 1.2},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_bev_config(path: Path, scale: float = 0.05) -> None:
    H = [[scale, 0.0, -1.0], [0.0, scale, -1.0], [0.0, 0.0, 1.0]]
    path.write_text(json.dumps({"H_pixel_to_world": H}))


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
    gt_trk = out / "gt_tracking.csv"
    gt_traj = out / "gt_trajectory.csv"
    gt_ssm = out / "gt_ssm.csv"
    pipeline_det = out / "pipeline_detections.csv"
    pipeline_pet = out / "pipeline_pet.csv"
    bev_cfg = out / "bev_config.json"
    schemas = out / "schemas"

    _write_gt_detection(gt_det)
    _write_gt_tracking(gt_trk)
    _write_gt_trajectory(gt_traj)
    _write_gt_ssm(gt_ssm)
    _write_pipeline_detections(pipeline_det)
    _write_pipeline_pet(pipeline_pet)
    _write_bev_config(bev_cfg)

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
            "scripts/pipeline_to_metric_schemas.py",
            "--detections-csv",
            str(pipeline_det),
            "--pet-csv",
            str(pipeline_pet),
            "--bev-config",
            str(bev_cfg),
            "--out-dir",
            str(schemas),
        ]
    )

    pred_det = schemas / "pred_detection.csv"
    pred_trk = schemas / "pred_tracking.csv"
    pred_traj = schemas / "pred_trajectory.csv"
    pred_ssm = schemas / "pred_ssm.csv"

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

    _run(
        [
            py,
            "scripts/evaluate_tracking_metrics.py",
            "--tracked",
            str(pred_trk),
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
