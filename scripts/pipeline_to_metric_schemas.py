#!/usr/bin/env python3
"""Convert NNDS pipeline output CSVs into the schemas the metric scripts use.

Inputs:
  --detections-csv  : outputs/*_detections.csv from run_pipeline.py
  --pet-csv         : outputs/*.csv (PET events)
  --bev-config      : JSON containing "H_pixel_to_world" (3x3)
  --out-dir         : directory to write the four PRED CSVs into
  --site            : optional; if set, only PET events with matching 'site' are written

Writes:
  pred_detection.csv    frame, x1, y1, x2, y2, class_name, conf
  pred_tracking.csv     frame, track_id, x, y, w, h
  pred_trajectory.csv   frame, track_id, x, y         (world meters)
  pred_ssm.csv          track_a, track_b, pet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.pipeline_outputs import (
    detections_from_pipeline_rows,
    ssm_events_from_pet_rows,
    tracks_from_pipeline_rows,
    traj_points_to_rows,
    trajectories_from_pipeline_rows,
)


def _read_csv(path: str) -> list[dict]:
    return pd.read_csv(path).to_dict(orient="records")


def _write_detection(dets, path: Path) -> None:
    rows = [
        {
            "frame": d.frame,
            "x1": d.box[0],
            "y1": d.box[1],
            "x2": d.box[2],
            "y2": d.box[3],
            "class_name": d.cls,
            "conf": d.conf,
        }
        for d in dets
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_tracking(tracks, path: Path) -> None:
    rows = [
        {
            "frame": t.frame,
            "track_id": t.track_id,
            "x": t.box[0],
            "y": t.box[1],
            "w": t.box[2] - t.box[0],
            "h": t.box[3] - t.box[1],
        }
        for t in tracks
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_trajectory(rows_world, path: Path) -> None:
    pd.DataFrame(rows_world).to_csv(path, index=False)


def _write_ssm(events, path: Path) -> None:
    rows = [
        {"track_a": e.track_a, "track_b": e.track_b, "pet": e.pet}
        for e in events
        if e.pet is not None
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections-csv", required=True)
    parser.add_argument("--pet-csv", default=None)
    parser.add_argument("--bev-config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--site", default=None)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    det_rows = _read_csv(args.detections_csv)

    H_raw = json.loads(Path(args.bev_config).read_text())["H_pixel_to_world"]
    H = np.array(H_raw, dtype=np.float64)

    _write_detection(detections_from_pipeline_rows(det_rows), out / "pred_detection.csv")
    _write_tracking(tracks_from_pipeline_rows(det_rows), out / "pred_tracking.csv")
    _write_trajectory(
        traj_points_to_rows(trajectories_from_pipeline_rows(det_rows, H)),
        out / "pred_trajectory.csv",
    )

    if args.pet_csv:
        pet_rows = _read_csv(args.pet_csv)
        if args.site:
            pet_rows = [r for r in pet_rows if str(r.get("site", "")) == args.site]
        _write_ssm(ssm_events_from_pet_rows(pet_rows), out / "pred_ssm.csv")

    print(f"Wrote: {out}/pred_detection.csv")
    print(f"Wrote: {out}/pred_tracking.csv")
    print(f"Wrote: {out}/pred_trajectory.csv")
    if args.pet_csv:
        print(f"Wrote: {out}/pred_ssm.csv")


if __name__ == "__main__":
    main()
