#!/usr/bin/env python3
"""Tracking metrics: HOTA, MOTA, IDF1, ID switches, fragmentation.

CSV format: frame, track_id, x, y, w, h (for both tracked and ground-truth)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.tracking_metrics import (
    Track,
    count_id_switches,
    hota,
    idf1,
    mota,
)


def _to_tracks(df: pd.DataFrame) -> list[Track]:
    frames = pd.to_numeric(df["frame"], errors="coerce").astype(int).to_numpy()
    tids = pd.to_numeric(df["track_id"], errors="coerce").astype(int).to_numpy()
    xs = pd.to_numeric(df["x"], errors="coerce").astype(float).to_numpy()
    ys = pd.to_numeric(df["y"], errors="coerce").astype(float).to_numpy()
    ws = pd.to_numeric(df["w"], errors="coerce").astype(float).to_numpy()
    hs = pd.to_numeric(df["h"], errors="coerce").astype(float).to_numpy()
    return [
        Track(
            frame=int(frames[i]),
            track_id=int(tids[i]),
            box=(float(xs[i]), float(ys[i]), float(xs[i] + ws[i]), float(ys[i] + hs[i])),
        )
        for i in range(len(df))
    ]


def _fragmentation(tracked: list[Track], ground_truth: list[Track]) -> int:
    gt_ids = sorted({t.track_id for t in ground_truth})
    pred_ids = sorted({t.track_id for t in tracked})
    frag = 0
    for gt_id in gt_ids:
        gt_frames = {t.frame for t in ground_truth if t.track_id == gt_id}
        matched_preds: set[int] = set()
        for pred_id in pred_ids:
            pred_frames = {t.frame for t in tracked if t.track_id == pred_id}
            if gt_frames & pred_frames:
                matched_preds.add(pred_id)
        frag += max(0, len(matched_preds) - 1)
    return frag


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracked", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    trk = _to_tracks(pd.read_csv(args.tracked))
    gt = _to_tracks(pd.read_csv(args.ground_truth))

    h = hota(trk, gt)
    i = idf1(trk, gt, iou_thr=args.iou_threshold)
    m = mota(trk, gt, iou_thr=args.iou_threshold)
    idsw = count_id_switches(trk, gt, iou_thr=args.iou_threshold)
    frag = _fragmentation(trk, gt)

    print("Tracking Metrics:")
    print(f"  HOTA:  {h:.4f}")
    print(f"  IDF1:  {i:.4f}")
    print(f"  MOTA:  {m:.4f}")
    print(f"  ID switches: {idsw}")
    print(f"  Fragmentation: {frag}")
    print(f"  GT tracks: {len({t.track_id for t in gt})}")
    print(f"  Predicted tracks: {len({t.track_id for t in trk})}")

    out_json = {"hota": h, "idf1": i, "mota": m}
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(out_json, indent=2, default=float))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
