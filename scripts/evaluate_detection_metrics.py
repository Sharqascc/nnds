#!/usr/bin/env python3
"""Detection metrics: mAP@50, mAP@75, mAP@50:95, per-class recall, APs/APm/APl.

Ground truth CSV must have columns: frame, x1, y1, x2, y2, class_name
Detection CSV must have columns: frame, x1, y1, x2, y2, class_name, conf

Usage:
    python scripts/evaluate_detection_metrics.py \
        --detections outputs/det.csv \
        --ground-truth tests/fixtures/ground_truth_sample.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.detection_metrics import (
    Detection,
    GroundTruth,
    ap_by_size,
    map_at_iou_range,
    per_class_recall,
)


def _to_detections(df: pd.DataFrame) -> list[Detection]:
    frames = pd.to_numeric(df["frame"], errors="coerce").astype(int).to_numpy()
    x1 = pd.to_numeric(df["x1"], errors="coerce").astype(float).to_numpy()
    y1 = pd.to_numeric(df["y1"], errors="coerce").astype(float).to_numpy()
    x2 = pd.to_numeric(df["x2"], errors="coerce").astype(float).to_numpy()
    y2 = pd.to_numeric(df["y2"], errors="coerce").astype(float).to_numpy()
    conf = pd.to_numeric(df["conf"], errors="coerce").astype(float).to_numpy()
    cls = df["class_name"].astype(str).to_numpy()
    return [
        Detection(
            frame=int(frames[i]),
            box=(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
            cls=str(cls[i]),
            conf=float(conf[i]),
        )
        for i in range(len(df))
    ]


def _to_ground_truths(df: pd.DataFrame) -> list[GroundTruth]:
    frames = pd.to_numeric(df["frame"], errors="coerce").astype(int).to_numpy()
    x1 = pd.to_numeric(df["x1"], errors="coerce").astype(float).to_numpy()
    y1 = pd.to_numeric(df["y1"], errors="coerce").astype(float).to_numpy()
    x2 = pd.to_numeric(df["x2"], errors="coerce").astype(float).to_numpy()
    y2 = pd.to_numeric(df["y2"], errors="coerce").astype(float).to_numpy()
    cls = df["class_name"].astype(str).to_numpy()
    return [
        GroundTruth(
            frame=int(frames[i]),
            box=(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
            cls=str(cls[i]),
        )
        for i in range(len(df))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    det_df = pd.read_csv(args.detections)
    gt_df = pd.read_csv(args.ground_truth)

    dets = _to_detections(det_df)
    gts = _to_ground_truths(gt_df)

    map_scores = map_at_iou_range(dets, gts)
    recalls = per_class_recall(dets, gts, iou_thr=args.iou_threshold)
    size_ap = ap_by_size(dets, gts, iou_thr=args.iou_threshold)

    print("Detection Metrics:")
    print(f"mAP@50:   {map_scores['mAP50']:.4f}")
    print(f"mAP@75:   {map_scores['mAP75']:.4f}")
    print(f"mAP@50:95 {map_scores['mAP50:95']:.4f}")
    print("Per-class recall:")
    for cls, r in recalls.items():
        print(f"  {cls}: {r:.3f}")
    print(f"APs/APm/APl: {size_ap['APs']:.4f} / {size_ap['APm']:.4f} / {size_ap['APl']:.4f}")


if __name__ == "__main__":
    main()
