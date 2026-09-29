#!/usr/bin/env python3
"""Detection metrics: mAP@50, mAP@75, mAP@50:95, per-class recall, APs/APm/APl.

Ground truth CSV columns: frame, x1, y1, x2, y2, class_name
Detection CSV columns:    frame, x1, y1, x2, y2, class_name, conf
"""

from __future__ import annotations

import argparse
import json
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
    precision_recall_f1,
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
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    det_df = pd.read_csv(args.detections)
    gt_df = pd.read_csv(args.ground_truth)

    dets = _to_detections(det_df)
    gts = _to_ground_truths(gt_df)

    map_scores = map_at_iou_range(dets, gts)
    recalls = per_class_recall(dets, gts, iou_thr=args.iou_threshold)
    size_ap = ap_by_size(dets, gts, iou_thr=args.iou_threshold)
    prf = precision_recall_f1(dets, gts, iou_thr=args.iou_threshold)

    mean_recall = float(sum(recalls.values()) / len(recalls)) if recalls else 0.0
    print("Detection Metrics:")
    print(f"mAP@50:   {map_scores['mAP50']:.4f}")
    print(f"mAP@75:   {map_scores['mAP75']:.4f}")
    print(f"mAP@50:95 {map_scores['mAP50:95']:.4f}")
    print("Per-class recall:")
    for cls, r in recalls.items():
        print(f"  {cls}: {r:.3f}")
    print(f"APs/APm/APl: {size_ap['APs']:.4f} / {size_ap['APm']:.4f} / {size_ap['APl']:.4f}")
    print(f"Precision: {prf['precision']:.4f}")
    print(f"Recall:    {prf['recall']:.4f}")
    print(f"F1:        {prf['f1']:.4f}")

    out_json = {
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "map50": map_scores["mAP50"],
        "map50_95": map_scores["mAP50:95"],
        "ap75": map_scores["mAP75"],
    }
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(out_json, indent=2, default=float))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
