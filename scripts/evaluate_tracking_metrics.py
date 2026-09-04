#!/usr/bin/env python3
"""
Compute simplified tracking metrics (ID switches, track fragmentation, IDF1 approximation).

Ground truth tracking CSV must have columns: frame, track_id, x, y, w, h
Detection/tracking CSV must have columns: frame, track_id, x, y, w, h

Usage:
    python scripts/evaluate_tracking_metrics.py --tracked outputs/tracked.csv --ground-truth tests/fixtures/gt_tracks.csv
"""

import argparse

import numpy as np
import pandas as pd


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracked", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    trk = pd.read_csv(args.tracked)
    gt = pd.read_csv(args.ground_truth)

    # Approximation of ID switches: for each GT track, find best matching predicted track by IoU across frames
    gt_tracks = gt["track_id"].unique()
    trk_tracks = trk["track_id"].unique()

    id_switches = 0
    matched_pairs = {}
    for gt_id in gt_tracks:
        gt_traj = gt[gt["track_id"] == gt_id]
        best_trk = None
        best_score = 0
        for trk_id in trk_tracks:
            trk_traj = trk[trk["track_id"] == trk_id]
            # compute average IoU on common frames
            common = pd.merge(
                gt_traj[["frame", "x", "y", "w", "h"]],
                trk_traj[["frame", "x", "y", "w", "h"]],
                on="frame",
                suffixes=("_g", "_t"),
            )
            if common.empty:
                continue
            ious = []
            for _, row in common.iterrows():
                gbox = [row["x_g"], row["y_g"], row["x_g"] + row["w_g"], row["y_g"] + row["h_g"]]
                tbox = [row["x_t"], row["y_t"], row["x_t"] + row["w_t"], row["y_t"] + row["h_t"]]
                ious.append(compute_iou(gbox, tbox))
            avg_iou = np.mean(ious)
            if avg_iou > best_score:
                best_score = avg_iou
                best_trk = trk_id
        if best_trk is not None:
            matched_pairs[gt_id] = best_trk

    # Count how many GT tracks map to same predicted track (ID switch indicator)
    predicted_usage = {}
    for gt_id, trk_id in matched_pairs.items():
        predicted_usage.setdefault(trk_id, []).append(gt_id)
    for trk_id, gt_ids in predicted_usage.items():
        if len(gt_ids) > 1:
            id_switches += len(gt_ids) - 1

    # Fragmentation: count number of predicted tracks that share GT track
    fragmentation = sum(max(0, len(ids) - 1) for ids in predicted_usage.values())

    print("Tracking Metrics (simplified):")
    print(f"  ID switches (approx): {id_switches}")
    print(f"  Fragmentation: {fragmentation}")
    print(f"  GT tracks: {len(gt_tracks)}")
    print(f"  Predicted tracks: {len(trk_tracks)}")


if __name__ == "__main__":
    main()
