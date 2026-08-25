#!/usr/bin/env python3
"""
Evaluate detections against ground truth annotations.

Expected ground truth format: CSV with columns:
    frame, x1, y1, x2, y2, class_name

Detection CSV from pipeline has additional columns, but we only need
frame and bounding boxes for IoU-based matching.

Usage:
    python scripts/evaluate_ground_truth.py \
        --detections outputs/petevents_bev_detections.csv \
        --ground-truth path/to/gt.csv \
        --iou-threshold 0.5
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
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
    parser.add_argument("--detections", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    det = pd.read_csv(args.detections)
    gt = pd.read_csv(args.ground_truth)

    # Filter detections to frames present in GT
    det_frames = set(det['frame'].unique())
    gt_frames = set(gt['frame'].unique())
    common_frames = det_frames & gt_frames
    print(f"Frames with GT: {len(gt_frames)}, frames with detections: {len(det_frames)}, common: {len(common_frames)}")

    matched = 0
    total_gt = 0
    total_det = 0

    for frame in sorted(common_frames):
        gt_frame = gt[gt['frame'] == frame]
        det_frame = det[det['frame'] == frame]
        total_gt += len(gt_frame)
        total_det += len(det_frame)

        matched_gt = set()
        matched_det = set()
        for gi, g in gt_frame.iterrows():
            best_iou = 0
            best_di = -1
            for di, d in det_frame.iterrows():
                if di in matched_det:
                    continue
                box1 = [g['x1'], g['y1'], g['x2'], g['y2']]
                box2 = [d['x1'], d['y1'], d['x2'], d['y2']]
                iou = compute_iou(box1, box2)
                if iou > best_iou:
                    best_iou = iou
                    best_di = di
            if best_iou >= args.iou_threshold:
                matched += 1
                matched_gt.add(gi)
                matched_det.add(best_di)

    precision = matched / total_det if total_det > 0 else 0
    recall = matched / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {f1:.4f}")

if __name__ == "__main__":
    main()
