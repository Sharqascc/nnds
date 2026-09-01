#!/usr/bin/env python3
"""
Compute precision at multiple confidence thresholds.

Usage:
    python scripts/detection_confidence_analysis.py --detections outputs/det.csv --ground-truth tests/fixtures/ground_truth_sample.csv
"""
import argparse

import pandas as pd


def iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1)*max(0, y2-y1)
    area1 = max(0, box1[2]-box1[0])*max(0, box1[3]-box1[1])
    area2 = max(0, box2[2]-box2[0])*max(0, box2[3]-box2[1])
    union = area1+area2-inter
    return inter/union if union>0 else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()
    det = pd.read_csv(args.detections)
    gt = pd.read_csv(args.ground_truth)
    thresholds = [0.25, 0.5, 0.75, 0.9]
    for th in thresholds:
        det_th = det[det['conf'] >= th]
        # count matches per frame (simple greedy)
        matched = 0
        total_gt = 0
        for frame in sorted(gt['frame'].unique()):
            gt_frame = gt[gt['frame']==frame]
            det_frame = det_th[det_th['frame']==frame]
            total_gt += len(gt_frame)
            used = set()
            for _, g in gt_frame.iterrows():
                best_iou = 0
                best_idx = -1
                for di, d in det_frame.iterrows():
                    if di in used: continue
                    iou_val = iou([g['x1'],g['y1'],g['x2'],g['y2']], [d['x1'],d['y1'],d['x2'],d['y2']])
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_idx = di
                if best_iou >= args.iou_threshold:
                    matched += 1
                    used.add(best_idx)
        precision = matched / len(det_th) if len(det_th)>0 else 0
        recall = matched / total_gt if total_gt>0 else 0
        print(f"Confidence >= {th}: Precision={precision:.3f}, Recall={recall:.3f}")

if __name__ == "__main__":
    main()
