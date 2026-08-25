#!/usr/bin/env python3
"""
Compute detection metrics (precision, recall, mAP@50) against ground truth.

Ground truth CSV must have columns: frame, x1, y1, x2, y2, class_name
Detection CSV must have columns: frame, x1, y1, x2, y2, class_name, conf

Usage:
    python scripts/evaluate_detection_metrics.py --detections outputs/det.csv --ground-truth tests/fixtures/ground_truth_sample.csv
"""
import argparse
import pandas as pd
import numpy as np

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def ap(recalls, precisions):
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i-1] = max(precisions[i-1], precisions[i])
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    return np.sum((recalls[indices+1] - recalls[indices]) * precisions[indices+1])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    det = pd.read_csv(args.detections)
    gt = pd.read_csv(args.ground_truth)

    det_classes = det['class_name'].unique()
    gt_classes = gt['class_name'].unique()
    all_classes = list(set(det_classes) | set(gt_classes))

    results = {}
    for cls in all_classes:
        gt_cls = gt[gt['class_name'] == cls]
        det_cls = det[det['class_name'] == cls].sort_values('conf', ascending=False).reset_index(drop=True)
        if gt_cls.empty:
            results[cls] = {'precision': 0.0, 'recall': 0.0, 'ap50': 0.0}
            continue

        tp = []
        fp = []
        matched_gt = set()
        for _, d in det_cls.iterrows():
            best_iou = 0
            best_gt_idx = -1
            for gi, g in gt_cls.iterrows():
                if gi in matched_gt:
                    continue
                box1 = [d['x1'], d['y1'], d['x2'], d['y2']]
                box2 = [g['x1'], g['y1'], g['x2'], g['y2']]
                val = iou(box1, box2)
                if val > best_iou:
                    best_iou = val
                    best_gt_idx = gi
            if best_iou >= args.iou_threshold:
                tp.append(1)
                fp.append(0)
                matched_gt.add(best_gt_idx)
            else:
                tp.append(0)
                fp.append(1)

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recalls = tp / len(gt_cls)
        precisions = tp / (tp + fp + 1e-6)
        ap_score = ap(recalls, precisions)
        final_precision = tp[-1] / (tp[-1] + fp[-1] + 1e-6)
        final_recall = tp[-1] / len(gt_cls)
        results[cls] = {
            'precision': float(final_precision),
            'recall': float(final_recall),
            'ap50': float(ap_score)
        }

    # mAP
    mAP = np.mean([r['ap50'] for r in results.values()]) if results else 0.0
    print("Detection Metrics:")
    print(f"mAP@50: {mAP:.4f}")
    for cls, r in results.items():
        print(f"  {cls}: P={r['precision']:.3f}, R={r['recall']:.3f}, AP@50={r['ap50']:.3f}")

if __name__ == "__main__":
    main()
