"""Detection metrics: mAP@50, mAP@75, mAP@50:95, per-class recall, size-bucketed AP.

Pure functions - no I/O, no pandas. The CLI wrapper in
``scripts/evaluate_detection_metrics.py`` converts DataFrames to these types.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

Box = tuple[float, float, float, float]

COCO_IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
COCO_SMALL_MAX_AREA: float = 32.0**2
COCO_MEDIUM_MAX_AREA: float = 96.0**2


@dataclass(frozen=True)
class Detection:
    frame: int
    box: Box
    cls: str
    conf: float


@dataclass(frozen=True)
class GroundTruth:
    frame: int
    box: Box
    cls: str


def box_area(box: Box) -> float:
    w = max(0.0, box[2] - box[0])
    h = max(0.0, box[3] - box[1])
    return w * h


def iou(box1: Box, box2: Box) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    union = box_area(box1) + box_area(box2) - inter
    return inter / union if union > 0.0 else 0.0


def average_precision_voc(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """VOC-style interpolated AP (monotone precision envelope)."""
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1]))


def _evaluate_class(
    dets_sorted: Sequence[Detection],
    gts: Sequence[GroundTruth],
    iou_thr: float,
) -> tuple[list[int], list[int], int]:
    """Greedy 1-to-1 matching, per frame, in conf-desc order.

    Returns (tp_flags, fp_flags, n_matched).
    """
    gts_by_frame: dict[int, list[GroundTruth]] = {}
    for g in gts:
        gts_by_frame.setdefault(g.frame, []).append(g)
    matched: dict[int, set[int]] = {f: set() for f in gts_by_frame}

    tp: list[int] = []
    fp: list[int] = []
    n_matched = 0
    for d in dets_sorted:
        frame_gts = gts_by_frame.get(d.frame, [])
        matched_in_frame = matched.setdefault(d.frame, set())
        best_iou = iou_thr
        best_gt = -1
        for gi, g in enumerate(frame_gts):
            if gi in matched_in_frame:
                continue
            v = iou(d.box, g.box)
            if v >= best_iou:
                best_iou = v
                best_gt = gi
        if best_gt >= 0:
            matched_in_frame.add(best_gt)
            tp.append(1)
            fp.append(0)
            n_matched += 1
        else:
            tp.append(0)
            fp.append(1)
    return tp, fp, n_matched


def _ap_from_flags(tp: list[int], fp: list[int], n_gt: int) -> float:
    if n_gt <= 0 or not tp:
        return 0.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / float(n_gt)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-9)
    return average_precision_voc(recalls, precisions)


def ap_at_iou(
    detections: Sequence[Detection],
    ground_truths: Sequence[GroundTruth],
    cls: str,
    iou_thr: float,
) -> float:
    cls_gts = [g for g in ground_truths if g.cls == cls]
    if not cls_gts:
        return 0.0
    cls_dets = sorted((d for d in detections if d.cls == cls), key=lambda d: -d.conf)
    tp, fp, _ = _evaluate_class(cls_dets, cls_gts, iou_thr)
    return _ap_from_flags(tp, fp, len(cls_gts))


def map_at_iou_range(
    detections: Sequence[Detection],
    ground_truths: Sequence[GroundTruth],
    thresholds: Iterable[float] = COCO_IOU_THRESHOLDS,
) -> dict[str, float]:
    """Return {'mAP50', 'mAP75', 'mAP50:95'}."""
    thr_list = list(thresholds)
    classes = sorted({g.cls for g in ground_truths})
    if not classes or not thr_list:
        return {"mAP50": 0.0, "mAP75": 0.0, "mAP50:95": 0.0}

    per_thr: dict[float, float] = {}
    for thr in thr_list:
        per_cls = [ap_at_iou(detections, ground_truths, c, thr) for c in classes]
        per_thr[thr] = float(np.mean(per_cls)) if per_cls else 0.0

    return {
        "mAP50": per_thr.get(0.5, 0.0),
        "mAP75": per_thr.get(0.75, 0.0),
        "mAP50:95": float(np.mean(list(per_thr.values()))),
    }


def per_class_recall(
    detections: Sequence[Detection],
    ground_truths: Sequence[GroundTruth],
    iou_thr: float = 0.5,
) -> dict[str, float]:
    classes = sorted({g.cls for g in ground_truths})
    out: dict[str, float] = {}
    for c in classes:
        cls_gts = [g for g in ground_truths if g.cls == c]
        cls_dets = sorted((d for d in detections if d.cls == c), key=lambda d: -d.conf)
        _, _, n_matched = _evaluate_class(cls_dets, cls_gts, iou_thr)
        out[c] = n_matched / len(cls_gts) if cls_gts else 0.0
    return out


def _size_bucket(
    area: float,
    small_max_area: float,
    medium_max_area: float,
) -> str:
    if area < small_max_area:
        return "small"
    if area < medium_max_area:
        return "medium"
    return "large"


def ap_by_size(
    detections: Sequence[Detection],
    ground_truths: Sequence[GroundTruth],
    iou_thr: float = 0.5,
    small_max_area: float = COCO_SMALL_MAX_AREA,
    medium_max_area: float = COCO_MEDIUM_MAX_AREA,
) -> dict[str, float]:
    """COCO-style APs / APm / APl at a single IoU threshold."""
    labels = {"small": "APs", "medium": "APm", "large": "APl"}
    result: dict[str, float] = {}
    for name, label in labels.items():
        subset = [
            g
            for g in ground_truths
            if _size_bucket(box_area(g.box), small_max_area, medium_max_area) == name
        ]
        if not subset:
            result[label] = 0.0
            continue
        classes = sorted({g.cls for g in subset})
        aps = [ap_at_iou(detections, subset, c, iou_thr) for c in classes]
        result[label] = float(np.mean(aps)) if aps else 0.0
    return result
