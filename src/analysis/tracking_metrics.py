"""Tracking metrics: HOTA, MOTA, IDF1. Pure functions; no I/O, no pandas."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

Box = tuple[float, float, float, float]

HOTA_ALPHAS: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))
MOTA_IOU: float = 0.5


@dataclass(frozen=True)
class Track:
    frame: int
    track_id: int
    box: Box


def _iou(a: Box, b: Box) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _group(tracks: Sequence[Track]) -> dict[int, list[Track]]:
    out: dict[int, list[Track]] = defaultdict(list)
    for t in tracks:
        out[t.frame].append(t)
    return out


def _match_frame(
    gt_frame: Sequence[Track], pred_frame: Sequence[Track], iou_thr: float
) -> list[tuple[int, int]]:
    if not gt_frame or not pred_frame:
        return []
    ious = np.zeros((len(gt_frame), len(pred_frame)), dtype=float)
    for gi, g in enumerate(gt_frame):
        for pi, p in enumerate(pred_frame):
            ious[gi, pi] = _iou(g.box, p.box)
    order = np.dstack(np.unravel_index(np.argsort(-ious, axis=None), ious.shape))[0]
    used_g: set[int] = set()
    used_p: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for gi, pi in order:
        if ious[gi, pi] < iou_thr:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(int(gi))
        used_p.add(int(pi))
        pairs.append((int(gi), int(pi)))
    return pairs


def _counts(
    gt_by_frame: dict[int, list[Track]],
    pred_by_frame: dict[int, list[Track]],
    iou_thr: float,
) -> dict[str, int]:
    tp = fp = fn = idsw = 0
    last_match: dict[int, int] = {}
    for frame in sorted(set(gt_by_frame) | set(pred_by_frame)):
        g = gt_by_frame.get(frame, [])
        p = pred_by_frame.get(frame, [])
        pairs = _match_frame(g, p, iou_thr)
        matched_gt = {gi for gi, _ in pairs}
        matched_p = {pi for _, pi in pairs}
        for gi, pi in pairs:
            gt_id = g[gi].track_id
            pred_id = p[pi].track_id
            if gt_id in last_match and last_match[gt_id] != pred_id:
                idsw += 1
            last_match[gt_id] = pred_id
        tp += len(pairs)
        fp += len(p) - len(matched_p)
        fn += len(g) - len(matched_gt)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "idsw": idsw,
        "n_gt": sum(len(v) for v in gt_by_frame.values()),
        "n_pred": sum(len(v) for v in pred_by_frame.values()),
    }


def mota(
    tracked: Sequence[Track], ground_truth: Sequence[Track], iou_thr: float = MOTA_IOU
) -> float:
    gt_by_frame = _group(ground_truth)
    pred_by_frame = _group(tracked)
    c = _counts(gt_by_frame, pred_by_frame, iou_thr)
    n_gt = c["n_gt"]
    if n_gt == 0:
        return 1.0 if c["n_pred"] == 0 else 0.0
    return 1.0 - (c["fn"] + c["fp"] + c["idsw"]) / float(n_gt)


def _association(
    gt_by_frame: dict[int, list[Track]],
    pred_by_frame: dict[int, list[Track]],
    iou_thr: float,
) -> tuple[float, int, int, int]:
    """Return (AssA, TP, FP, FN) at the given IoU threshold."""
    gt_ids = sorted({t.track_id for v in gt_by_frame.values() for t in v})
    pred_ids = sorted({t.track_id for v in pred_by_frame.values() for t in v})
    if not gt_ids or not pred_ids:
        tp = sum(len(v) for v in pred_by_frame.values()) if not gt_ids else 0
        fn = sum(len(v) for v in gt_by_frame.values()) if not pred_ids else 0
        return 0.0, 0, tp if not gt_ids else 0, fn
    gt_idx = {tid: i for i, tid in enumerate(gt_ids)}
    pred_idx = {tid: i for i, tid in enumerate(pred_ids)}
    overlap = np.zeros((len(gt_ids), len(pred_ids)), dtype=float)
    tp = fp = fn = 0
    for frame in sorted(set(gt_by_frame) | set(pred_by_frame)):
        g = gt_by_frame.get(frame, [])
        p = pred_by_frame.get(frame, [])
        pairs = _match_frame(g, p, iou_thr)
        matched_gt = {gi for gi, _ in pairs}
        matched_p = {pi for _, pi in pairs}
        tp += len(pairs)
        fp += len(p) - len(matched_p)
        fn += len(g) - len(matched_gt)
        for gi, pi in pairs:
            overlap[gt_idx[g[gi].track_id], pred_idx[p[pi].track_id]] += 1.0
    row, col = linear_sum_assignment(-overlap)
    tpa = float(overlap[row, col].sum())
    n_gt_total = sum(len(v) for v in gt_by_frame.values())
    n_pred_total = sum(len(v) for v in pred_by_frame.values())
    fna = n_gt_total - tpa
    fpa = n_pred_total - tpa
    denom = tpa + fna + fpa
    return (tpa / denom if denom > 0 else 0.0), tp, fp, fn


def idf1(
    tracked: Sequence[Track], ground_truth: Sequence[Track], iou_thr: float = MOTA_IOU
) -> float:
    gt_by_frame = _group(ground_truth)
    pred_by_frame = _group(tracked)
    n_gt = sum(len(v) for v in gt_by_frame.values())
    n_pred = sum(len(v) for v in pred_by_frame.values())
    if n_gt == 0 and n_pred == 0:
        return 1.0
    if n_gt == 0 or n_pred == 0:
        return 0.0
    gt_ids = sorted({t.track_id for t in ground_truth})
    pred_ids = sorted({t.track_id for t in tracked})
    gt_idx = {tid: i for i, tid in enumerate(gt_ids)}
    pred_idx = {tid: i for i, tid in enumerate(pred_ids)}
    overlap = np.zeros((len(gt_ids), len(pred_ids)), dtype=float)
    for frame in sorted(set(gt_by_frame) | set(pred_by_frame)):
        g = gt_by_frame.get(frame, [])
        p = pred_by_frame.get(frame, [])
        for gi, pi in _match_frame(g, p, iou_thr):
            overlap[gt_idx[g[gi].track_id], pred_idx[p[pi].track_id]] += 1.0
    row, col = linear_sum_assignment(-overlap)
    idtp = float(overlap[row, col].sum())
    idfn = n_gt - idtp
    idfp = n_pred - idtp
    denom = 2 * idtp + idfp + idfn
    return (2 * idtp / denom) if denom > 0 else 0.0


def hota(
    tracked: Sequence[Track],
    ground_truth: Sequence[Track],
    thresholds: Iterable[float] = HOTA_ALPHAS,
) -> float:
    gt_by_frame = _group(ground_truth)
    pred_by_frame = _group(tracked)
    scores: list[float] = []
    for alpha in thresholds:
        ass_a, tp, fp, fn = _association(gt_by_frame, pred_by_frame, alpha)
        det_denom = tp + fp + fn
        det_a = tp / det_denom if det_denom > 0 else 1.0
        scores.append(float(np.sqrt(det_a * ass_a)))
    return float(np.mean(scores)) if scores else 0.0


def count_id_switches(
    tracked: Sequence[Track], ground_truth: Sequence[Track], iou_thr: float = MOTA_IOU
) -> int:
    return _counts(_group(ground_truth), _group(tracked), iou_thr)["idsw"]
