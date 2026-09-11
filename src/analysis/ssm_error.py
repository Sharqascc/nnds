"""PET / TTC value accuracy vs ground truth. Pure functions; no I/O.

An SSM event identifies a conflict by an unordered pair of tracks (a, b) and
carries a numeric PET and/or TTC value in seconds. Matching is by pair, so
PET values can be compared directly between GT and predictions.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SsmEvent:
    track_a: int
    track_b: int
    pet: float | None = None
    ttc: float | None = None


def pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _dedupe_min(events: Sequence[SsmEvent], field: str) -> dict[tuple[int, int], float]:
    """Collapse events by unordered pair, keeping the minimum non-None value."""
    out: dict[tuple[int, int], float] = {}
    for e in events:
        v = getattr(e, field)
        if v is None or not np.isfinite(v):
            continue
        k = pair_key(e.track_a, e.track_b)
        prev = out.get(k)
        out[k] = v if prev is None else min(prev, v)
    return out


def _value_metrics(
    pred_events: Sequence[SsmEvent],
    gt_events: Sequence[SsmEvent],
    field: str,
) -> dict[str, float]:
    pred = _dedupe_min(pred_events, field)
    gt = _dedupe_min(gt_events, field)
    common = sorted(set(pred) & set(gt))
    if not common:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "n_matched": 0,
            "n_pred_only": len(set(pred) - set(gt)),
            "n_gt_only": len(set(gt) - set(pred)),
        }
    p = np.array([pred[k] for k in common], dtype=np.float64)
    g = np.array([gt[k] for k in common], dtype=np.float64)
    errs = np.abs(p - g)
    return {
        "mae": float(np.mean(errs)),
        "rmse": float(np.sqrt(np.mean(errs**2))),
        "n_matched": int(errs.size),
        "n_pred_only": len(set(pred) - set(gt)),
        "n_gt_only": len(set(gt) - set(pred)),
    }


def pet_value_metrics(
    pred_events: Sequence[SsmEvent], gt_events: Sequence[SsmEvent]
) -> dict[str, float]:
    return _value_metrics(pred_events, gt_events, "pet")


def ttc_value_metrics(
    pred_events: Sequence[SsmEvent], gt_events: Sequence[SsmEvent]
) -> dict[str, float]:
    return _value_metrics(pred_events, gt_events, "ttc")


def ssm_value_metrics(
    pred_events: Sequence[SsmEvent], gt_events: Sequence[SsmEvent]
) -> dict[str, dict[str, float]]:
    return {
        "pet": pet_value_metrics(pred_events, gt_events),
        "ttc": ttc_value_metrics(pred_events, gt_events),
    }


def _pair_set(events: Sequence[SsmEvent], field: str) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for e in events:
        v = getattr(e, field)
        if v is None or not np.isfinite(v):
            continue
        out.add(pair_key(e.track_a, e.track_b))
    return out


def conflict_counts(
    pred_events: Sequence[SsmEvent],
    gt_events: Sequence[SsmEvent],
    field: str = "pet",
) -> dict[str, int]:
    """TP / FP / FN at the pair level for events carrying a numeric value."""
    pred = _pair_set(pred_events, field)
    gt = _pair_set(gt_events, field)
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)
    return {"tp": tp, "fp": fp, "fn": fn}


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def conflict_prf(
    pred_events: Sequence[SsmEvent],
    gt_events: Sequence[SsmEvent],
    field: str = "pet",
) -> dict[str, float]:
    c = conflict_counts(pred_events, gt_events, field=field)
    p, r, f = _prf(c["tp"], c["fp"], c["fn"])
    return {
        "precision": p,
        "recall": r,
        "f1": f,
        "tp": c["tp"],
        "fp": c["fp"],
        "fn": c["fn"],
    }


def critical_conflict_recall(
    pred_events: Sequence[SsmEvent],
    gt_events: Sequence[SsmEvent],
    field: str = "pet",
    threshold: float = 1.5,
) -> dict[str, float]:
    """Recall restricted to GT events whose SSM value is below `threshold` seconds."""
    gt_critical = {
        pair_key(e.track_a, e.track_b)
        for e in gt_events
        if getattr(e, field) is not None
        and np.isfinite(getattr(e, field))
        and float(getattr(e, field)) < threshold
    }
    if not gt_critical:
        return {"recall": 0.0, "n_critical_gt": 0, "n_hit": 0}
    pred = _pair_set(pred_events, field)
    hit = len(gt_critical & pred)
    return {
        "recall": hit / len(gt_critical),
        "n_critical_gt": len(gt_critical),
        "n_hit": hit,
    }
