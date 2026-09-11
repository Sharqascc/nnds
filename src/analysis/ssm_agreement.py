"""Agreement metrics between GT and predicted SSM values. Pure functions.

Focus: does the perception pipeline preserve the safety information in the
ground-truth trajectories? Reports MAE, RMSE, R^2, and Spearman rho on the
matched (unordered track-pair) subset.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats

from src.analysis.ssm_error import SsmEvent, pair_key


def _aligned_arrays(
    pred_events: Sequence[SsmEvent],
    gt_events: Sequence[SsmEvent],
    field: str,
) -> tuple[np.ndarray, np.ndarray]:
    pred_map: dict[tuple[int, int], float] = {}
    for e in pred_events:
        v = getattr(e, field)
        if v is None or not np.isfinite(v):
            continue
        k = pair_key(e.track_a, e.track_b)
        prev = pred_map.get(k)
        pred_map[k] = v if prev is None else min(prev, v)
    gt_map: dict[tuple[int, int], float] = {}
    for e in gt_events:
        v = getattr(e, field)
        if v is None or not np.isfinite(v):
            continue
        k = pair_key(e.track_a, e.track_b)
        prev = gt_map.get(k)
        gt_map[k] = v if prev is None else min(prev, v)
    common = sorted(set(pred_map) & set(gt_map))
    if not common:
        return np.zeros((0,)), np.zeros((0,))
    return (
        np.array([pred_map[k] for k in common], dtype=np.float64),
        np.array([gt_map[k] for k in common], dtype=np.float64),
    )


def _r_squared(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.size < 2:
        return 0.0
    ss_res = float(np.sum((gt - pred) ** 2))
    ss_tot = float(np.sum((gt - np.mean(gt)) ** 2))
    if ss_tot <= 0.0:
        return 1.0 if ss_res <= 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def _spearman(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.size < 3:
        return 0.0
    if np.all(pred == pred[0]) or np.all(gt == gt[0]):
        return 0.0
    rho, _ = stats.spearmanr(pred, gt)
    if not np.isfinite(rho):
        return 0.0
    return float(rho)


def agreement_metrics(
    pred_events: Sequence[SsmEvent],
    gt_events: Sequence[SsmEvent],
    field: str = "pet",
) -> dict[str, float]:
    pred, gt = _aligned_arrays(pred_events, gt_events, field)
    if pred.size == 0:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "r2": 0.0,
            "spearman": 0.0,
            "n": 0,
        }
    errs = np.abs(pred - gt)
    return {
        "mae": float(np.mean(errs)),
        "rmse": float(np.sqrt(np.mean(errs**2))),
        "r2": _r_squared(pred, gt),
        "spearman": _spearman(pred, gt),
        "n": int(pred.size),
    }


def pet_vs_pet(pred_events: Sequence[SsmEvent], gt_events: Sequence[SsmEvent]) -> dict[str, float]:
    return agreement_metrics(pred_events, gt_events, "pet")


def ttc_vs_ttc(pred_events: Sequence[SsmEvent], gt_events: Sequence[SsmEvent]) -> dict[str, float]:
    return agreement_metrics(pred_events, gt_events, "ttc")


def aligned_pairs(
    pred_events: Sequence[SsmEvent],
    gt_events: Sequence[SsmEvent],
    field: str = "pet",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred_values, gt_values) for downstream plotting."""
    return _aligned_arrays(pred_events, gt_events, field)
