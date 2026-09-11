"""PET_GT vs PET_NNDS scatter plot. Visualization only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_pet_agreement(
    pred_values: np.ndarray,
    gt_values: np.ndarray,
    out_path: str | Path,
    field_label: str = "PET",
    unit: str = "s",
) -> Path:
    pred = np.asarray(pred_values, dtype=np.float64)
    gt = np.asarray(gt_values, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError("pred_values and gt_values must have the same shape")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    if pred.size > 0:
        ax.scatter(gt, pred, s=18, alpha=0.7, edgecolor="none")
        lo = float(min(pred.min(), gt.min()))
        hi = float(max(pred.max(), gt.max()))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xlabel(f"{field_label} ground truth ({unit})")
    ax.set_ylabel(f"{field_label} NNDS ({unit})")
    ax.set_title(f"{field_label} agreement")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
