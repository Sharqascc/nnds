from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ValidationMetrics:
    mean_error: float
    max_error: float
    rmse: float
    inlier_ratio: float | None = None
    num_samples: int | None = None


def compute_error_metrics(errors: Sequence[float] | np.ndarray) -> ValidationMetrics:
    arr = np.asarray(errors, dtype=float)
    if arr.size == 0:
        raise ValueError("errors must not be empty")
    return ValidationMetrics(
        mean_error=float(np.mean(arr)),
        max_error=float(np.max(arr)),
        rmse=float(np.sqrt(np.mean(arr**2))),
        num_samples=int(arr.size),
    )


def validate_numeric_array(
    name: str, value: Any, ndim: int | None = None
) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {arr.ndim}")
    return arr


def validate_bev_result(result: dict[str, Any]) -> None:
    required = {
        "pointerrors",
        "meanerrorall",
        "meanerrorinliers",
        "stderrorall",
        "maxerror",
        "rmse",
    }
    missing = required - set(result)
    if missing:
        raise ValueError(f"missing validation keys: {sorted(missing)}")
    validate_numeric_array(
        "pointerrors", [r["error"] for r in result["pointerrors"]], ndim=1
    )
