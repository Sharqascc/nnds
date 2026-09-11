"""BEV / world-space position error metrics. Pure functions; no I/O.

All distances are in the same unit as the world coordinates. For the NNDS
calibration that unit is meters (easting/northing), so MAE/RMSE/p95 are
meter-space errors.
"""

from __future__ import annotations

import numpy as np


def project_homography(H: np.ndarray, pixel_points: np.ndarray) -> np.ndarray:
    """Apply 3x3 homography H to Nx2 pixel points, return Nx2 world points."""
    pts = np.asarray(pixel_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("pixel_points must be Nx2")
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError("H must be 3x3")
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    proj = (H @ pts_h.T).T
    w = proj[:, 2:3]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("homography produced a point at infinity")
    return proj[:, :2] / w


def position_errors(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-point Euclidean distance between predicted and target world points."""
    p = np.asarray(predicted, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError("predicted and target must have the same shape")
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("inputs must be Nx2")
    return np.linalg.norm(p - t, axis=1)


def position_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    percentile: float = 95.0,
) -> dict[str, float]:
    """Return MAE, RMSE, p95, max, n in world units (meters for NNDS)."""
    errs = position_errors(predicted, target)
    if errs.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "p95": 0.0, "max": 0.0, "n": 0}
    mae = float(np.mean(errs))
    rmse = float(np.sqrt(np.mean(errs**2)))
    p95 = float(np.percentile(errs, percentile))
    return {
        "mae": mae,
        "rmse": rmse,
        "p95": p95,
        "max": float(np.max(errs)),
        "n": int(errs.size),
    }


def homography_error_metrics(
    H: np.ndarray,
    pixel_points: np.ndarray,
    world_points: np.ndarray,
    percentile: float = 95.0,
) -> dict[str, float]:
    """Project pixel_points through H and compare to world_points."""
    projected = project_homography(H, pixel_points)
    return position_metrics(projected, world_points, percentile=percentile)
