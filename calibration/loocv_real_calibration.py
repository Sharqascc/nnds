#!/usr/bin/env python
"""
loocv_real_calibration.py

Leave-One-Out Cross-Validation (LOOCV) for the real 6-point GITI calibration.

Unlike self-residual error (fit-and-test on the same points), LOOCV holds out
one point at a time, fits the homography on the remaining N-1 points, and
measures reprojection error on the held-out point. This gives an honest
estimate of generalization error appropriate for very small calibration sets.

With only 6 points and 8 homography DOF, self-residual error is expected to
be near-zero regardless of true accuracy (the model has enough freedom to
nearly interpolate through all fitting points). LOOCV avoids this by never
testing a point on a model that was fit using that point.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def load_calibration_points(json_path: Path) -> Tuple[np.ndarray, np.ndarray, list]:
    with json_path.open() as f:
        data = json.load(f)

    points = data["calibration_points"]
    pixel_pts = np.array([[p["pixel"]["x"], p["pixel"]["y"]] for p in points], dtype=np.float64)
    world_pts = np.array([[p["world"]["easting"], p["world"]["northing"]] for p in points], dtype=np.float64)
    ids = [p.get("id", i) for i, p in enumerate(points)]
    return pixel_pts, world_pts, ids


def fit_homography(pixel_pts: np.ndarray, world_pts: np.ndarray) -> np.ndarray:
    """Fit homography with plain least-squares (no RANSAC) since with N-1=5
    points and 8 DOF there are no spare points for RANSAC to reject outliers
    from -- every point must be used for the fit to be well-determined."""
    H, _ = cv2.findHomography(pixel_pts, world_pts, method=0)
    if H is None:
        raise RuntimeError("Homography fit failed (degenerate point configuration)")
    return H


def reproject(pixel_pt: np.ndarray, H: np.ndarray) -> np.ndarray:
    pt_h = np.array([pixel_pt[0], pixel_pt[1], 1.0])
    proj_h = H @ pt_h
    if abs(proj_h[2]) < 1e-12:
        return np.array([np.nan, np.nan])
    return proj_h[:2] / proj_h[2]


def run_loocv(pixel_pts: np.ndarray, world_pts: np.ndarray, ids: list) -> dict:
    n = len(pixel_pts)
    if n < 5:
        raise ValueError(
            f"LOOCV with homography (8 DOF) needs at least 5 points to leave "
            f"one out and still have 4 remaining for a well-posed fit; got {n}."
        )

    errors = []
    per_point = []

    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        pixel_train = pixel_pts[train_idx]
        world_train = world_pts[train_idx]

        pixel_test = pixel_pts[i]
        world_test = world_pts[i]

        try:
            H = fit_homography(pixel_train, world_train)
            world_pred = reproject(pixel_test, H)
            error = float(np.linalg.norm(world_pred - world_test))
        except Exception as e:
            logger.warning("LOOCV fold %d (held-out id=%s) failed: %s", i, ids[i], e)
            error = float("nan")
            world_pred = np.array([np.nan, np.nan])

        errors.append(error)
        per_point.append({
            "held_out_id": ids[i],
            "pixel": pixel_test.tolist(),
            "world_true": world_test.tolist(),
            "world_predicted": world_pred.tolist(),
            "error_m": error,
        })

        status = "OK" if np.isfinite(error) else "FAILED"
        logger.info(
            "Fold %d (held-out point id=%s): error=%.4f m [%s]",
            i, ids[i], error if np.isfinite(error) else -1, status
        )

    errors_arr = np.array(errors, dtype=np.float64)
    valid = np.isfinite(errors_arr)

    if valid.sum() == 0:
        raise RuntimeError("All LOOCV folds failed -- calibration points may be degenerate.")

    results = {
        "n_points": n,
        "n_folds_valid": int(valid.sum()),
        "n_folds_failed": int((~valid).sum()),
        "loocv_mean_error_m": float(np.mean(errors_arr[valid])),
        "loocv_median_error_m": float(np.median(errors_arr[valid])),
        "loocv_max_error_m": float(np.max(errors_arr[valid])),
        "loocv_std_error_m": float(np.std(errors_arr[valid])),
        "loocv_rmse_m": float(np.sqrt(np.mean(errors_arr[valid] ** 2))),
        "per_point": per_point,
    }
    return results


def compute_self_residual_for_comparison(pixel_pts: np.ndarray, world_pts: np.ndarray) -> dict:
    """Fit-and-test on ALL points (the misleading metric) for direct comparison."""
    H, mask = cv2.findHomography(pixel_pts, world_pts, cv2.RANSAC, 5.0)
    proj = np.array([reproject(p, H) for p in pixel_pts])
    errors = np.linalg.norm(proj - world_pts, axis=1)
    return {
        "self_residual_mean_error_m": float(np.mean(errors)),
        "self_residual_rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "inliers": int(mask.sum()),
        "total": int(len(mask)),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    project_root = Path(__file__).resolve().parents[1]
    calib_path = project_root / "configs" / "giti_calibration_points.json"

    logger.info("=" * 70)
    logger.info("LOOCV VALIDATION -- REAL 6-POINT GITI CALIBRATION")
    logger.info("=" * 70)
    logger.info("Loading calibration points from %s", calib_path)

    pixel_pts, world_pts, ids = load_calibration_points(calib_path)
    logger.info("Loaded %d calibration points", len(pixel_pts))

    logger.info("")
    logger.info("Running LOOCV (fit on N-1, test on held-out point)...")
    loocv_results = run_loocv(pixel_pts, world_pts, ids)

    logger.info("")
    logger.info("For comparison, computing self-residual error (fit-and-test on same points)...")
    self_residual = compute_self_residual_for_comparison(pixel_pts, world_pts)

    print()
    print("=" * 70)
    print("COMPARISON: SELF-RESIDUAL (misleading) vs LOOCV (honest)")
    print("=" * 70)
    print(f"{'Metric':<30} | {'Self-residual':<15} | {'LOOCV':<15}")
    print("-" * 70)
    print(f"{'Mean error (m)':<30} | {self_residual['self_residual_mean_error_m']:<15.4f} | {loocv_results['loocv_mean_error_m']:<15.4f}")
    print(f"{'RMSE (m)':<30} | {self_residual['self_residual_rmse_m']:<15.4f} | {loocv_results['loocv_rmse_m']:<15.4f}")
    print(f"{'Max error (m)':<30} | {'N/A':<15} | {loocv_results['loocv_max_error_m']:<15.4f}")
    print()
    print("Per-point LOOCV breakdown:")
    for p in loocv_results["per_point"]:
        print(f"  id={p['held_out_id']}: error = {p['error_m']:.4f} m")
    print()

    inflation_factor = (
        loocv_results['loocv_rmse_m'] / self_residual['self_residual_rmse_m']
        if self_residual['self_residual_rmse_m'] > 0 else float('inf')
    )
    print(f"LOOCV RMSE is {inflation_factor:.1f}x the self-residual RMSE.")
    if inflation_factor > 3:
        print("=> Self-residual substantially UNDERSTATES true calibration error.")
        print("   Report LOOCV (or better, held-out) error in the paper, not self-residual.")
    print()

    output = {
        "self_residual": self_residual,
        "loocv": loocv_results,
        "inflation_factor": inflation_factor,
    }
    output_path = project_root / "calibration" / "loocv_real_calibration_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved results to %s", output_path)


if __name__ == "__main__":
    main()
