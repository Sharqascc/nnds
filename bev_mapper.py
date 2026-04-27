import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np


def compute_homography_dlt(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """
    Compute homography matrix using Direct Linear Transform (DLT).

    Args:
        src_points: Nx2 array of source (pixel) points
        dst_points: Nx2 array of destination (world) points

    Returns:
        3x3 homography matrix
    """
    src_points = np.asarray(src_points, dtype=np.float64)
    dst_points = np.asarray(dst_points, dtype=np.float64)

    if src_points.shape != dst_points.shape or src_points.ndim != 2 or src_points.shape[1] != 2:
        raise ValueError("src_points and dst_points must both be Nx2 arrays")

    n = len(src_points)
    if n < 4:
        raise ValueError("At least 4 point correspondences are required")

    A = np.zeros((2 * n, 9), dtype=np.float64)

    for i in range(n):
        x, y = src_points[i]
        u, v = dst_points[i]

        A[2 * i] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]

    _, _, vt = np.linalg.svd(A)
    H = vt[-1].reshape(3, 3)

    if np.isclose(H[2, 2], 0.0):
        raise RuntimeError("Degenerate homography in DLT solution")

    H = H / H[2, 2]
    return H


def _project_points_homography(pixel_points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Project Nx2 points through a 3x3 homography, returning Nx2 points.
    """
    pixel_points = np.asarray(pixel_points, dtype=np.float64)
    ones = np.ones((len(pixel_points), 1), dtype=np.float64)
    pts_h = np.hstack([pixel_points, ones])
    proj_h = (H @ pts_h.T).T
    denom = proj_h[:, 2:3]

    valid = ~np.isclose(denom[:, 0], 0.0)
    out = np.full((len(pixel_points), 2), np.nan, dtype=np.float64)
    out[valid] = proj_h[valid, :2] / denom[valid]
    return out


def test_with_real_calibration(
    calib_json: str = "configs/giti_calibration_points.json",
    bev_json: str = "configs/bev_config.json",
    pixel_error_std: float = 0.5,
):
    """
    Test BEV mapper with actual calibration points.

    Validates homography transformation accuracy by computing reprojection errors
    on known calibration points.

    Returns:
        dict: Validation results including reprojection errors and status
    """
    print("=" * 80)
    print("REAL CALIBRATION VALIDATION TEST")
    print("=" * 80)

    calib_path = Path(calib_json)
    bev_config_path = Path(bev_json)

    if not calib_path.exists():
        print(f"❌ Calibration file not found: {calib_path}")
        return None

    if not bev_config_path.exists():
        print(f"❌ BEV config not found: {bev_config_path}")
        return None

    with open(calib_path, "r") as f:
        calib = json.load(f)

    with open(bev_config_path, "r") as f:
        bev_config = json.load(f)

    pixel_points = np.array(calib.get("pixel_points", []), dtype=np.float64)
    world_points = np.array(calib.get("world_points", []), dtype=np.float64)

    if len(pixel_points) < 4 or len(world_points) < 4:
        print(f"❌ Insufficient calibration points: {len(pixel_points)}")
        print("   Need at least 4 point correspondences")
        return None

    if pixel_points.shape[0] != world_points.shape[0]:
        print("❌ pixel_points and world_points have different lengths")
        return None

    if pixel_points.ndim != 2 or pixel_points.shape[1] != 2:
        print(f"❌ pixel_points must be Nx2, got shape {pixel_points.shape}")
        return None

    if world_points.ndim != 2 or world_points.shape[1] < 2:
        print(f"❌ world_points must be Nx2 or Nx3, got shape {world_points.shape}")
        return None

    world_points_xy = world_points[:, :2]

    print(f"\n✓ Loaded {len(pixel_points)} calibration point pairs")

    try:
        import cv2

        H, mask = cv2.findHomography(
            pixel_points,
            world_points_xy,
            cv2.RANSAC,
            5.0,
        )
        if H is None:
            raise RuntimeError("cv2.findHomography returned None")

        mask = mask.ravel().astype(bool) if mask is not None else np.ones(len(pixel_points), dtype=bool)
        print("✓ Homography computed using RANSAC")
        print(f"  Inliers: {int(mask.sum())}/{len(mask)}")
    except Exception as e:
        print(f"⚠ OpenCV homography failed, falling back to DLT: {e}")
        H = compute_homography_dlt(pixel_points, world_points_xy)
        mask = np.ones(len(pixel_points), dtype=bool)
        print("✓ Homography computed using DLT")

    try:
        from bev_mapper import BEVMapper
    except ImportError as e:
        print(f"❌ Could not import BEVMapper: {e}")
        return None

    bounds = bev_config.get("bounds", bev_config.get("bev_bounds"))
    resolution = bev_config.get("resolution", bev_config.get("bev_resolution"))

    if bounds is None or resolution is None:
        print("❌ BEV config must contain either ('bounds','resolution') or ('bev_bounds','bev_resolution')")
        return None

    mapper = BEVMapper(
        H_pixel_to_world=H,
        bev_bounds=bounds,
        bev_resolution=resolution,
    )

    print(f"\n✓ Mapper initialized:")
    print(f"  {mapper}")

    print("\n" + "─" * 80)
    print("TEST 1: Forward Projection Accuracy (Pixel → World)")
    print("─" * 80)

    world_reproj = _project_points_homography(pixel_points, H)
    valid = np.all(np.isfinite(world_reproj), axis=1)

    errors = []
    for i in range(len(pixel_points)):
        if valid[i]:
            error = np.linalg.norm(world_reproj[i] - world_points_xy[i])
            errors.append(error)
            status = "✓" if error < 0.5 else "⚠" if error < 1.0 else "✗"
            inlier_tag = "inlier" if mask[i] else "outlier"
            print(f"  Point {i:2d}: {status} Error = {error:.4f} m  ({inlier_tag})")

    if len(errors) == 0:
        print("❌ No valid projected points")
        return None

    errors = np.array(errors, dtype=np.float64)
    mean_error = float(np.mean(errors))
    max_error = float(np.max(errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    print("\n  Summary:")
    print(f"    Valid points: {int(valid.sum())}/{len(valid)}")
    print(f"    Mean error:   {mean_error:.4f} m")
    print(f"    Max error:    {max_error:.4f} m")
    print(f"    RMSE:         {rmse:.4f} m")

    if rmse < 0.25:
        quality = "EXCELLENT"
    elif rmse < 0.50:
        quality = "GOOD"
    elif rmse < 1.00:
        quality = "ACCEPTABLE"
    else:
        quality = "POOR - RECALIBRATION RECOMMENDED"

    print(f"    Quality:      {quality}")

    print("\n" + "─" * 80)
    print("TEST 2: Uncertainty Quantification")
    print("─" * 80)

    uncertainties = []
    if hasattr(mapper, "estimate_transformation_error"):
        for i, pixel_pt in enumerate(pixel_points[: min(5, len(pixel_points))]):
            try:
                unc = mapper.estimate_transformation_error(
                    tuple(pixel_pt),
                    pixel_error_std=pixel_error_std,
                )
            except TypeError:
                unc = mapper.estimate_transformation_error(tuple(pixel_pt))

            if unc is not None and np.isfinite(unc):
                uncertainties.append(float(unc))
                print(f"  Point {i}: ±{float(unc):.4f} m  (from ±{pixel_error_std:.1f} px)")
    else:
        print("  ⚠ Mapper has no estimate_transformation_error() method")

    mean_uncertainty = float(np.mean(uncertainties)) if uncertainties else float("nan")
    if uncertainties:
        print(f"\n  Mean position uncertainty: ±{mean_uncertainty:.4f} m")
    else:
        print("\n  Mean position uncertainty: N/A")

    print("\n" + "─" * 80)
    print("TEST 3: Boundary Validation")
    print("─" * 80)

    try:
        x_min = bounds["x_min"]
        y_min = bounds["y_min"]
        x_max = bounds["x_max"]
        y_max = bounds["y_max"]
    except Exception:
        print("  ❌ bounds must be a dict with x_min, y_min, x_max, y_max")
        return None

    test_world_points = np.array(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ],
        dtype=np.float64,
    )

    if hasattr(mapper, "world_to_bev_batch"):
        bev_coords, valid_bev = mapper.world_to_bev_batch(test_world_points)
        print(f"  Corner points within grid: {int(np.sum(valid_bev))}/4")
        for world_pt, bev_pt, is_valid in zip(test_world_points, bev_coords, valid_bev):
            status = "✓" if is_valid else "✗"
            print(f"    {status} World {world_pt} -> BEV {bev_pt}")
    else:
        print("  ⚠ Mapper has no world_to_bev_batch() method")

    print("\n" + "─" * 80)
    print("TEST 4: Inverse Transformation Consistency")
    print("─" * 80)

    roundtrip_mean_px = float("nan")
    try:
        H_inv = np.linalg.inv(H)
        test_pixels = pixel_points[: min(3, len(pixel_points))]
        world_fwd = _project_points_homography(test_pixels, H)
        pixels_back = _project_points_homography(world_fwd, H_inv)

        roundtrip_errors = np.linalg.norm(test_pixels - pixels_back, axis=1)
        print("  Round-trip error (pixel→world→pixel):")
        for i, err in enumerate(roundtrip_errors):
            print(f"    Point {i}: {float(err):.4f} pixels")

        roundtrip_mean_px = float(np.mean(roundtrip_errors))
        print(f"  Mean round-trip error: {roundtrip_mean_px:.4f} px")
    except Exception as e:
        print(f"  ⚠ Inverse mapper test failed: {e}")

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    results = {
        "calibration_points": int(len(pixel_points)),
        "inliers": int(mask.sum()),
        "valid_transforms": int(valid.sum()),
        "mean_reprojection_error_m": mean_error,
        "max_reprojection_error_m": max_error,
        "rmse_m": rmse,
        "quality": quality,
        "mean_uncertainty_m": mean_uncertainty,
        "roundtrip_mean_px": roundtrip_mean_px,
        "passes_validation": bool(rmse < 1.0),
    }

    print(f"  Calibration Quality:    {quality}")
    print(f"  Reprojection RMSE:      {rmse:.4f} m")
    if np.isfinite(mean_uncertainty):
        print(f"  Position Uncertainty:   ±{mean_uncertainty:.4f} m")
    else:
        print("  Position Uncertainty:   N/A")
    if np.isfinite(roundtrip_mean_px):
        print(f"  Round-trip Error:       {roundtrip_mean_px:.4f} px")
    else:
        print("  Round-trip Error:       N/A")

    if results["passes_validation"]:
        print("\n  ✅ CALIBRATION VALIDATED - Ready for production use")
    else:
        print("\n  ⚠ CALIBRATION NEEDS REVIEW - Consider re-calibration")

    print("=" * 80)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate BEV calibration for NNDS pipeline"
    )
    parser.add_argument(
        "--calib",
        default="configs/giti_calibration_points.json",
        help="Path to calibration points JSON",
    )
    parser.add_argument(
        "--bev-config",
        default="configs/bev_config.json",
        help="Path to BEV config JSON",
    )
    parser.add_argument(
        "--pixel-error-std",
        type=float,
        default=0.5,
        help="Assumed pixel localization std-dev for uncertainty analysis",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output and return exit code only",
    )
    parser.add_argument(
        "--raise-on-failure",
        action="store_true",
        help="Raise exception on validation failure",
    )

    args = parser.parse_args()

    if args.quiet:
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            results = test_with_real_calibration(
                calib_json=args.calib,
                bev_json=args.bev_config,
                pixel_error_std=args.pixel_error_std,
            )
    else:
        results = test_with_real_calibration(
            calib_json=args.calib,
            bev_json=args.bev_config,
            pixel_error_std=args.pixel_error_std,
        )

    if results is None:
        sys.exit(1)

    if not results["passes_validation"]:
        if args.raise_on_failure:
            raise RuntimeError(
                f"Calibration validation failed: RMSE={results['rmse_m']:.4f} m"
            )
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
