#!/usr/bin/env python
"""
monte_carlo_calibration_benchmark.py

Monte Carlo benchmark comparing:
- Homography (biased world plane),
- Homography (Z=0 world),
- PnP ITERATIVE with biased Z,
- PnP ITERATIVE with Z=0,
- P3P + RANSAC (if available),

under:
- known camera intrinsics + distortion,
- synthetic ground grid,
- plane bias in Z,
- anisotropic, correlated pixel noise.

Refinements:
- argparse for configuration (num trials, seed, plotting, multi-noise),
- logging,
- type hints,
- tqdm progress bar,
- JSON export of summary metrics,
- simple comparative analysis helper.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # optional dep
    tqdm = lambda x, **kwargs: x  # type: ignore[misc]

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# CAMERA AND GRID CONFIG
# -------------------------------------------------------------------------
IMG_W, IMG_H = 1920, 1080

fx, fy = 1600.0, 1600.0
cx, cy = IMG_W / 2.0, IMG_H / 2.0
K = np.array(
    [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
)

# Radial distortion only, as in original script
dist_coeffs = np.array([-0.12, 0.02, 0.0, 0.0, 0.0], dtype=np.float32)

# Ground-plane grid (X-Y in meters)
NX, NY = 27, 12
W_X, W_Y = 49.5, 22.0
x_coords = np.linspace(0.0, W_X, NX)
y_coords = np.linspace(0.0, W_Y, NY)
XX, YY = np.meshgrid(x_coords, y_coords)
ZW = np.zeros_like(XX)
world_points_true = np.stack(
    [XX.ravel(), YY.ravel(), ZW.ravel()], axis=1
).astype(np.float32)

# Default noise / bias parameters (can be varied in multi-noise mode)
sigma_px_x_default = 2.0       # pixel noise std in x
sigma_px_y_default = 3.0       # pixel noise std in y
rho_noise_default = 0.83       # correlation between x and y noise
plane_bias_cm_default = 1.5    # max plane bias across X (centimeters)


# -------------------------------------------------------------------------
# CLI / LOGGING
# -------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Monte Carlo calibration benchmark: homography vs PnP vs P3P."
    )
    p.add_argument(
        "--num-trials",
        type=int,
        default=50,
        help="Number of Monte Carlo trials (default: 50).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed (default: 0).",
    )
    p.add_argument(
        "--output-summary",
        type=Path,
        default=Path("calibration/monte_carlo_calibration_summary.json"),
        help="Path to JSON summary output (single scenario or base name for multi-noise).",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Generate and save a basic MAE histogram (PnP ITERATIVE).",
    )
    p.add_argument(
        "--multi-noise",
        action="store_true",
        help="Sweep multiple pixel-noise levels in one run (overrides defaults).",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return p


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


# -------------------------------------------------------------------------
# CORE HELPERS
# -------------------------------------------------------------------------
def make_example_pose() -> Tuple[np.ndarray, np.ndarray]:
    """Example camera pose above ground plane, as in original script."""
    rvec = np.array([0.4, 0.0, 0.0], dtype=np.float32)
    R, _ = cv2.Rodrigues(rvec)
    t = np.array([[0.0], [-15.0], [25.0]], dtype=np.float32)
    return R, t


def project_points(
    world_pts: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K_: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    """Project 3D world points into image using cv2.projectPoints."""
    rvec, _ = cv2.Rodrigues(R)
    img_pts, _ = cv2.projectPoints(world_pts, rvec, t, K_, dist)
    return img_pts.reshape(-1, 2)


R_true, t_true = make_example_pose()


def add_plane_bias(world_pts: np.ndarray, bias_cm: float = 1.5) -> np.ndarray:
    """
    Add a gentle plane bias in Z across X (in meters),
    simulating a slightly tilted ground plane.
    """
    bias_m = bias_cm / 100.0
    wp = world_pts.copy()
    x_range = np.ptp(wp[:, 0])
    x_norm = (wp[:, 0] - wp[:, 0].min()) / max(1e-6, x_range)
    wp[:, 2] = x_norm * bias_m
    return wp


def add_anisotropic_noise_2d(
    points_2d: np.ndarray,
    rng: np.random.Generator,
    sx: float,
    sy: float,
    rho: float,
) -> np.ndarray:
    """
    Add correlated Gaussian noise to 2D points:
    cov = [[sx^2, rho*sx*sy], [rho*sx*sy, sy^2]].
    """
    N = points_2d.shape[0]
    cov = np.array(
        [[sx**2, rho * sx * sy], [rho * sx * sy, sy**2]], dtype=np.float32
    )
    noise = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=N)
    return points_2d + noise.astype(np.float32)


def mae_world(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean absolute Euclidean error in world (meters)."""
    return float(np.mean(np.linalg.norm(pred - gt, axis=1)))


def world_from_pnp(
    R: np.ndarray,
    t: np.ndarray,
    K_: np.ndarray,
    dist: np.ndarray,
    img_pts: np.ndarray,
) -> np.ndarray:
    """
    Back-project image points to Z=0 ground plane using estimated pose.
    Returns Nx2 world coordinates.
    """
    img_pts_undist = cv2.undistortPoints(
        img_pts.reshape(-1, 1, 2), K_, dist
    ).reshape(-1, 2)
    rays_cam = np.concatenate(
        [img_pts_undist, np.ones((img_pts_undist.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    R_inv = R.T
    t_vec = t.reshape(3)
    cam_center_world = -R_inv @ t_vec
    pts_world: List[List[float]] = []

    for d_cam in rays_cam:
        d_world = R_inv @ d_cam
        if abs(d_world[2]) < 1e-6:
            pts_world.append([np.nan, np.nan])
            continue
        s = -cam_center_world[2] / d_world[2]
        P_world = cam_center_world + s * d_world
        pts_world.append(P_world[:2])

    return np.array(pts_world, dtype=np.float32)


def estimate_homography(
    world_pts: np.ndarray,
    img_pts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate homography from image -> world (X,Y) with RANSAC."""
    src = img_pts.astype(np.float32)
    dst = world_pts[:, :2].astype(np.float32)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return H, mask


def apply_homography(H: np.ndarray, img_pts: np.ndarray) -> np.ndarray:
    """Apply homography to 2D image points, return Nx2 world coordinates."""
    pts = img_pts.astype(np.float32)
    pts_h = cv2.convertPointsToHomogeneous(pts).reshape(-1, 3).T
    mapped = H @ pts_h
    mapped = (mapped[:2] / mapped[2]).T
    return mapped


def solve_pnp_world_error(
    world_pts: np.ndarray,
    img_pts: np.ndarray,
    method_flag: int,
) -> Tuple[float, np.ndarray | None, np.ndarray | None]:
    """
    Solve PnP and compute MAE in world (2D ground).
    Returns (mae, R_est, tvec) or (inf, None, None) on failure.
    """
    obj = world_pts.reshape(-1, 1, 3).astype(np.float32)
    img = img_pts.reshape(-1, 1, 2).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist_coeffs, flags=method_flag)
    if not ok:
        return float("inf"), None, None
    R_est, _ = cv2.Rodrigues(rvec)
    world_est = world_from_pnp(R_est, tvec, K, dist_coeffs, img_pts)
    mae = mae_world(world_est, world_points_true[:, :2])
    return mae, R_est, tvec


def solve_pnp_Z0_world_error(
    world_pts: np.ndarray,
    img_pts: np.ndarray,
    method_flag: int,
) -> Tuple[float, np.ndarray | None, np.ndarray | None]:
    """PnP assuming Z=0 for all world points before PnP."""
    wp = world_pts.copy()
    wp[:, 2] = 0.0
    return solve_pnp_world_error(wp, img_pts, method_flag)


def solve_p3p_ransac_world_error(
    world_pts: np.ndarray,
    img_pts: np.ndarray,
    iterations: int = 200,
    sample_size: int = 4,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Simple P3P+RANSAC wrapper: repeatedly sample minimal sets, keep best MAE.
    Returns best MAE or inf if unavailable.
    """
    if not hasattr(cv2, "SOLVEPNP_P3P"):
        return float("inf")
    if rng is None:
        rng = np.random.default_rng()

    obj_all = world_pts.astype(np.float32)
    img_all = img_pts.astype(np.float32)
    N = obj_all.shape[0]
    best_mae = float("inf")

    for _ in range(iterations):
        idx = rng.choice(N, size=sample_size, replace=False)
        obj_sample = obj_all[idx].reshape(-1, 1, 3)
        img_sample = img_all[idx].reshape(-1, 1, 2)
        ok, rvec, tvec = cv2.solvePnP(
            obj_sample, img_sample, K, dist_coeffs, flags=cv2.SOLVEPNP_P3P
        )
        if not ok:
            continue
        R_est, _ = cv2.Rodrigues(rvec)
        world_est = world_from_pnp(R_est, tvec, K, dist_coeffs, img_all)
        mae = mae_world(world_est, world_points_true[:, :2])
        if mae < best_mae:
            best_mae = mae
    return best_mae


# -------------------------------------------------------------------------
# MONTE CARLO DRIVER
# -------------------------------------------------------------------------
def run_single_trial(
    seed: int,
    sigma_px_x: float,
    sigma_px_y: float,
    rho_noise: float,
    plane_bias_cm: float,
) -> Tuple[float, float, float, float, float]:
    """Run a single noisy realization and return MAEs for all methods."""
    rng = np.random.default_rng(seed)

    world_points_biased = add_plane_bias(world_points_true, plane_bias_cm)
    img_points_ideal = project_points(
        world_points_biased, R_true, t_true, K, dist_coeffs
    )
    img_points_noisy = add_anisotropic_noise_2d(
        img_points_ideal, rng, sigma_px_x, sigma_px_y, rho_noise
    )

    # Homography with biased Z world
    H_biased, _ = estimate_homography(world_points_biased, img_points_noisy)
    pred_world_h_biased = apply_homography(H_biased, img_points_noisy)
    mae_h_true_biased = mae_world(
        pred_world_h_biased, world_points_true[:, :2]
    )

    # Homography with Z=0 world
    H_Z0, _ = estimate_homography(world_points_true, img_points_noisy)
    pred_world_h_Z0 = apply_homography(H_Z0, img_points_noisy)
    mae_h_true_Z0 = mae_world(pred_world_h_Z0, world_points_true[:, :2])

    # PnP ITERATIVE with biased Z
    mae_pnp_iter, _, _ = solve_pnp_world_error(
        world_points_biased, img_points_noisy, cv2.SOLVEPNP_ITERATIVE
    )

    # PnP ITERATIVE with Z=0 world
    mae_pnp_iter_Z0, _, _ = solve_pnp_Z0_world_error(
        world_points_biased, img_points_noisy, cv2.SOLVEPNP_ITERATIVE
    )

    # P3P (if available)
    mae_p3p = solve_p3p_ransac_world_error(
        world_points_biased, img_points_noisy, rng=rng
    )

    return (
        mae_h_true_biased,
        mae_h_true_Z0,
        mae_pnp_iter,
        mae_pnp_iter_Z0,
        mae_p3p,
    )


def run_monte_carlo(
    num_trials: int,
    seed: int,
    sigma_px_x: float,
    sigma_px_y: float,
    rho_noise: float,
    plane_bias_cm: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run multiple trials and stack MAEs into arrays (with progress bar)."""
    rng_global = np.random.default_rng(seed)
    mae_H_biased, mae_H_Z0, mae_PNP, mae_PNP_Z0, mae_P3P = [], [], [], [], []

    for _ in tqdm(range(num_trials), desc="Monte Carlo trials", leave=False):
        trial_seed = int(rng_global.integers(1e9))
        h_b, h_z0, p, pz0, p3p = run_single_trial(
            seed=trial_seed,
            sigma_px_x=sigma_px_x,
            sigma_px_y=sigma_px_y,
            rho_noise=rho_noise,
            plane_bias_cm=plane_bias_cm,
        )
        mae_H_biased.append(h_b)
        mae_H_Z0.append(h_z0)
        mae_PNP.append(p)
        mae_PNP_Z0.append(pz0)
        mae_P3P.append(p3p)

    return (
        np.array(mae_H_biased, dtype=np.float32),
        np.array(mae_H_Z0, dtype=np.float32),
        np.array(mae_PNP, dtype=np.float32),
        np.array(mae_PNP_Z0, dtype=np.float32),
        np.array(mae_P3P, dtype=np.float32),
    )


def summarize(name: str, arr: np.ndarray) -> Tuple[float, float]:
    """Log and return mean/std for a metric array."""
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    logger.info("%s: mean = %.4f m, std = %.4f m", name, mean, std)
    return mean, std


def export_summary(
    output_path: Path,
    num_trials: int,
    sigma_px_x: float,
    sigma_px_y: float,
    rho_noise: float,
    plane_bias_cm: float,
    mae_H_biased: np.ndarray,
    mae_H_Z0: np.ndarray,
    mae_PNP: np.ndarray,
    mae_PNP_Z0: np.ndarray,
    mae_P3P: np.ndarray,
) -> Dict:
    """Export key Monte Carlo metrics to JSON and return summary dict."""
    summary: Dict = {
        "num_trials": int(num_trials),
        "camera": {
            "img_w": IMG_W,
            "img_h": IMG_H,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "dist_coeffs": dist_coeffs.tolist(),
        },
        "grid": {
            "NX": int(NX),
            "NY": int(NY),
            "W_X": float(W_X),
            "W_Y": float(W_Y),
        },
        "noise": {
            "sigma_px_x": float(sigma_px_x),
            "sigma_px_y": float(sigma_px_y),
            "rho_noise": float(rho_noise),
            "plane_bias_cm": float(plane_bias_cm),
        },
        "mae_stats": {
            "H_biased_mean": float(np.nanmean(mae_H_biased)),
            "H_biased_std": float(np.nanstd(mae_H_biased)),
            "H_Z0_mean": float(np.nanmean(mae_H_Z0)),
            "H_Z0_std": float(np.nanstd(mae_H_Z0)),
            "PnP_iter_mean": float(np.nanmean(mae_PNP)),
            "PnP_iter_std": float(np.nanstd(mae_PNP)),
            "PnP_iter_Z0_mean": float(np.nanmean(mae_PNP_Z0)),
            "PnP_iter_Z0_std": float(np.nanstd(mae_PNP_Z0)),
            "P3P_mean": float(np.nanmean(mae_P3P)),
            "P3P_std": float(np.nanstd(mae_P3P)),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Exported Monte Carlo summary to %s", output_path)
    return summary


def compare_methods(summary: Dict) -> Dict:
    """
    Simple benchmark comparison: how much better is PnP vs homography (biased)?
    Returns a dict with improvement factor (>1 => PnP worse, <1 => PnP better).
    """
    mae_stats = summary["mae_stats"]
    h_mean = mae_stats["H_biased_mean"]
    pnp_mean = mae_stats["PnP_iter_mean"]
    ratio_pnp_vs_h = pnp_mean / h_mean if h_mean > 0 else float("inf")
    comp = {"pnp_vs_homography_factor": float(ratio_pnp_vs_h)}
    logger.info(
        "PnP vs homography (biased): factor = %.3f ( <1 => PnP better )",
        ratio_pnp_vs_h,
    )
    return comp


def maybe_plot_mae_hist(
    mae_PNP: np.ndarray, output_path: Path = Path("calibration/calibration_errors_pnp.png")
) -> None:
    """Optionally generate a simple histogram of PnP MAE."""
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(mae_PNP, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
    plt.xlabel("MAE (m)")
    plt.ylabel("Frequency")
    plt.title("PnP ITERATIVE world MAE distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Saved MAE histogram to %s", output_path)


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def run_single_scenario(
    args: argparse.Namespace,
    sigma_px_x: float,
    sigma_px_y: float,
    rho_noise: float,
    plane_bias_cm: float,
    suffix: str | None = None,
) -> Dict:
    """Run one noise scenario, log, export summary, and optionally plot."""
    logger.info(
        "Scenario: sigma_px_x=%.2f, sigma_px_y=%.2f, rho=%.2f, plane_bias_cm=%.2f",
        sigma_px_x,
        sigma_px_y,
        rho_noise,
        plane_bias_cm,
    )
    (
        mae_H_biased,
        mae_H_Z0,
        mae_PNP,
        mae_PNP_Z0,
        mae_P3P,
    ) = run_monte_carlo(
        num_trials=args.num_trials,
        seed=args.seed,
        sigma_px_x=sigma_px_x,
        sigma_px_y=sigma_px_y,
        rho_noise=rho_noise,
        plane_bias_cm=plane_bias_cm,
    )

    summarize("Homography (biased world)", mae_H_biased)
    summarize("Homography (Z=0 world)", mae_H_Z0)
    summarize("PnP ITERATIVE (biased Z)", mae_PNP)
    summarize("PnP ITERATIVE (Z=0)", mae_PNP_Z0)
    if np.isfinite(mae_P3P).any():
        summarize("P3P (RANSAC wrapper)", mae_P3P)
    else:
        logger.info("P3P (RANSAC wrapper): not available / all inf")

    # Decide output path (add suffix in multi-noise mode)
    if suffix:
        stem = args.output_summary.with_suffix("").name
        out_path = args.output_summary.with_name(f"{stem}_{suffix}.json")
    else:
        out_path = args.output_summary

    summary = export_summary(
        out_path,
        args.num_trials,
        sigma_px_x,
        sigma_px_y,
        rho_noise,
        plane_bias_cm,
        mae_H_biased,
        mae_H_Z0,
        mae_PNP,
        mae_PNP_Z0,
        mae_P3P,
    )

    comp = compare_methods(summary)
    summary["comparison"] = comp

    if args.plot and not suffix:
        # Only plot once in single scenario mode
        maybe_plot_mae_hist(mae_PNP)

    return summary


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    if args.multi_noise:
        logger.info(
            "Running multi-noise Monte Carlo: %d trials per scenario.", args.num_trials
        )
        # Example: sweep a few sigma_x/y levels (y slightly larger as in default)
        noise_levels = [0.5, 1.0, 2.0, 3.0]
        all_summaries: Dict[str, Dict] = {}
        for nl in noise_levels:
            sigma_x = nl
            sigma_y = nl * (sigma_px_y_default / sigma_px_x_default)
            suffix = f"sigx{sigma_x:.1f}_sigy{sigma_y:.1f}".replace(".", "p")
            summary = run_single_scenario(
                args,
                sigma_px_x=sigma_x,
                sigma_px_y=sigma_y,
                rho_noise=rho_noise_default,
                plane_bias_cm=plane_bias_cm_default,
                suffix=suffix,
            )
            all_summaries[suffix] = summary

        # Optionally, write a small index file consolidating scenario names
        index_path = args.output_summary.with_suffix("").with_name(
            args.output_summary.with_suffix("").name + "_index.json"
        )
        index_data = {
            "scenarios": list(all_summaries.keys()),
            "base_output": str(args.output_summary),
        }
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with index_path.open("w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)
        logger.info("Wrote multi-noise index to %s", index_path)

    else:
        logger.info(
            "=== Monte Carlo results over %d trials (seed=%d) ===",
            args.num_trials,
            args.seed,
        )
        run_single_scenario(
            args,
            sigma_px_x=sigma_px_x_default,
            sigma_px_y=sigma_px_y_default,
            rho_noise=rho_noise_default,
            plane_bias_cm=plane_bias_cm_default,
            suffix=None,
        )


if __name__ == "__main__":
    main()
