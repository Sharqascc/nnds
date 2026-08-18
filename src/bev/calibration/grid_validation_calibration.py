#!/usr/bin/env python
"""
grid_validation_calibration.py

Simulation-based validation of a grid-derived homography compared
to an original 6-point calibration, with proper train/val split
and K-fold cross-validation.

Refinements:
- argparse configuration (no hard-coded project_root),
- logging with optional verbose mode,
- type hints,
- tqdm progress bar for large grids,
- JSON export of summary metrics,
- --real-data flag stub for future integration.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

try:
    from tqdm import tqdm
except ImportError:  # optional dependency
    tqdm = lambda x, **kwargs: x  # type: ignore[misc]

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grid-based homography validation with simulation and CV."
    )
    p.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Repo root containing configs/ and calibration/ (default: .)",
    )
    p.add_argument("--grid-cols", type=int, default=27, help="Grid columns.")
    p.add_argument("--grid-rows", type=int, default=12, help="Grid rows.")
    p.add_argument(
        "--activa-length-m",
        type=float,
        default=1.833,
        help="Grid spacing (meters) used as physical reference.",
    )
    p.add_argument(
        "--noise-std-m",
        type=float,
        default=0.025,
        help="Simulated survey noise (meters).",
    )
    p.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of KFold splits for cross-validation.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    p.add_argument(
        "--ransac-thresh",
        type=float,
        default=2.0,
        help="RANSAC reprojection threshold (pixels) for cv2.findHomography.",
    )
    p.add_argument(
        "--save-prefix",
        type=str,
        default="calibration_validation_grid_vs_original",
        help="Prefix for saved PNG under project_root.",
    )
    p.add_argument(
        "--real-data",
        action="store_true",
        help="Use real calibration points instead of synthetic grid (stub for future).",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return p.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def load_grid_config(grid_cfg_path: Path) -> Tuple[float, float, float, float]:
    with grid_cfg_path.open() as f:
        grid_config = json.load(f)
    corners = grid_config["corners"]
    x_min_px = corners["top_left"][0]
    x_max_px = corners["top_right"][0]
    y_min_px = corners["top_left"][1]
    y_max_px = corners["bottom_left"][1]
    return x_min_px, x_max_px, y_min_px, y_max_px


def reprojection_errors(
    pixel_points: np.ndarray,
    world_points_target: np.ndarray,
    H: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    projected = cv2.perspectiveTransform(
        pixel_points.reshape(-1, 1, 2), H
    ).reshape(-1, 2)
    errs = np.linalg.norm(projected - world_points_target[:, :2], axis=1)
    return errs, projected


def generate_synthetic_grid(
    grid_rows: int,
    grid_cols: int,
    x_min_px: float,
    x_max_px: float,
    y_min_px: float,
    y_max_px: float,
    pixels_per_meter_x: float,
    pixels_per_meter_y: float,
    noise_std: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pixel_pts: List[List[float]] = []
    world_pts_noisy: List[List[float]] = []
    world_pts_true: List[List[float]] = []

    x_range_px = x_max_px - x_min_px
    y_range_px = y_max_px - y_min_px

    for row_idx in tqdm(
        range(grid_rows), desc="Generating synthetic grid", leave=False
    ):
        for col_idx in range(grid_cols):
            px = (
                x_min_px + (col_idx / (grid_cols - 1)) * x_range_px
                if grid_cols > 1
                else x_min_px
            )
            py = (
                y_min_px + (row_idx / (grid_rows - 1)) * y_range_px
                if grid_rows > 1
                else y_min_px
            )

            world_offset_x = (px - x_min_px) / pixels_per_meter_x
            world_offset_y = (py - y_min_px) / pixels_per_meter_y

            world_pts_true.append([world_offset_x, world_offset_y])

            noise_x = rng.normal(0, noise_std)
            noise_y = rng.normal(0, noise_std)
            wx_measured = world_offset_x + noise_x
            wy_measured = world_offset_y + noise_y

            pixel_pts.append([px, py])
            world_pts_noisy.append([wx_measured, wy_measured])

    return (
        np.array(pixel_pts, dtype=np.float32),
        np.array(world_pts_noisy, dtype=np.float32),
        np.array(world_pts_true, dtype=np.float32),
    )


def load_original_calibration(
    orig_calib_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    with orig_calib_path.open() as f:
        orig_calib_data = json.load(f)

    orig_pixel_pts: List[List[float]] = []
    orig_world_pts: List[List[float]] = []

    for p in orig_calib_data["calibration_points"]:
        orig_pixel_pts.append([p["pixel"]["x"], p["pixel"]["y"]])
        orig_world_pts.append([p["world"]["easting"], p["world"]["northing"]])

    return (
        np.array(orig_pixel_pts, dtype=np.float32),
        np.array(orig_world_pts, dtype=np.float32),
    )


def export_results(results: Dict, output_path: Path) -> None:
    """Export validation results to JSON for downstream analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Exported validation summary to %s", output_path)


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    rng = np.random.default_rng(args.seed)
    project_root: Path = args.project_root

    ACTIVA_LENGTH_METERS = args.activa_length_m
    GRID_COLS = args.grid_cols
    GRID_ROWS = args.grid_rows
    NOISE_STD_DEV = args.noise_std_m
    N_SPLITS = args.n_splits
    RANSAC_THRESH = args.ransac_thresh

    logger.info("=" * 70)
    logger.info("GRID-BASED CALIBRATION WITH PROPER VALIDATION")
    logger.info("=" * 70)

    EXPECTED_WORLD_WIDTH = GRID_COLS * ACTIVA_LENGTH_METERS
    EXPECTED_WORLD_HEIGHT = GRID_ROWS * ACTIVA_LENGTH_METERS

    logger.info("PHYSICAL PARAMETERS")
    logger.info("  Activa length: %.3f m", ACTIVA_LENGTH_METERS)
    logger.info("  Grid: %d x %d", GRID_COLS, GRID_ROWS)
    logger.info(
        "  Expected coverage: %.1f m x %.1f m",
        EXPECTED_WORLD_WIDTH,
        EXPECTED_WORLD_HEIGHT,
    )
    logger.info(
        "  Simulated measurement noise (sigma): %.1f cm", NOISE_STD_DEV * 100.0
    )

    # ------------------------------------------------------------------
    # LOAD GRID CONFIG (IMAGE SPACE)
    # ------------------------------------------------------------------
    grid_cfg_path = project_root / "configs" / "GITI_grid_config.json"
    if not grid_cfg_path.exists():
        raise FileNotFoundError(f"Grid config not found: {grid_cfg_path}")

    logger.info("Loading grid config from %s", grid_cfg_path)
    x_min_px, x_max_px, y_min_px, y_max_px = load_grid_config(grid_cfg_path)
    x_range_px = x_max_px - x_min_px
    y_range_px = y_max_px - y_min_px

    logger.info("Image ranges:")
    logger.info("  X: %d to %d = %d px", x_min_px, x_max_px, x_range_px)
    logger.info("  Y: %d to %d = %d px", y_min_px, y_max_px, y_range_px)

    pixels_per_meter_x = x_range_px / EXPECTED_WORLD_WIDTH
    pixels_per_meter_y = y_range_px / EXPECTED_WORLD_HEIGHT
    pixels_per_meter = 0.5 * (pixels_per_meter_x + pixels_per_meter_y)

    logger.info("Pixels-per-meter estimate:")
    logger.info("  X: %.2f px/m", pixels_per_meter_x)
    logger.info("  Y: %.2f px/m", pixels_per_meter_y)
    logger.info("  Avg: %.2f px/m", pixels_per_meter)

    # ------------------------------------------------------------------
    # GENERATE OR LOAD CALIBRATION POINTS
    # ------------------------------------------------------------------
    if args.real_data:
        # Placeholder for future real-data integration
        # Example: pixel_pts, world_pts_noisy, world_pts_true = load_real_calibration_points(...)
        raise NotImplementedError(
            "--real-data is a stub; integrate real GCP loading here."
        )
    else:
        pixel_pts, world_pts_noisy, world_pts_true = generate_synthetic_grid(
            GRID_ROWS,
            GRID_COLS,
            x_min_px,
            x_max_px,
            y_min_px,
            y_max_px,
            pixels_per_meter_x,
            pixels_per_meter_y,
            NOISE_STD_DEV,
            rng,
        )
    N_POINTS = len(pixel_pts)
    logger.info("Generated %d pixel–world correspondences.", N_POINTS)

    # ------------------------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # ------------------------------------------------------------------
    VAL_RATIO = 0.15
    val_size = int(N_POINTS * VAL_RATIO)
    train_size = N_POINTS - val_size

    indices = np.arange(N_POINTS)
    rng.shuffle(indices)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    pixel_train = pixel_pts[train_idx]
    world_train_noisy = world_pts_noisy[train_idx]

    pixel_val = pixel_pts[val_idx]
    world_val_true = world_pts_true[val_idx]

    logger.info("Train / validation split:")
    logger.info("  Train points: %d", len(pixel_train))
    logger.info("  Val points:   %d", len(pixel_val))

    # ------------------------------------------------------------------
    # ESTIMATE HOMOGRAPHY ON TRAIN SET (RANSAC)
    # ------------------------------------------------------------------
    logger.info(
        "Estimating homography on train set (RANSAC, thresh=%.2f px)...", RANSAC_THRESH
    )
    H_grid, mask_train = cv2.findHomography(
        pixel_train,
        world_train_noisy[:, :2],
        cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESH,
        confidence=0.99,
        maxIters=5000,
    )
    if H_grid is None:
        raise RuntimeError("cv2.findHomography failed on train set.")

    train_inliers = mask_train.ravel().astype(bool)
    train_inlier_ratio = float(np.mean(train_inliers))
    logger.info(
        "Train inliers: %d / %d (%.1f%%)",
        int(train_inliers.sum()),
        len(train_inliers),
        train_inlier_ratio * 100.0,
    )

    # ------------------------------------------------------------------
    # EVALUATE ON TRAIN (NOISY WORLD) VS VALIDATION (TRUE WORLD)
    # ------------------------------------------------------------------
    train_errors, _ = reprojection_errors(pixel_train, world_train_noisy, H_grid)
    train_mae = float(np.mean(train_errors[train_inliers]))
    train_std = float(np.std(train_errors[train_inliers]))

    val_errors, _ = reprojection_errors(pixel_val, world_val_true, H_grid)
    val_mae = float(np.mean(val_errors))
    val_std = float(np.std(val_errors))

    logger.info("Error metrics:")
    logger.info(
        "  TRAIN (noisy survey): MAE=%.4f m (%.2f cm), Std=%.4f m (%.2f cm)",
        train_mae,
        train_mae * 100.0,
        train_std,
        train_std * 100.0,
    )
    logger.info(
        "  VALIDATION (TRUE world): MAE=%.4f m (%.2f cm), Std=%.4f m (%.2f cm)",
        val_mae,
        val_mae * 100.0,
        val_std,
        val_std * 100.0,
    )

    # ------------------------------------------------------------------
    # K-FOLD CROSS-VALIDATION (AGAINST TRUE WORLD)
    # ------------------------------------------------------------------
    logger.info("K-fold cross-validation (n_splits=%d) against TRUE world.", N_SPLITS)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=args.seed)

    cv_maes: List[float] = []
    cv_stds: List[float] = []

    for fold, (train_idx_cv, val_idx_cv) in enumerate(kf.split(pixel_pts), start=1):
        pix_train_cv = pixel_pts[train_idx_cv]
        world_train_cv_noisy = world_pts_noisy[train_idx_cv]
        pix_val_cv = pixel_pts[val_idx_cv]
        world_val_cv_true = world_pts_true[val_idx_cv]

        H_cv, mask_cv = cv2.findHomography(
            pix_train_cv,
            world_train_cv_noisy[:, :2],
            cv2.RANSAC,
            ransacReprojThreshold=RANSAC_THRESH,
            confidence=0.99,
            maxIters=5000,
        )
        if H_cv is None:
            raise RuntimeError("cv2.findHomography failed during CV.")

        val_errs_cv, _ = reprojection_errors(pix_val_cv, world_val_cv_true, H_cv)
        mae_cv = float(np.mean(val_errs_cv))
        std_cv = float(np.std(val_errs_cv))
        cv_maes.append(mae_cv)
        cv_stds.append(std_cv)

        logger.debug(
            "  Fold %d: MAE=%.4f m (%.2f cm), Std=%.4f m (%.2f cm)",
            fold,
            mae_cv,
            mae_cv * 100.0,
            std_cv,
            std_cv * 100.0,
        )

    cv_mae_mean = float(np.mean(cv_maes))
    cv_mae_std = float(np.std(cv_maes))
    logger.info("Cross-validation summary:")
    logger.info(
        "  Mean MAE: %.4f m (%.2f cm), MAE Std: %.4f m (%.2f cm)",
        cv_mae_mean,
        cv_mae_mean * 100.0,
        cv_mae_std,
        cv_mae_std * 100.0,
    )

    # ------------------------------------------------------------------
    # LOAD ORIGINAL 6-POINT CALIBRATION
    # ------------------------------------------------------------------
    orig_calib_path = project_root / "configs" / "giti_calibration_points.json"
    if not orig_calib_path.exists():
        raise FileNotFoundError(f"Original calibration not found: {orig_calib_path}")

    logger.info("Loading original 6-point calibration from %s", orig_calib_path)
    orig_pixel_pts, orig_world_pts = load_original_calibration(orig_calib_path)

    H_orig, mask_orig = cv2.findHomography(
        orig_pixel_pts,
        orig_world_pts[:, :2],
        cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESH,
    )
    if H_orig is None:
        raise RuntimeError("cv2.findHomography failed for original calibration.")

    orig_projected = cv2.perspectiveTransform(
        orig_pixel_pts.reshape(-1, 1, 2),
        H_orig,
    ).reshape(-1, 2)

    orig_errors = np.linalg.norm(orig_projected - orig_world_pts[:, :2], axis=1)
    orig_inliers = mask_orig.ravel().astype(bool)
    mae_orig = float(np.mean(orig_errors[orig_inliers]))
    std_orig = float(np.std(orig_errors[orig_inliers]))

    orig_area = float(
        (orig_world_pts[:, 0].max() - orig_world_pts[:, 0].min())
        * (orig_world_pts[:, 1].max() - orig_world_pts[:, 1].min())
    )
    grid_area = float(EXPECTED_WORLD_WIDTH * EXPECTED_WORLD_HEIGHT)

    logger.info("Original calibration:")
    logger.info("  Points: %d", len(orig_pixel_pts))
    logger.info(
        "  MAE (self-residual): %.4f m (%.2f cm)", mae_orig, mae_orig * 100.0
    )
    logger.info("  Area: %.1f m^2 vs Grid area: %.1f m^2", orig_area, grid_area)

    # ------------------------------------------------------------------
    # JSON EXPORT OF SUMMARY METRICS
    # ------------------------------------------------------------------
    results = {
        "grid_params": {
            "rows": GRID_ROWS,
            "cols": GRID_COLS,
            "spacing_m": ACTIVA_LENGTH_METERS,
            "noise_std_m": NOISE_STD_DEV,
        },
        "train_mae_m": train_mae,
        "train_std_m": train_std,
        "val_mae_m": val_mae,
        "val_std_m": val_std,
        "cv_mae_mean_m": cv_mae_mean,
        "cv_mae_std_m": cv_mae_std,
        "original_mae_m": mae_orig,
        "original_std_m": std_orig,
        "orig_area_m2": orig_area,
        "grid_area_m2": grid_area,
    }
    export_results(results, project_root / "calibration" / "grid_validation_summary.json")

    # ------------------------------------------------------------------
    # TEXTUAL COMPARISON TABLE
    # ------------------------------------------------------------------
    print()  # keep table readable in stdout
    print("=" * 70)
    print("COMPARISON (SIMULATION-BASED GRID VS ORIGINAL 6-POINT)")
    print("=" * 70)
    print()
    print("Metric                | Original (6 pts)     | Grid (simulated)")
    print("-------------------------------------------------------------------")
    print(f"Points used           | {len(orig_pixel_pts):<20d} | {int(N_POINTS):<20d}")
    print(f"Area covered (m^2)    | {orig_area:<20.1f} | {grid_area:<20.1f}")
    print(f"MAE (self-residual)   | {mae_orig:<20.4f} | {train_mae:<20.4f}")
    print(f"Std (self-residual)   | {std_orig:<20.4f} | {train_std:<20.4f}")
    print(f"Val MAE (TRUE world)  | {'N/A':<20}       | {val_mae:<20.4f}")
    print(f"CV MAE mean (TRUE)    | {'N/A':<20}       | {cv_mae_mean:<20.4f}")
    print()
    print("NOTE:")
    print("  - 'Self-residual' = error on the same points used for fitting (optimistic).")
    print("  - 'Val MAE / CV MAE' = error on held-out points w.r.t TRUE world coordinates.")

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------
    logger.info("Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1) Original 6-point world layout
    ax = axes[0, 0]
    ax.scatter(
        orig_world_pts[:, 0],
        orig_world_pts[:, 1],
        c="red",
        s=200,
        marker="o",
        alpha=0.7,
        edgecolors="darkred",
        linewidth=2,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Original (6 pts): MAE={mae_orig:.3f} m", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # 2) Grid true world layout (color = val error if it was in val set)
    ax = axes[0, 1]
    colors = np.zeros(N_POINTS)
    colors[val_idx] = np.interp(
        val_errors, (val_errors.min(), val_errors.max()), (0.2, 1.0)
    )
    scatter = ax.scatter(
        world_pts_true[:, 0],
        world_pts_true[:, 1],
        c=colors,
        s=40,
        cmap="Reds",
        alpha=0.8,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        "Grid TRUE world points\n(val pts darker = higher error)",
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Relative validation error (scaled)", fontsize=10)

    # 3) Validation error histogram (TRUE world)
    ax = axes[1, 0]
    ax.hist(
        val_errors,
        bins=30,
        color="blue",
        alpha=0.7,
        edgecolor="black",
    )
    ax.axvline(
        val_mae, color="red", linestyle="--", linewidth=2, label=f"MAE={val_mae:.3f} m"
    )
    ax.set_xlabel("Validation reprojection error (m)")
    ax.set_ylabel("Frequency")
    ax.set_title("Validation error distribution (TRUE world)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4) Cross-validation MAE per fold
    ax = axes[1, 1]
    fold_ids = np.arange(1, N_SPLITS + 1)
    ax.bar(
        fold_ids,
        np.array(cv_maes) * 100.0,
        color="green",
        alpha=0.7,
        edgecolor="black",
    )
    for i, mae_cv in enumerate(cv_maes, start=1):
        ax.text(
            i,
            mae_cv * 100.0 + 0.05,
            f"{mae_cv*100:.2f} cm",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xlabel("Fold")
    ax.set_ylabel("MAE (cm)")
    ax.set_title("K-fold CV (MAE on TRUE world)", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = project_root / f"{args.save_prefix}.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()

    # ------------------------------------------------------------------
    # SAVE HOMOGRAPHY FOR PIPELINE USE
    # ------------------------------------------------------------------
    H_path = project_root / "calibration" / "H_grid_simulation_validated.npy"
    H_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(H_path, H_grid)

    logger.info("Saved artifacts:")
    logger.info("  Homography (simulation-based, validated): %s", H_path)
    logger.info("  Visualization: %s", fig_path)
    logger.info(
        "IMPORTANT: This script validates the GRID METHOD under a SIMULATED "
        "measurement model.\nFor real deployment, replace synthetic world_pts_noisy "
        "with real surveyed GCPs and keep the same train/val + CV evaluation logic."
    )
    logger.info("✅ PIPELINE-READY (SIMULATION LEVEL) – NEXT STEP: REAL GCP INTEGRATION")


if __name__ == "__main__":
    main()
