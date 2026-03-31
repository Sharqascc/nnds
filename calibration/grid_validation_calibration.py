# grid_validation_calibration.py

import json
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

# =============================================================================
# SCIENTIFIC NOTE:
# -----------------------------------------------------------------------------
# This script validates a grid-based homography under an *ideal* pinhole + planar
# assumption with Gaussian survey noise. It does NOT explicitly model:
#   - Lens radial/tangential distortion (k1, k2, p1, p2, ...),
#   - Strong non-planarity of the scene,
#   - Anisotropic or correlated survey biases beyond the simple Gaussian model.
#
# For real deployment on traffic cameras, a recommended architecture is:
#   1) Calibrate camera intrinsics and distortion using cv2.calibrateCamera()
#      on a checkerboard / calibration grid (estimate K, distortion coeffs).
#   2) Undistort pixel coordinates of detections with cv2.undistortPoints().
#   3) Fit the ground-plane homography H on UNDISTORTED points only.
#   4) Validate against independent surveyed checkpoints not used in fitting.
#
# The separate script:
#   calibration/grid_validation_calibration_realistic_test.py
# simulates lens distortion, plane bias, and anisotropic survey noise to study
# how a pure homography degrades under realistic artefacts. It is intended as a
# stress-test and not as the primary calibration pipeline.
# =============================================================================


print("="*70)
print("GRID-BASED CALIBRATION WITH PROPER VALIDATION")
print("="*70)

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
project_root = "/content/drive/MyDrive/shared_pipeline/4D_tracking_project"

ACTIVA_LENGTH_METERS = 1.833
GRID_COLS = 27
GRID_ROWS = 12
EXPECTED_WORLD_WIDTH = GRID_COLS * ACTIVA_LENGTH_METERS
EXPECTED_WORLD_HEIGHT = GRID_ROWS * ACTIVA_LENGTH_METERS

# Simulated measurement noise (STD in meters)
NOISE_STD_DEV = 0.025

# Cross‑validation
N_SPLITS = 5
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("\nPHYSICAL PARAMETERS")
print("-------------------")
print(f"  Activa length: {ACTIVA_LENGTH_METERS} m")
print(f"  Grid: {GRID_COLS} x {GRID_ROWS}")
print(f"  Expected coverage: {EXPECTED_WORLD_WIDTH:.1f} m x {EXPECTED_WORLD_HEIGHT:.1f} m")
print(f"  Expected area: {EXPECTED_WORLD_WIDTH * EXPECTED_WORLD_HEIGHT:.0f} m²")
print(f"  Simulated measurement noise: σ={NOISE_STD_DEV*100:.1f} cm")

# -------------------------------------------------------------------
# LOAD GRID CORNERS (IMAGE SPACE)
# -------------------------------------------------------------------
print("\nLOADING GRID CONFIG (IMAGE CORNERS)")
print("-----------------------------------")
grid_cfg_path = os.path.join(project_root, "configs/GITI_grid_config.json")
with open(grid_cfg_path) as f:
    grid_config = json.load(f)

corners = grid_config["corners"]
x_min_px = corners["top_left"][0]
x_max_px = corners["top_right"][0]
y_min_px = corners["top_left"][1]
y_max_px = corners["bottom_left"][1]
x_range_px = x_max_px - x_min_px
y_range_px = y_max_px - y_min_px

print(f"  X range: {x_min_px} to {x_max_px} = {x_range_px}px")
print(f"  Y range: {y_min_px} to {y_max_px} = {y_range_px}px")

pixels_per_meter_x = x_range_px / EXPECTED_WORLD_WIDTH
pixels_per_meter_y = y_range_px / EXPECTED_WORLD_HEIGHT
pixels_per_meter = 0.5 * (pixels_per_meter_x + pixels_per_meter_y)

print("\nPIXELS PER METER ESTIMATE")
print("-------------------------")
print(f"  X: {pixels_per_meter_x:.2f} px/m")
print(f"  Y: {pixels_per_meter_y:.2f} px/m")
print(f"  Average: {pixels_per_meter:.2f} px/m")

# -------------------------------------------------------------------
# GENERATE SYNTHETIC GRID + NOISE (SIMULATION OF FIELD MEASUREMENTS)
# -------------------------------------------------------------------
print(f"\nGENERATING {GRID_COLS * GRID_ROWS} GRID POINTS WITH NOISE (SIMULATION)")
print("------------------------------------------------------------------------")

pixel_pts = []
world_pts_noisy = []
world_pts_true = []

for row_idx in range(GRID_ROWS):
    for col_idx in range(GRID_COLS):
        # Ideal pixel position on the image plane (uniform in pixel space)
        px = x_min_px + (col_idx / (GRID_COLS - 1)) * x_range_px if GRID_COLS > 1 else x_min_px
        py = y_min_px + (row_idx / (GRID_ROWS - 1)) * y_range_px if GRID_ROWS > 1 else y_min_px

        # Ideal world offset in meters (using simple planar mapping)
        world_offset_x = (px - x_min_px) / pixels_per_meter_x
        world_offset_y = (py - y_min_px) / pixels_per_meter_y

        # Store "true" world coordinates (no noise) – this is the hidden ground truth
        world_pts_true.append([world_offset_x, world_offset_y])

        # Simulate measurement error in world coordinates (what surveyor would give)
        noise_x = np.random.normal(0, NOISE_STD_DEV)
        noise_y = np.random.normal(0, NOISE_STD_DEV)
        wx_measured = world_offset_x + noise_x
        wy_measured = world_offset_y + noise_y

        pixel_pts.append([px, py])
        world_pts_noisy.append([wx_measured, wy_measured])

pixel_pts = np.array(pixel_pts, dtype=np.float32)
world_pts_noisy = np.array(world_pts_noisy, dtype=np.float32)
world_pts_true = np.array(world_pts_true, dtype=np.float32)

N_POINTS = len(pixel_pts)
print(f"  Generated {N_POINTS} pixel–world correspondences (with simulated measurement noise)")

# -------------------------------------------------------------------
# TRAIN / VALIDATION SPLIT (NO MORE CIRCULAR VALIDATION)
# -------------------------------------------------------------------
VAL_RATIO = 0.15
val_size = int(N_POINTS * VAL_RATIO)
train_size = N_POINTS - val_size

indices = np.arange(N_POINTS)
np.random.shuffle(indices)

train_idx = indices[:train_size]
val_idx = indices[train_size:]

pixel_train = pixel_pts[train_idx]
world_train_noisy = world_pts_noisy[train_idx]

pixel_val = pixel_pts[val_idx]
world_val_true = world_pts_true[val_idx]   # validate against TRUE world, not noisy

print("\nTRAIN / VALIDATION SPLIT")
print("------------------------")
print(f"  Train points: {len(pixel_train)}")
print(f"  Val points:   {len(pixel_val)} (held-out, uses TRUE world coords for evaluation)")

# -------------------------------------------------------------------
# ESTIMATE HOMOGRAPHY FROM TRAIN SET ONLY
# -------------------------------------------------------------------
print("\nESTIMATING HOMOGRAPHY ON TRAIN SET (RANSAC)")
print("-------------------------------------------")
H_grid, mask_train = cv2.findHomography(
    pixel_train,
    world_train_noisy[:, :2],
    cv2.RANSAC,
    ransacReprojThreshold=2.0,
    confidence=0.99,
    maxIters=5000
)

train_inliers = mask_train.ravel().astype(bool)
train_inlier_ratio = np.mean(train_inliers)
print(f"  Train inliers: {train_inliers.sum()}/{len(train_inliers)} ({train_inlier_ratio:.1%})")

# -------------------------------------------------------------------
# EVALUATE ON TRAIN (NOISY WORLD) VS VALIDATION (TRUE WORLD)
# -------------------------------------------------------------------
def reprojection_errors(pixel_points, world_points_target, H):
    # Compute reprojection error (in meters) from pixel points to target world coords.
    projected = cv2.perspectiveTransform(pixel_points.reshape(-1, 1, 2), H).reshape(-1, 2)
    errs = np.linalg.norm(projected - world_points_target[:, :2], axis=1)
    return errs, projected

# 1) Train error (relative to noisy measured world coords)
train_errors, train_projected = reprojection_errors(pixel_train, world_train_noisy, H_grid)
train_mae = np.mean(train_errors[train_inliers])
train_std = np.std(train_errors[train_inliers])

# 2) Validation error (relative to TRUE world coords – independent)
val_errors, val_projected = reprojection_errors(pixel_val, world_val_true, H_grid)
val_mae = np.mean(val_errors)
val_std = np.std(val_errors)

print("\nERROR METRICS")
print("-------------")
print(f"  TRAIN (fit to noisy survey data):")
print(f"    MAE: {train_mae:.4f} m ({train_mae*100:.2f} cm)")
print(f"    Std: {train_std:.4f} m ({train_std*100:.2f} cm)")

print(f"\n  VALIDATION (independent TRUE world coordinates):")
print(f"    MAE: {val_mae:.4f} m ({val_mae*100:.2f} cm)")
print(f"    Std: {val_std:.4f} m ({val_std*100:.2f} cm)")

# -------------------------------------------------------------------
# K-FOLD CROSS-VALIDATION ON FULL SYNTHETIC GRID (AGAINST TRUE WORLD)
# -------------------------------------------------------------------
print("\nK-FOLD CROSS-VALIDATION (AGAINST TRUE WORLD)")
print("--------------------------------------------")
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

cv_maes = []
cv_stds = []

for fold, (train_idx_cv, val_idx_cv) in enumerate(kf.split(pixel_pts), start=1):
    pix_train_cv = pixel_pts[train_idx_cv]
    world_train_cv_noisy = world_pts_noisy[train_idx_cv]
    pix_val_cv = pixel_pts[val_idx_cv]
    world_val_cv_true = world_pts_true[val_idx_cv]

    H_cv, mask_cv = cv2.findHomography(
        pix_train_cv,
        world_train_cv_noisy[:, :2],
        cv2.RANSAC,
        ransacReprojThreshold=2.0,
        confidence=0.99,
        maxIters=5000
    )
    val_errs_cv, _ = reprojection_errors(pix_val_cv, world_val_cv_true, H_cv)
    cv_maes.append(np.mean(val_errs_cv))
    cv_stds.append(np.std(val_errs_cv))

    print(f"  Fold {fold}: MAE={cv_maes[-1]:.4f} m ({cv_maes[-1]*100:.2f} cm), "
          f"Std={cv_stds[-1]:.4f} m ({cv_stds[-1]*100:.2f} cm)")

cv_mae_mean = np.mean(cv_maes)
cv_mae_std = np.std(cv_maes)
print("\n  CROSS-VALIDATION SUMMARY:")
print(f"    Mean MAE: {cv_mae_mean:.4f} m ({cv_mae_mean*100:.2f} cm)")
print(f"    MAE Std:  {cv_mae_std:.4f} m ({cv_mae_std*100:.2f} cm)")

# -------------------------------------------------------------------
# LOAD ORIGINAL 6-POINT CALIBRATION FOR REFERENCE
# -------------------------------------------------------------------
print("\nLOADING ORIGINAL 6-POINT CALIBRATION")
print("------------------------------------")
orig_calib_path = os.path.join(project_root, "configs/giti_calibration_points.json")
with open(orig_calib_path) as f:
    orig_calib_data = json.load(f)

orig_pixel_pts = []
orig_world_pts = []

for p in orig_calib_data["calibration_points"]:
    orig_pixel_pts.append([p["pixel"]["x"], p["pixel"]["y"]])
    orig_world_pts.append([p["world"]["easting"], p["world"]["northing"]])

orig_pixel_pts = np.array(orig_pixel_pts, dtype=np.float32)
orig_world_pts = np.array(orig_world_pts, dtype=np.float32)

H_orig, mask_orig = cv2.findHomography(
    orig_pixel_pts,
    orig_world_pts[:, :2],
    cv2.RANSAC,
    ransacReprojThreshold=2.0
)

orig_projected = cv2.perspectiveTransform(
    orig_pixel_pts.reshape(-1, 1, 2),
    H_orig
).reshape(-1, 2)

orig_errors = np.linalg.norm(orig_projected - orig_world_pts[:, :2], axis=1)
orig_inliers = mask_orig.ravel().astype(bool)
mae_orig = np.mean(orig_errors[orig_inliers])
std_orig = np.std(orig_errors[orig_inliers])
max_error_orig = np.max(orig_errors[orig_inliers])

orig_area = (orig_world_pts[:, 0].max() - orig_world_pts[:, 0].min()) * \
            (orig_world_pts[:, 1].max() - orig_world_pts[:, 1].min())
grid_area = EXPECTED_WORLD_WIDTH * EXPECTED_WORLD_HEIGHT

print(f"  Original points: {len(orig_pixel_pts)}")
print(f"  Original MAE:    {mae_orig:.4f} m ({mae_orig*100:.2f} cm)")
print(f"  Original area:   {orig_area:.1f} m² vs Grid area: {grid_area:.1f} m²")

# -------------------------------------------------------------------
# COMPARISON TABLE
# -------------------------------------------------------------------
print("\n" + "="*70)
print("COMPARISON (SIMULATION-BASED GRID VS ORIGINAL 6-POINT)")
print("="*70)

print("\nMetric                | Original (6 pts)     | Grid (324 pts, sim)")
print("-------------------------------------------------------------------")
print(f"Points used           | {len(orig_pixel_pts):<20d} | {N_POINTS:<20d}")
print(f"Area covered (m²)     | {orig_area:<20.1f} | {grid_area:<20.1f}")
print(f"MAE (self-residual)   | {mae_orig:<20.4f} | {train_mae:<20.4f}")
print(f"Std (self-residual)   | {std_orig:<20.4f} | {train_std:<20.4f}")
print(f"Val MAE (TRUE world)  | {'N/A':<20}       | {val_mae:<20.4f}")
print(f"CV MAE mean (TRUE)    | {'N/A':<20}       | {cv_mae_mean:<20.4f}")

print("\nNOTE:")
print("  - 'Self-residual' = error on the same points used for fitting (optimistic).")
print("  - 'Val MAE / CV MAE' = error on held-out points w.r.t TRUE world coordinates (scientifically meaningful).")

# -------------------------------------------------------------------
# VISUALIZATION
# -------------------------------------------------------------------
print("\nCREATING VISUALIZATION...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1) Original 6-point world layout
ax = axes[0, 0]
ax.scatter(orig_world_pts[:, 0], orig_world_pts[:, 1],
           c='red', s=200, marker='o', alpha=0.7, edgecolors='darkred', linewidth=2)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title(f'Original (6 pts): MAE={mae_orig:.3f} m', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axis('equal')

# 2) Grid true world layout (color = val error if it was in val set)
ax = axes[0, 1]
colors = np.zeros(N_POINTS)
colors[val_idx] = np.interp(val_errors, (val_errors.min(), val_errors.max()), (0.2, 1.0))
scatter = ax.scatter(world_pts_true[:, 0], world_pts_true[:, 1],
                     c=colors, s=40, cmap='Reds', alpha=0.8)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Grid TRUE world points (val pts darker = higher error)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axis('equal')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Relative validation error (scaled)', fontsize=10)

# 3) Validation error histogram (TRUE world)
ax = axes[1, 0]
ax.hist(val_errors, bins=30, color='blue', alpha=0.7, edgecolor='black')
ax.axvline(val_mae, color='red', linestyle='--', linewidth=2, label=f'MAE={val_mae:.3f} m')
ax.set_xlabel('Validation reprojection error (m)')
ax.set_ylabel('Frequency')
ax.set_title('Validation error distribution (TRUE world)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# 4) Cross-validation MAE per fold
ax = axes[1, 1]
fold_ids = np.arange(1, N_SPLITS + 1)
ax.bar(fold_ids, np.array(cv_maes) * 100.0, color='green', alpha=0.7, edgecolor='black')
for i, mae_cv in enumerate(cv_maes, start=1):
    ax.text(i, mae_cv*100.0 + 0.05, f"{mae_cv*100:.2f} cm",
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xlabel('Fold')
ax.set_ylabel('MAE (cm)')
ax.set_title('K-fold CV (MAE on TRUE world)', fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(project_root, "calibration_validation_grid_vs_original.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------------------------------------------
# SAVE HOMOGRAPHY FOR PIPELINE USE
# -------------------------------------------------------------------
H_path = os.path.join(project_root, "H_grid_simulation_validated.npy")
np.save(H_path, H_grid)

print("\nSAVED ARTIFACTS")
print("---------------")
print(f"  Homography (simulation-based, validated): {H_path}")
print(f"  Visualization: {fig_path}")
print("\nIMPORTANT:")
print("  This script validates the GRID METHOD under a SIMULATED measurement model.")
print("  For real deployment, replace synthetic world_pts_noisy with real surveyed GCPs,")
print("  and keep the same train/val + CV evaluation logic.")
print("\n✅ PIPELINE-READY (SIMULATION LEVEL) – NEXT STEP: REAL GCP INTEGRATION")
