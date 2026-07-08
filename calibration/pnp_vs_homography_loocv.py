"""
Compare planar homography vs PnP-based calibration on the REAL G-ITI GCPs,
using leave-one-out cross-validation (LOOCV) for both methods.
"""
import json
from pathlib import Path

import cv2
import numpy as np

FEET_TO_M = 0.3048

CALIB_JSON = Path("configs/giti_calibration_points.json")


def load_points(calib_json=CALIB_JSON):
    data = json.loads(Path(calib_json).read_text())
    pts = data["calibration_points"]

    pixel = np.array([[p["pixel"]["x"], p["pixel"]["y"]] for p in pts], dtype=np.float64)

    easting_northing = np.array(
        [[p["world"]["easting"], p["world"]["northing"]] for p in pts], dtype=np.float64
    )
    elevation_m = np.array([[p["world"]["elevation_ft"] * FEET_TO_M] for p in pts], dtype=np.float64)
    world_m = np.hstack([easting_northing, elevation_m])

    ids = [p["id"] for p in pts]
    return ids, pixel, world_m


def guess_intrinsics(image_size_str, pixel_pts):
    w_str, h_str = image_size_str.lower().split("x")
    w, h = int(w_str), int(h_str)
    f = float(w)
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    return K, dist


def homography_loocv(pixel_xy, world_xy):
    n = len(pixel_xy)
    errors = []
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        H, _ = cv2.findHomography(pixel_xy[train_idx], world_xy[train_idx], 0)
        if H is None:
            errors.append(np.nan)
            continue
        pred = cv2.perspectiveTransform(pixel_xy[i].reshape(1, 1, 2), H).reshape(2)
        err = np.linalg.norm(pred - world_xy[i])
        errors.append(err)
    return np.array(errors)


def pnp_loocv(pixel_xy, world_xyz, K, dist):
    n = len(pixel_xy)
    errors = []

    for i in range(n):
        train_idx = np.array([j for j in range(n) if j != i])
        obj_pts = world_xyz[train_idx].astype(np.float64)
        img_pts = pixel_xy[train_idx].astype(np.float64)

        if len(train_idx) < 4:
            errors.append(np.nan)
            continue

        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_EPNP
        )
        if not ok:
            errors.append(np.nan)
            continue

        R, _ = cv2.Rodrigues(rvec)
        plane_z = float(np.mean(obj_pts[:, 2]))

        u, v = pixel_xy[i]
        pt_img = np.array([u, v, 1.0], dtype=np.float64)
        ray_cam = np.linalg.inv(K) @ pt_img
        ray_world = R.T @ ray_cam
        cam_center = (-R.T @ tvec).flatten()

        if abs(ray_world[2]) < 1e-9:
            errors.append(np.nan)
            continue
        t = (plane_z - cam_center[2]) / ray_world[2]
        world_pred = cam_center + t * ray_world

        true_xy = world_xyz[i, :2]
        err = np.linalg.norm(world_pred[:2] - true_xy)
        errors.append(err)

    return np.array(errors)


def summarize(name, errors):
    valid = errors[np.isfinite(errors)]
    print(f"\n{name}")
    print(f"  valid folds: {len(valid)}/{len(errors)}")
    if len(valid) == 0:
        print("  no valid folds")
        return None
    print(f"  mean error:   {valid.mean():.4f} m")
    print(f"  median error: {np.median(valid):.4f} m")
    print(f"  max error:    {valid.max():.4f} m")
    print(f"  rmse:         {np.sqrt(np.mean(valid**2)):.4f} m")
    return {
        "mean_m": float(valid.mean()),
        "median_m": float(np.median(valid)),
        "max_m": float(valid.max()),
        "rmse_m": float(np.sqrt(np.mean(valid**2))),
        "n_valid": int(len(valid)),
        "n_total": int(len(errors)),
    }


def main():
    data = json.loads(CALIB_JSON.read_text())
    image_size_str = data["metadata"].get("image_size", "1600x720")

    ids, pixel, world_m = load_points()
    world_xy = world_m[:, :2]

    print("=" * 80)
    print("HOMOGRAPHY LOOCV (baseline, should match loocv_real_calibration_results.json)")
    print("=" * 80)
    h_errors = homography_loocv(pixel.astype(np.float32), world_xy.astype(np.float32))
    for pid, e in zip(ids, h_errors):
        print(f"  held-out id {pid}: error = {e:.4f} m")
    h_summary = summarize("Homography LOOCV summary", h_errors)

    print("\n" + "=" * 80)
    print("PnP LOOCV (using real 3D GCPs incl. elevation, approximate intrinsics, EPNP)")
    print("=" * 80)
    K, dist = guess_intrinsics(image_size_str, pixel)
    print("  Using ASSUMED intrinsics (no calibrated camera matrix found in repo):")
    print(f"  K =\n{K}")
    p_errors = pnp_loocv(pixel, world_m, K, dist)
    for pid, e in zip(ids, p_errors):
        print(f"  held-out id {pid}: error = {e:.4f} m" if np.isfinite(e) else f"  held-out id {pid}: FAILED")
    p_summary = summarize("PnP LOOCV summary", p_errors)

    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    if h_summary and p_summary:
        factor = p_summary["rmse_m"] / h_summary["rmse_m"]
        print(f"  Homography LOOCV RMSE: {h_summary['rmse_m']:.4f} m")
        print(f"  PnP LOOCV RMSE:        {p_summary['rmse_m']:.4f} m")
        print(f"  PnP/Homography ratio:  {factor:.3f}  ({'PnP better' if factor < 1 else 'Homography better'})")

    out = {
        "homography_loocv": h_summary,
        "pnp_loocv": p_summary,
        "assumed_intrinsics": K.tolist(),
        "note": "PnP uses assumed intrinsics; treat as feasibility check only. easting/northing NOT unit-converted (already metric).",
    }
    out_path = Path("calibration/pnp_vs_homography_loocv_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
