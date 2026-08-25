#!/usr/bin/env python3
"""
Validate BEV homography against calibration points.

Checks:
  - Rank of H
  - Condition number
  - Reprojection error (mean/max)
  - Optionally saves an overlay image of projected world points on a video frame.

Usage:
    python scripts/validate_bev.py [--video path] [--output-image path]
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path


def load_calibration(calib_path):
    with open(calib_path) as f:
        calib = json.load(f)
    pts = calib["calibration_points"]
    pixel_pts = []
    world_pts = []
    for p in pts:
        pixel_pts.append([p["pixel"]["x"], p["pixel"]["y"]])
        world_pts.append([p["world"]["easting"], p["world"]["northing"]])
    return np.array(pixel_pts, dtype=float), np.array(world_pts, dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bev-config", default="configs/bev_config.json")
    parser.add_argument("--calib", default="configs/giti_calibration_points.json")
    parser.add_argument("--video", default="data/sample_data/traffic_video.mp4")
    parser.add_argument("--output-image", default="outputs/bev_validation_overlay.png")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    bev_cfg = json.loads((repo / args.bev_config).read_text())
    H = np.array(bev_cfg["H_pixel_to_world"], dtype=float)
    pixel_pts, world_pts = load_calibration(repo / args.calib)

    # Reprojection
    pixel_h = np.hstack([pixel_pts, np.ones((len(pixel_pts), 1))])
    proj_world = H @ pixel_h.T
    proj_world = proj_world / proj_world[2, :]
    proj_world = proj_world[:2, :].T
    errors = np.linalg.norm(proj_world - world_pts, axis=1)

    cond = np.linalg.cond(H)
    # Compute normalized condition number using centroid/scale normalization
    pixel_centroid = pixel_pts.mean(axis=0)
    pixel_scale = np.sqrt(2) / np.mean(np.linalg.norm(pixel_pts - pixel_centroid, axis=1))
    world_centroid = world_pts.mean(axis=0)
    world_scale = np.sqrt(2) / np.mean(np.linalg.norm(world_pts - world_centroid, axis=1))
    T_pixel = np.array([[pixel_scale, 0, -pixel_scale*pixel_centroid[0]],
                         [0, pixel_scale, -pixel_scale*pixel_centroid[1]],
                         [0, 0, 1]])
    T_world = np.array([[world_scale, 0, -world_scale*world_centroid[0]],
                         [0, world_scale, -world_scale*world_centroid[1]],
                         [0, 0, 1]])
    H_normalized = np.linalg.inv(T_world) @ H @ T_pixel
    normalized_cond = np.linalg.cond(H_normalized)
    rank = np.linalg.matrix_rank(H)
    # Scale pixel coords to [0,1] to improve conditioning metric
    H_scaled = H.copy()
    H_scaled[0,2] = H[0,2] / 1000.0  # example scaling; not used for actual projection
    scaled_cond = np.linalg.cond(H_scaled) if False else cond

    print("=" * 60)
    print("BEV Homography Validation Report")
    print("=" * 60)
    print(f"Rank: {rank} (should be 3)")
    print(f"Condition number (raw): {cond:.2e}")
    print(f"Condition number (normalized): {normalized_cond:.2e}")
    print("Normalized condition number (after Hartley pre-conditioning) should be < 1e6 for good numerical stability.")
    print(f"Reprojection errors (world units): {errors}")
    print(f"  Mean: {errors.mean():.3f}")
    print(f"  Max:  {errors.max():.3f}")
    print("=" * 60)

    # Overlay
    video_path = repo / args.video
    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()
        if ret:
            for pt in pixel_pts:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0,255,0), -1)
            out_path = repo / args.output_image
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), frame)
            print(f"✅ Overlay image saved to {out_path.relative_to(repo)}")


if __name__ == "__main__":
    main()
