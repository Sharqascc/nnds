#!/usr/bin/env python3
"""
Leave-one-out cross-validation for BEV homography.

For each calibration point, fit H on the remaining points (with Hartley normalization)
and measure the reprojection error on the left-out point.

Usage:
    python scripts/bev_heldout_validation.py
"""
import json
import cv2
import numpy as np
from pathlib import Path

def hartley_normalize(pts):
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    mean_dist = np.mean(np.linalg.norm(centered, axis=1))
    scale = np.sqrt(2) / mean_dist if mean_dist > 0 else 1.0
    T = np.array([[scale, 0, -scale*centroid[0]],
                  [0, scale, -scale*centroid[1]],
                  [0, 0, 1]])
    pts_h = np.hstack([pts, np.ones((len(pts),1))])
    normalized = (T @ pts_h.T).T[:, :2]
    return normalized, T

def main():
    repo = Path(__file__).resolve().parents[1]
    calib_path = repo / 'configs/giti_calibration_points.json'
    bev_path = repo / 'configs/bev_config.json'
    with open(calib_path) as f:
        calib = json.load(f)
    with open(bev_path) as f:
        bev = json.load(f)
    points = calib['calibration_points']
    pixel = np.array([[p['pixel']['x'], p['pixel']['y']] for p in points], dtype=float)
    world = np.array([[p['world']['easting'], p['world']['northing']] for p in points], dtype=float)
    errors = []
    for i in range(len(pixel)):
        idx = [j for j in range(len(pixel)) if j != i]
        px_train = pixel[idx]
        wx_train = world[idx]
        # Hartley normalize train points
        px_norm, T_px = hartley_normalize(px_train)
        wx_norm, T_wx = hartley_normalize(wx_train)
        H_norm, _ = cv2.findHomography(px_norm, wx_norm, cv2.RANSAC, 5.0)
        H = np.linalg.inv(T_wx) @ H_norm @ T_px
        H = H / H[2,2]
        # Test on left-out point
        px_test = pixel[i]
        p_h = np.array([px_test[0], px_test[1], 1.0])
        proj = H @ p_h
        proj = proj[:2] / proj[2]
        err = np.linalg.norm(proj - world[i])
        errors.append(err)
    errors = np.array(errors)
    print("Leave-one-out BEV validation")
    print(f"  Mean held-out reprojection error: {errors.mean():.6f} ft")
    print(f"  Max held-out reprojection error: {errors.max():.6f} ft")
    print(f"  Std held-out reprojection error: {errors.std():.6f} ft")

if __name__ == "__main__":
    main()
