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


class BEVMapper:
    """Bird's Eye View mapper for converting between pixel, world, and BEV coordinates."""
    
    def __init__(self, H_pixel_to_world, bev_bounds, bev_resolution):
        import numpy as np
        self.H = np.asarray(H_pixel_to_world, dtype=np.float32)
        self.bev_x_min = float(bev_bounds["x_min"])
        self.bev_x_max = float(bev_bounds["x_max"])
        self.bev_y_min = float(bev_bounds["y_min"])
        self.bev_y_max = float(bev_bounds["y_max"])
        self.bev_w, self.bev_h = map(int, bev_resolution)
        self.mpp_x = (self.bev_x_max - self.bev_x_min) / max(self.bev_w, 1)
        self.mpp_y = (self.bev_y_max - self.bev_y_min) / max(self.bev_h, 1)
    
    def pixel_to_world(self, p):
        import numpy as np
        try:
            x, y = p
            v = np.array([x, y, 1.0], dtype=np.float32)
            w = self.H @ v
            if abs(w[2]) < 1e-6:
                return None
            w /= w[2]
            return float(w[0]), float(w[1])
        except Exception:
            return None
    
    def world_to_bev(self, world_xy):
        try:
            X, Y = world_xy
            u = int((X - self.bev_x_min) / self.mpp_x)
            v = int((Y - self.bev_y_min) / self.mpp_y)
            return u, v
        except Exception:
            return None
    
    def pixel_to_bev(self, p):
        world = self.pixel_to_world(p)
        if world is None:
            return None
        return self.world_to_bev(world)
