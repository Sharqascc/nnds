import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


@lru_cache(maxsize=4)
def load_giti_homography(
    json_path: PathLike,
    ransac_thresh: float = 5.0,
    pixel_key: str = "pixel",
    world_key: str = "world",
    x_key: str = "x",
    y_key: str = "y",
    easting_key: str = "easting",
    northing_key: str = "northing",
    return_stats: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]],
]:
    """
    Load GITI-style calibration points and compute a pixel->world homography.

    Parameters
    ----------
    json_path : str or Path
        Path to a JSON file with structure like:
        {
          "calibration_points": [
            {
              "<pixel_key>": {"<x_key>": ..., "<y_key>": ...},
              "<world_key>": {"<easting_key>": ..., "<northing_key>": ...}
            },
            ...
          ]
        }
    ransac_thresh : float, optional
        RANSAC reprojection threshold in pixels passed to cv2.findHomography.
    pixel_key, world_key, x_key, y_key, easting_key, northing_key : str
        Keys used to access pixel/world coordinates in the JSON.
    return_stats : bool, optional
        If True, also return a stats dict with reprojection error metrics.

    Returns
    -------
    H : np.ndarray
        3x3 homography matrix mapping [x_pix, y_pix, 1]^T -> [X_world, Y_world, w]^T.
    mask : np.ndarray
        Inlier mask returned by cv2.findHomography (shape (N, 1)).
    pts_world : np.ndarray
        Nx2 array of world points used for calibration (easting, northing).
    stats : dict, optional
        Only if return_stats=True. Contains fields:
        - mean_error
        - max_error
        - inlier_ratio
        - num_inliers
        - num_points

    Raises
    ------
    FileNotFoundError
        If json_path does not exist.
    ValueError
        If calibration_points is missing or has fewer than 4 points.
    RuntimeError
        If cv2.findHomography fails.
    """
    json_path = Path(json_path)
    logger.info("Loading BEV homography calibration from %s", json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    points = data.get("calibration_points", [])
    if not points or len(points) < 4:
        raise ValueError(
            f"Expected at least 4 calibration_points, got {len(points)} "
            f"in {json_path}"
        )

    pts_pix = []
    pts_world = []

    for p in points:
        try:
            px = float(p[pixel_key][x_key])
            py = float(p[pixel_key][y_key])
            X = float(p[world_key][easting_key])
            Y = float(p[world_key][northing_key])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Invalid calibration point entry {p}: {e}")

        pts_pix.append([px, py])
        pts_world.append([X, Y])

    pts_pix = np.asarray(pts_pix, dtype=np.float32)
    pts_world = np.asarray(pts_world, dtype=np.float32)

    logger.debug(
        "Computing homography with %d calibration points (ransac_thresh=%.3f)",
        len(pts_pix),
        ransac_thresh,
    )

    H, mask = cv2.findHomography(
        pts_pix,
        pts_world,
        cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
    )
    if H is None:
        raise RuntimeError(
            f"cv2.findHomography failed for calibration file {json_path}"
        )

    logger.info(
        "Homography computed for %s (inliers: %d / %d)",
        json_path,
        int(mask.sum()),
        len(mask),
    )

    if not return_stats:
        return H, mask, pts_world

    # Compute reprojection error statistics for inliers
    pts_pix_reshaped = pts_pix.reshape(-1, 1, 2)
    pts_world_proj = cv2.perspectiveTransform(pts_pix_reshaped, H).reshape(-1, 2)
    errors = np.linalg.norm(pts_world_proj - pts_world, axis=1)

    inlier_mask = mask.astype(bool).ravel()
    if inlier_mask.any():
        inlier_errors = errors[inlier_mask]
        mean_error = float(np.mean(inlier_errors))
        max_error = float(np.max(inlier_errors))
        num_inliers = int(inlier_mask.sum())
    else:
        mean_error = float("nan")
        max_error = float("nan")
        num_inliers = 0

    stats = {
        "mean_error": mean_error,
        "max_error": max_error,
        "inlier_ratio": float(num_inliers / len(mask)),
        "num_inliers": num_inliers,
        "num_points": int(len(pts_pix)),
    }

    logger.info(
        "Calibration stats for %s: mean_error=%.4f, max_error=%.4f, "
        "inlier_ratio=%.3f",
        json_path,
        stats["mean_error"],
        stats["max_error"],
        stats["inlier_ratio"],
    )

    return H, mask, pts_world, stats
