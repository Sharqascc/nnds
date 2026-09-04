import json
from unittest.mock import patch

import numpy as np
import pytest

from src.bev.giti_bev_calib import load_giti_homography


def make_calib_json(tmp_path, n_points=4, invalid_point=False):
    data = {"calibration_points": []}
    for i in range(n_points):
        if invalid_point and i == 0:
            data["calibration_points"].append(
                {"pixel": {"x": "bad", "y": 0}, "world": {"easting": 0, "northing": 0}}
            )
        else:
            data["calibration_points"].append(
                {
                    "pixel": {"x": float(i), "y": float(i)},
                    "world": {"easting": float(i * 2), "northing": float(i * 3)},
                }
            )
    path = tmp_path / "calib.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_too_few_points(tmp_path):
    p = tmp_path / "calib.json"
    p.write_text(
        json.dumps(
            {
                "calibration_points": [
                    {"pixel": {"x": 0, "y": 0}, "world": {"easting": 0, "northing": 0}}
                ]
            }
        )
    )
    with pytest.raises(ValueError):
        load_giti_homography(str(p))


def test_invalid_point_entry(tmp_path):
    p = make_calib_json(tmp_path, n_points=4, invalid_point=True)
    with pytest.raises(ValueError, match="Invalid calibration point entry"):
        load_giti_homography(p)


def test_find_homography_returns_none(tmp_path):
    p = make_calib_json(tmp_path, n_points=4)
    with patch("cv2.findHomography", return_value=(None, None)):
        with pytest.raises(RuntimeError, match="cv2.findHomography failed"):
            load_giti_homography(p)


def test_no_inliers_with_stats(tmp_path):
    p = make_calib_json(tmp_path, n_points=4)
    H = np.eye(3, dtype=np.float32)
    mask = np.zeros((4, 1), dtype=np.uint8)
    # Mock perspectiveTransform to return some points (not actually used for stats if no inliers)
    with patch("cv2.findHomography", return_value=(H, mask)):
        result = load_giti_homography(p, return_stats=True)
    assert len(result) == 4
    stats = result[3]
    assert np.isnan(stats["mean_error"])
    assert np.isnan(stats["max_error"])
    assert stats["num_inliers"] == 0
    assert stats["inlier_ratio"] == 0.0


def test_success_no_inliers_without_stats(tmp_path):
    p = make_calib_json(tmp_path, n_points=4)
    H = np.eye(3, dtype=np.float32)
    mask = np.zeros((4, 1), dtype=np.uint8)
    with patch("cv2.findHomography", return_value=(H, mask)):
        result = load_giti_homography(p)
    assert len(result) == 3
    assert result[0] is H
    assert result[1] is mask


# Clear cache before each test to avoid interference
@pytest.fixture(autouse=True)
def clear_cache():
    load_giti_homography.cache_clear()
    yield
    load_giti_homography.cache_clear()
