import json
from unittest.mock import patch

import numpy as np
import pytest

from src.bev.bev_mapper import (
    BEVMapper,
    _project_points_homography,
    compute_homography_dlt,
    main,
    test_with_real_calibration,
)

# ------------------- compute_homography_dlt -------------------


def test_compute_homography_dlt_valid():
    src = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    H = compute_homography_dlt(src, dst)
    assert H.shape == (3, 3)
    assert np.isclose(H[2, 2], 1.0)


def test_compute_homography_dlt_shape_mismatch():
    src = np.zeros((4, 2))
    dst = np.zeros((3, 2))
    with pytest.raises(ValueError):
        compute_homography_dlt(src, dst)


def test_compute_homography_dlt_not_2d():
    src = np.zeros((4, 3))
    dst = np.zeros((4, 2))
    with pytest.raises(ValueError):
        compute_homography_dlt(src, dst)


def test_compute_homography_dlt_too_few_points():
    src = np.zeros((3, 2))
    dst = np.zeros((3, 2))
    with pytest.raises(ValueError):
        compute_homography_dlt(src, dst)


def test_compute_homography_dlt_degenerate():
    src = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
    dst = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
    # Patch np.linalg.svd to return a Vt with last row having H[2,2]=0
    fake_vt = np.eye(9)[::-1].copy()
    fake_vt[-1, -1] = 0.0  # set the last element to 0
    with patch("numpy.linalg.svd", return_value=(None, None, fake_vt)):
        with pytest.raises(RuntimeError):
            compute_homography_dlt(src, dst)


# ------------------- _project_points_homography -------------------


def test_project_points_homography_valid():
    pts = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
    H = np.eye(3)
    result = _project_points_homography(pts, H)
    assert result.shape == (3, 2)
    assert np.allclose(result, pts)


def test_project_points_homography_denom_zero():
    pts = np.array([[1, 2]])
    H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64)
    result = _project_points_homography(pts, H)
    assert np.isnan(result[0, 0]) and np.isnan(result[0, 1])


# ------------------- BEVMapper -------------------


@pytest.fixture
def simple_mapper():
    H = np.eye(3, dtype=np.float32)
    bounds = {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100}
    res = (100, 100)
    return BEVMapper(H, bounds, res)


def test_bev_mapper_init(simple_mapper):
    assert simple_mapper.bev_w == 100
    assert simple_mapper.bev_h == 100
    assert simple_mapper.mpp_x == 1.0
    assert simple_mapper.mpp_y == 1.0


def test_pixel_to_world_valid(simple_mapper):
    world = simple_mapper.pixel_to_world((10.0, 20.0))
    assert world == (10.0, 20.0)


def test_pixel_to_world_denom_zero(simple_mapper):
    simple_mapper.H[2, 2] = 0.0
    assert simple_mapper.pixel_to_world((1, 2)) is None


def test_pixel_to_world_exception(simple_mapper):
    assert simple_mapper.pixel_to_world(None) is None


def test_world_to_bev_valid(simple_mapper):
    assert simple_mapper.world_to_bev((10.0, 20.0)) == (10, 20)


def test_world_to_bev_exception(simple_mapper):
    assert simple_mapper.world_to_bev(None) is None


def test_pixel_to_bev_valid(simple_mapper):
    assert simple_mapper.pixel_to_bev((10.0, 20.0)) == (10, 20)


def test_pixel_to_bev_world_none(simple_mapper):
    with patch.object(simple_mapper, "pixel_to_world", return_value=None):
        assert simple_mapper.pixel_to_bev((1, 2)) is None


# ------------------- test_with_real_calibration -------------------


def make_calib_files(tmp_path, calib_data=None, bev_data=None):
    calib_path = tmp_path / "calib.json"
    bev_path = tmp_path / "bev.json"
    if calib_data is not None:
        calib_path.write_text(json.dumps(calib_data))
    if bev_data is not None:
        bev_path.write_text(json.dumps(bev_data))
    return str(calib_path), str(bev_path)


def test_real_calibration_missing_calib(tmp_path):
    calib_path, bev_path = make_calib_files(tmp_path, None, {})
    result = test_with_real_calibration(calib_json=calib_path, bev_json=bev_path)
    assert result is None


def test_real_calibration_missing_bev(tmp_path):
    calib_path, bev_path = make_calib_files(
        tmp_path, {"pixel_points": [], "world_points": []}, None
    )
    result = test_with_real_calibration(calib_json=calib_path, bev_json=bev_path)
    assert result is None


def test_real_calibration_insufficient_points(tmp_path):
    calib_data = {"pixel_points": [[0, 0]], "world_points": [[0, 0]]}
    bev_data = {
        "bounds": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100},
        "resolution": [100, 100],
    }
    calib_path, bev_path = make_calib_files(tmp_path, calib_data, bev_data)
    result = test_with_real_calibration(calib_json=calib_path, bev_json=bev_path)
    assert result is None


def test_real_calibration_shape_mismatch(tmp_path):
    calib_data = {
        "pixel_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
        "world_points": [[0, 0], [1, 0], [0, 1], [1, 1], [2, 2]],
    }
    bev_data = {
        "bounds": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100},
        "resolution": [100, 100],
    }
    calib_path, bev_path = make_calib_files(tmp_path, calib_data, bev_data)
    result = test_with_real_calibration(calib_json=calib_path, bev_json=bev_path)
    assert result is None


def test_real_calibration_pixel_not_2d(tmp_path):
    calib_data = {
        "pixel_points": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        "world_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
    }
    bev_data = {
        "bounds": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100},
        "resolution": [100, 100],
    }
    calib_path, bev_path = make_calib_files(tmp_path, calib_data, bev_data)
    result = test_with_real_calibration(calib_json=calib_path, bev_json=bev_path)
    assert result is None


def test_real_calibration_world_not_2d(tmp_path):
    calib_data = {"pixel_points": [[0, 0], [1, 0], [0, 1], [1, 1]], "world_points": [0, 1, 2, 3]}
    bev_data = {
        "bounds": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100},
        "resolution": [100, 100],
    }
    calib_path, bev_path = make_calib_files(tmp_path, calib_data, bev_data)
    result = test_with_real_calibration(calib_json=calib_path, bev_json=bev_path)
    assert result is None


def test_real_calibration_missing_bounds_res(tmp_path):
    calib_data = {
        "pixel_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
        "world_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
    }
    bev_data = {}
    calib_path, bev_path = make_calib_files(tmp_path, calib_data, bev_data)
    result = test_with_real_calibration(calib_json=calib_path, bev_json=bev_path)
    assert result is None


def test_real_calibration_success_cv2(tmp_path):
    calib_data = {
        "pixel_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
        "world_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
    }
    bev_data = {
        "bounds": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100},
        "resolution": [100, 100],
    }
    calib_path, bev_path = make_calib_files(tmp_path, calib_data, bev_data)

    H = np.eye(3, dtype=np.float32)
    mask = np.ones((4, 1), dtype=np.uint8)  # inliers

    with patch("cv2.findHomography", return_value=(H, mask)):
        result = test_with_real_calibration(calib_json=calib_path, bev_json=bev_path)
    assert result is not None
    assert result["passes_validation"]
    assert result["rmse_m"] < 1.0


def test_real_calibration_fallback_to_dlt(tmp_path):
    calib_data = {
        "pixel_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
        "world_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
    }
    bev_data = {
        "bounds": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100},
        "resolution": [100, 100],
    }
    calib_path, bev_path = make_calib_files(tmp_path, calib_data, bev_data)

    with patch("cv2.findHomography", side_effect=Exception("fail")):
        result = test_with_real_calibration(calib_json=calib_path, bev_json=bev_path)
    assert result is not None


def test_real_calibration_bounds_missing_keys(tmp_path):
    calib_data = {
        "pixel_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
        "world_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
    }
    bev_data = {"bounds": {"x_min": 0}, "resolution": [100, 100]}  # missing y_min, x_max, y_max
    calib_path, bev_path = make_calib_files(tmp_path, calib_data, bev_data)

    # Dummy mapper class that doesn't access bounds during init
    class DummyMapper:
        def __init__(self, H_pixel_to_world=None, bev_bounds=None, bev_resolution=None):
            pass

    with (
        patch("cv2.findHomography", return_value=(np.eye(3), np.ones((4, 1), dtype=np.uint8))),
        patch("src.bev.bev_mapper.BEVMapper", DummyMapper),
    ):
        result = test_with_real_calibration(calib_json=calib_path, bev_json=bev_path)
    assert result is None


def test_real_calibration_uncertainty_with_method(tmp_path):
    calib_data = {
        "pixel_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
        "world_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
    }
    bev_data = {
        "bounds": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100},
        "resolution": [100, 100],
    }
    calib_path, bev_path = make_calib_files(tmp_path, calib_data, bev_data)

    # Mock BEVMapper to have estimate_transformation_error and world_to_bev_batch
    class MockMapper(BEVMapper):
        def estimate_transformation_error(self, p, pixel_error_std=0.5):
            return 0.1

        def world_to_bev_batch(self, points):
            return np.zeros_like(points), np.ones(len(points), dtype=bool)

    with (
        patch("cv2.findHomography", return_value=(np.eye(3), np.ones((4, 1), dtype=np.uint8))),
        patch("src.bev.bev_mapper.BEVMapper", MockMapper),
    ):
        result = test_with_real_calibration(calib_json=calib_path, bev_json=bev_path)
    assert result is not None
    assert result["mean_uncertainty_m"] == 0.1


# ------------------- main -------------------


def test_main_success(tmp_path, capsys):
    calib_data = {
        "pixel_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
        "world_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
    }
    bev_data = {
        "bounds": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100},
        "resolution": [100, 100],
    }
    calib_path, bev_path = make_calib_files(tmp_path, calib_data, bev_data)

    with (
        patch("sys.argv", ["prog", "--calib", calib_path, "--bev-config", bev_path]),
        patch("cv2.findHomography", return_value=(np.eye(3), np.ones((4, 1), dtype=np.uint8))),
    ):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 0


def test_main_failure_none(tmp_path):
    with patch(
        "sys.argv",
        [
            "prog",
            "--calib",
            str(tmp_path / "missing.json"),
            "--bev-config",
            str(tmp_path / "missing_bev.json"),
        ],
    ):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 1


def test_main_quiet(tmp_path):
    calib_data = {
        "pixel_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
        "world_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
    }
    bev_data = {
        "bounds": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100},
        "resolution": [100, 100],
    }
    calib_path, bev_path = make_calib_files(tmp_path, calib_data, bev_data)
    with (
        patch("sys.argv", ["prog", "--calib", calib_path, "--bev-config", bev_path, "--quiet"]),
        patch("cv2.findHomography", return_value=(np.eye(3), np.ones((4, 1), dtype=np.uint8))),
    ):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 0


def test_main_raise_on_failure(tmp_path):
    # Make a calibration that passes? Actually raise_on_failure only if passes_validation False.
    # We can patch test_with_real_calibration to return failure.
    with (
        patch(
            "sys.argv", ["prog", "--calib", "dummy", "--bev-config", "dummy", "--raise-on-failure"]
        ),
        patch(
            "src.bev.bev_mapper.test_with_real_calibration",
            return_value={"passes_validation": False, "rmse_m": 2.0},
        ),
        pytest.raises(RuntimeError),
    ):
        main()


def test_real_calibration_shape_mismatch(tmp_path):
    calib_data = {
        "pixel_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
        "world_points": [[0, 0], [1, 0], [0, 1], [1, 1], [2, 2]],
    }
    bev_data = {
        "bounds": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100},
        "resolution": [100, 100],
    }
    calib_path, bev_path = make_calib_files(tmp_path, calib_data, bev_data)
    result = test_with_real_calibration(calib_json=calib_path, bev_json=bev_path)
    assert result is None


def test_real_calibration_cv2_returns_none(tmp_path):
    calib_data = {
        "pixel_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
        "world_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
    }
    bev_data = {
        "bounds": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100},
        "resolution": [100, 100],
    }
    calib_path, bev_path = make_calib_files(tmp_path, calib_data, bev_data)
    with patch("cv2.findHomography", return_value=(None, None)):
        result = test_with_real_calibration(calib_json=calib_path, bev_json=bev_path)
    assert result is not None


def test_real_calibration_no_valid_projection(tmp_path):
    calib_data = {
        "pixel_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
        "world_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
    }
    bev_data = {
        "bounds": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100},
        "resolution": [100, 100],
    }
    calib_path, bev_path = make_calib_files(tmp_path, calib_data, bev_data)
    with (
        patch("cv2.findHomography", return_value=(np.eye(3), np.ones((4, 1), dtype=np.uint8))),
        patch(
            "src.bev.bev_mapper._project_points_homography", return_value=np.full((4, 2), np.nan)
        ),
    ):
        result = test_with_real_calibration(calib_json=calib_path, bev_json=bev_path)
    assert result is None


def test_real_calibration_quality_branches(tmp_path):
    """Cover quality GOOD, ACCEPTABLE, POOR branches."""
    # We'll patch _project_points_homography to give errors corresponding to each quality
    for rmse, expected in [
        (0.3, "GOOD"),
        (0.7, "ACCEPTABLE"),
        (2.0, "POOR - RECALIBRATION RECOMMENDED"),
    ]:
        calib_data = {
            "pixel_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
            "world_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
        }
        bev_data = {
            "bounds": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100},
            "resolution": [100, 100],
        }
        calib_path, bev_path = make_calib_files(tmp_path, calib_data, bev_data)
        # Simulate world_reproj with errors that lead to desired rmse
        # We'll patch _project_points_homography to return known points with error
        world_reproj = np.array(
            [[0, 0], [1, 0], [0, 1], [1 + rmse * 2, 1]]
        )  # one point off by rmse*2
        with (
            patch("cv2.findHomography", return_value=(np.eye(3), np.ones((4, 1), dtype=np.uint8))),
            patch("src.bev.bev_mapper._project_points_homography", return_value=world_reproj),
        ):
            result = test_with_real_calibration(calib_json=calib_path, bev_json=bev_path)
        assert result is not None
        assert result["quality"] == expected


def test_real_calibration_typeerror_in_uncertainty(tmp_path):
    """Cover TypeError fallback in estimate_transformation_error."""
    calib_data = {
        "pixel_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
        "world_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
    }
    bev_data = {
        "bounds": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100},
        "resolution": [100, 100],
    }
    calib_path, bev_path = make_calib_files(tmp_path, calib_data, bev_data)

    class MockMapper(BEVMapper):
        def estimate_transformation_error(self, p, **kwargs):
            # Raise TypeError if pixel_error_std passed, to force fallback
            if "pixel_error_std" in kwargs:
                raise TypeError("unexpected kwarg")
            return 0.2

        def world_to_bev_batch(self, points):
            return np.zeros_like(points), np.ones(len(points), dtype=bool)

    with (
        patch("cv2.findHomography", return_value=(np.eye(3), np.ones((4, 1), dtype=np.uint8))),
        patch("src.bev.bev_mapper.BEVMapper", MockMapper),
    ):
        result = test_with_real_calibration(calib_json=calib_path, bev_json=bev_path)
    assert result is not None
    assert result["mean_uncertainty_m"] == 0.2


def test_real_calibration_inverse_failure(tmp_path):
    """Cover inverse transformation exception and N/A roundtrip."""
    calib_data = {
        "pixel_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
        "world_points": [[0, 0], [1, 0], [0, 1], [1, 1]],
    }
    bev_data = {
        "bounds": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100},
        "resolution": [100, 100],
    }
    calib_path, bev_path = make_calib_files(tmp_path, calib_data, bev_data)

    with (
        patch("cv2.findHomography", return_value=(np.eye(3), np.ones((4, 1), dtype=np.uint8))),
        patch("numpy.linalg.inv", side_effect=Exception("inverse failed")),
    ):
        result = test_with_real_calibration(calib_json=calib_path, bev_json=bev_path)
    assert result is not None
    assert np.isnan(result["roundtrip_mean_px"])


def test_main_validation_failure_exit_1(tmp_path):
    """Cover sys.exit(1) when validation fails without raise_on_failure."""
    with (
        patch("sys.argv", ["prog", "--calib", "dummy", "--bev-config", "dummy"]),
        patch(
            "src.bev.bev_mapper.test_with_real_calibration",
            return_value={"passes_validation": False, "rmse_m": 2.0},
        ),
    ):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 1
