import json
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.bev.calibration.grid_validation_calibration import (
    configure_logging,
    export_results,
    generate_synthetic_grid,
    load_grid_config,
    load_original_calibration,
    main,
    parse_args,
    reprojection_errors,
)


# ---------------- parse_args ----------------
def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])
    args = parse_args()
    assert args.grid_cols == 27
    assert args.grid_rows == 12
    assert args.activa_length_m == 1.833
    assert args.n_splits == 5
    assert args.real_data is False


def test_parse_args_custom(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--grid-cols",
            "3",
            "--grid-rows",
            "2",
            "--noise-std-m",
            "0.1",
            "--real-data",
            "--verbose",
        ],
    )
    args = parse_args()
    assert args.grid_cols == 3
    assert args.grid_rows == 2
    assert args.noise_std_m == 0.1
    assert args.real_data is True
    assert args.verbose is True


# ---------------- configure_logging ----------------
def test_configure_logging(capsys):
    configure_logging(verbose=True)
    # Just ensure no error
    configure_logging(verbose=False)


# ---------------- load_grid_config ----------------
def test_load_grid_config(tmp_path):
    cfg = {
        "corners": {
            "top_left": [10, 20],
            "top_right": [110, 20],
            "bottom_left": [10, 120],
            "bottom_right": [110, 120],
        }
    }
    path = tmp_path / "grid.json"
    path.write_text(json.dumps(cfg))
    x_min, x_max, y_min, y_max = load_grid_config(path)
    assert (x_min, x_max, y_min, y_max) == (10, 110, 20, 120)


# ---------------- reprojection_errors ----------------
def test_reprojection_errors_identity():
    pixel = np.array([[1, 2], [3, 4]], dtype=np.float32)
    world = np.array([[1, 2], [3, 4]], dtype=np.float32)
    H = np.eye(3, dtype=np.float32)
    errors, projected = reprojection_errors(pixel, world, H)
    assert np.allclose(errors, 0.0)
    assert projected.shape == (2, 2)


# ---------------- generate_synthetic_grid ----------------
def test_generate_synthetic_grid_shapes():
    rng = np.random.default_rng(42)
    pixel, noisy, true = generate_synthetic_grid(
        grid_rows=2,
        grid_cols=3,
        x_min_px=0,
        x_max_px=100,
        y_min_px=0,
        y_max_px=50,
        pixels_per_meter_x=10.0,
        pixels_per_meter_y=10.0,
        noise_std=0.01,
        rng=rng,
    )
    assert pixel.shape == (6, 2)
    assert noisy.shape == (6, 2)
    assert true.shape == (6, 2)
    # True world should be noiseless and deterministic
    assert not np.allclose(noisy, true)


def test_generate_synthetic_grid_single_row_col():
    rng = np.random.default_rng(42)
    pixel, noisy, true = generate_synthetic_grid(
        grid_rows=1,
        grid_cols=1,
        x_min_px=0,
        x_max_px=100,
        y_min_px=0,
        y_max_px=50,
        pixels_per_meter_x=10.0,
        pixels_per_meter_y=10.0,
        noise_std=0.0,
        rng=rng,
    )
    assert pixel.shape == (1, 2)
    assert pixel[0][0] == 0
    assert pixel[0][1] == 0
    assert true[0][0] == 0
    assert true[0][1] == 0


# ---------------- load_original_calibration ----------------
def test_load_original_calibration(tmp_path):
    calib = {
        "calibration_points": [
            {"pixel": {"x": 1, "y": 2}, "world": {"easting": 10, "northing": 20}},
            {"pixel": {"x": 3, "y": 4}, "world": {"easting": 30, "northing": 40}},
        ]
    }
    path = tmp_path / "orig.json"
    path.write_text(json.dumps(calib))
    pix, world = load_original_calibration(path)
    assert pix.shape == (2, 2)
    assert world.shape == (2, 2)
    assert pix[1][0] == 3
    assert world[1][1] == 40


# ---------------- export_results ----------------
def test_export_results(tmp_path):
    out = tmp_path / "results.json"
    export_results({"a": 1, "b": [2, 3]}, out)
    data = json.loads(out.read_text())
    assert data["a"] == 1
    assert data["b"] == [2, 3]


# ---------------- main full mocked run ----------------
def make_project_files(tmp_path):
    """Create minimal config and calibration files for main()."""
    configs = tmp_path / "configs"
    configs.mkdir()
    grid_cfg = {
        "corners": {
            "top_left": [0, 0],
            "top_right": [100, 0],
            "bottom_left": [0, 80],
            "bottom_right": [100, 80],
        }
    }
    (configs / "GITI_grid_config.json").write_text(json.dumps(grid_cfg))
    orig_calib = {
        "calibration_points": [
            {"pixel": {"x": 0, "y": 0}, "world": {"easting": 0, "northing": 0}},
            {"pixel": {"x": 50, "y": 0}, "world": {"easting": 5, "northing": 0}},
            {"pixel": {"x": 100, "y": 0}, "world": {"easting": 10, "northing": 0}},
            {"pixel": {"x": 0, "y": 40}, "world": {"easting": 0, "northing": 4}},
            {"pixel": {"x": 50, "y": 40}, "world": {"easting": 5, "northing": 4}},
            {"pixel": {"x": 100, "y": 40}, "world": {"easting": 10, "northing": 4}},
        ]
    }
    (configs / "giti_calibration_points.json").write_text(json.dumps(orig_calib))


def test_main_end_to_end(tmp_path, monkeypatch):
    make_project_files(tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--project-root",
            str(tmp_path),
            "--grid-cols",
            "4",
            "--grid-rows",
            "3",
            "--n-splits",
            "2",
            "--save-prefix",
            "test_output",
        ],
    )

    H = np.eye(3, dtype=np.float32)
    mask = np.ones((6, 1), dtype=np.uint8)

    def fake_find_homography(*args, **kwargs):
        # Return H and mask with same length as input points
        n = len(args[0])
        return H, np.ones((n, 1), dtype=np.uint8)

    with (
        patch("cv2.findHomography", side_effect=fake_find_homography),
        patch("cv2.perspectiveTransform", side_effect=lambda pts, h: pts.reshape(-1, 2)),
        patch("matplotlib.pyplot.subplots", return_value=(MagicMock(), MagicMock())),
        patch("matplotlib.pyplot.show", MagicMock()),
        patch("matplotlib.pyplot.savefig", MagicMock()),
        patch("matplotlib.pyplot.colorbar", MagicMock()),
        patch("matplotlib.pyplot.tight_layout", MagicMock()),
        patch("numpy.save", MagicMock()),
        patch("src.bev.calibration.grid_validation_calibration.export_results") as export_mock,
    ):
        main()
    export_mock.assert_called_once()


def test_main_real_data_raises(tmp_path, monkeypatch):
    make_project_files(tmp_path)
    monkeypatch.setattr(sys, "argv", ["prog", "--project-root", str(tmp_path), "--real-data"])
    with pytest.raises(NotImplementedError):
        main()
