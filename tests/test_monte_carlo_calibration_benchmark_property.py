"""Property tests for pure helpers in monte_carlo_calibration_benchmark."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.bev.calibration import monte_carlo_calibration_benchmark as mc


def test_make_example_pose_shapes():
    R, t = mc.make_example_pose()
    assert isinstance(R, np.ndarray)
    assert isinstance(t, np.ndarray)
    assert R.shape == (3, 3)
    assert t.shape == (3, 1)


def test_make_example_pose_is_orthonormal():
    R, _ = mc.make_example_pose()
    eye = R.T @ R
    assert np.allclose(eye, np.eye(3), atol=1e-6)
    assert abs(np.linalg.det(R) - 1.0) < 1e-6


def test_make_example_pose_is_deterministic():
    R1, t1 = mc.make_example_pose()
    R2, t2 = mc.make_example_pose()
    assert np.allclose(R1, R2)
    assert np.allclose(t1, t2)


def test_camera_matrix_shape():
    assert mc.K.shape == (3, 3)
    assert mc.K[2, 2] == pytest.approx(1.0)
    assert mc.K[0, 0] > 0
    assert mc.K[1, 1] > 0


def test_distortion_coeffs_shape():
    assert mc.dist_coeffs.shape == (5,)


def test_grid_dimensions_match():
    # Grid is NX x NY points
    assert mc.XX.shape == (mc.NY, mc.NX)
    assert mc.YY.shape == (mc.NY, mc.NX)
    assert mc.ZW.shape == (mc.NY, mc.NX)
    assert mc.world_points_true.shape == (mc.NX * mc.NY, 3)


def test_grid_spans_declared_bounds():
    assert mc.XX.min() == pytest.approx(0.0)
    assert mc.XX.max() == pytest.approx(mc.W_X)
    assert mc.YY.min() == pytest.approx(0.0)
    assert mc.YY.max() == pytest.approx(mc.W_Y)


def test_world_points_are_z_zero():
    assert np.all(mc.ZW == 0.0)
    assert np.all(mc.world_points_true[:, 2] == 0.0)
