import numpy as np
import pytest

from src.analysis.bev_error import (
    homography_error_metrics,
    position_errors,
    position_metrics,
    project_homography,
)

IDENTITY = np.eye(3, dtype=np.float64)


def test_project_identity():
    pts = np.array([[0.0, 0.0], [10.0, 20.0], [100.0, 200.0]])
    out = project_homography(IDENTITY, pts)
    np.testing.assert_allclose(out, pts)


def test_project_rejects_bad_shapes():
    with pytest.raises(ValueError):
        project_homography(np.eye(2), np.zeros((3, 2)))
    with pytest.raises(ValueError):
        project_homography(IDENTITY, np.zeros((3, 3)))


def test_project_rejects_point_at_infinity():
    H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        project_homography(H, np.array([[0.0, 0.0]]))


def test_position_errors_zero_for_identical():
    pts = np.array([[1.0, 2.0], [3.0, 4.0]])
    errs = position_errors(pts, pts)
    np.testing.assert_allclose(errs, 0.0)


def test_position_errors_known_values():
    a = np.array([[0.0, 0.0], [3.0, 0.0]])
    b = np.array([[0.0, 0.0], [0.0, 4.0]])
    errs = position_errors(a, b)
    np.testing.assert_allclose(errs, [0.0, 5.0])


def test_position_metrics_empty():
    m = position_metrics(np.zeros((0, 2)), np.zeros((0, 2)))
    assert m == {"mae": 0.0, "rmse": 0.0, "p95": 0.0, "max": 0.0, "n": 0}


def test_position_metrics_simple():
    a = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    b = np.zeros((4, 2))
    m = position_metrics(a, b)
    # errors: [0, 3, 0, 0] -> mae = 0.75, rmse = sqrt(9/4) = 1.5
    assert m["mae"] == pytest.approx(0.75)
    assert m["rmse"] == pytest.approx(1.5)
    assert m["max"] == pytest.approx(3.0)
    assert m["n"] == 4


def test_position_metrics_shape_mismatch():
    with pytest.raises(ValueError):
        position_errors(np.zeros((3, 2)), np.zeros((4, 2)))


def test_homography_error_identity():
    pixel = np.array([[0.0, 0.0], [100.0, 100.0]])
    world = pixel.copy()
    m = homography_error_metrics(IDENTITY, pixel, world)
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["p95"] == 0.0


def test_homography_error_scaled():
    # H scales by 2x; GT is 2x pixel => zero error
    H = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    pixel = np.array([[1.0, 1.0], [10.0, 20.0]])
    world = pixel * 2.0
    m = homography_error_metrics(H, pixel, world)
    assert m["mae"] == pytest.approx(0.0, abs=1e-9)


def test_homography_error_constant_offset():
    # H shifts x by +1 m; GT is unshifted => 1 m error per point
    H = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    pixel = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    world = pixel.copy()
    m = homography_error_metrics(H, pixel, world)
    assert m["mae"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(1.0)
    assert m["p95"] == pytest.approx(1.0)
