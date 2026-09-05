
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.bev.calibration.grid_validation_calibration import (
    generate_synthetic_grid,
    reprojection_errors,
)

# Generate arrays of points (N,2) or (N,3) with finite values
point2_st = st.lists(
    st.tuples(st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
              st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False)),
    min_size=1, max_size=20
).map(lambda lst: np.array(lst, dtype=np.float32))

@st.composite
def homography(draw):
    # Simple invertible homography: identity with optional translation
    tx = draw(st.floats(min_value=-10, max_value=10))
    ty = draw(st.floats(min_value=-10, max_value=10))
    H = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]], dtype=np.float32)
    return H

@given(point2_st, homography())
def test_reprojection_errors_shape_and_non_negative(pixel_points, H):
    world_points = np.copy(pixel_points)  # same shape (N,2)
    errs, projected = reprojection_errors(pixel_points, world_points, H)
    assert isinstance(errs, np.ndarray)
    assert isinstance(projected, np.ndarray)
    assert errs.shape == (len(pixel_points),)
    assert projected.shape == pixel_points.shape
    assert np.all(np.isfinite(errs))
    assert np.all(errs >= 0.0)

@given(point2_st)
def test_reprojection_errors_identity_zero(pixel_points):
    H = np.eye(3, dtype=np.float32)
    world_points = np.copy(pixel_points)
    errs, _ = reprojection_errors(pixel_points, world_points, H)
    assert np.allclose(errs, 0.0, atol=1e-6)

@given(st.integers(min_value=1, max_value=5),  # grid_rows
       st.integers(min_value=1, max_value=5),  # grid_cols
       st.floats(min_value=0.0, max_value=100.0),  # x_min_px
       st.floats(min_value=100.0, max_value=200.0),  # x_max_px
       st.floats(min_value=0.0, max_value=100.0),  # y_min_px
       st.floats(min_value=100.0, max_value=200.0),  # y_max_px
       st.floats(min_value=0.1, max_value=10.0),  # ppm_x
       st.floats(min_value=0.1, max_value=10.0),  # ppm_y
       st.floats(min_value=0.0, max_value=0.0))  # noise_std = 0
def test_generate_synthetic_grid_no_noise(grid_rows, grid_cols,
                                          x_min_px, x_max_px,
                                          y_min_px, y_max_px,
                                          ppm_x, ppm_y, noise_std):
    # Ensure x_max > x_min and y_max > y_min
    if x_max_px <= x_min_px:
        x_max_px = x_min_px + 1.0
    if y_max_px <= y_min_px:
        y_max_px = y_min_px + 1.0

    rng = np.random.default_rng(12345)
    pixel_pts, world_pts_noisy, world_pts_true = generate_synthetic_grid(
        grid_rows, grid_cols, x_min_px, x_max_px, y_min_px, y_max_px,
        ppm_x, ppm_y, noise_std, rng
    )
    n = grid_rows * grid_cols
    assert pixel_pts.shape == (n, 2)
    assert world_pts_noisy.shape == (n, 2)
    assert world_pts_true.shape == (n, 2)
    assert np.allclose(world_pts_noisy, world_pts_true, atol=1e-5)  # no noise

    # Check true world offsets
    expected_x = (pixel_pts[:, 0] - x_min_px) / ppm_x
    expected_y = (pixel_pts[:, 1] - y_min_px) / ppm_y
    assert np.allclose(world_pts_true[:, 0], expected_x, atol=1e-5)
    assert np.allclose(world_pts_true[:, 1], expected_y, atol=1e-5)
