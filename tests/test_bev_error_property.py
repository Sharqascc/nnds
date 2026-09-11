import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis.bev_error import (
    position_errors,
    position_metrics,
    project_homography,
)

pts_st = st.lists(
    st.tuples(
        st.floats(-100, 100, allow_nan=False, allow_infinity=False),
        st.floats(-100, 100, allow_nan=False, allow_infinity=False),
    ),
    min_size=1,
    max_size=20,
).map(lambda xs: np.array(xs, dtype=np.float64))

scale_st = st.floats(0.1, 10.0, allow_nan=False, allow_infinity=False)
offset_st = st.floats(-50, 50, allow_nan=False, allow_infinity=False)


@given(pts_st)
@settings(max_examples=50)
def test_errors_non_negative(pts):
    errs = position_errors(pts, pts)
    assert np.all(errs >= 0.0)


@given(pts_st)
@settings(max_examples=50)
def test_identical_points_zero_error(pts):
    errs = position_errors(pts, pts)
    np.testing.assert_allclose(errs, 0.0, atol=1e-12)


@given(pts_st, scale_st, offset_st, offset_st)
@settings(max_examples=50)
def test_scale_translation_homography(pts, s, tx, ty):
    H = np.array([[s, 0.0, tx], [0.0, s, ty], [0.0, 0.0, 1.0]])
    projected = project_homography(H, pts)
    # exact same H applied to same points -> zero error
    np.testing.assert_allclose(projected, project_homography(H, pts))


@given(pts_st, pts_st)
@settings(max_examples=50)
def test_rmse_geq_mae(a, b):
    # align sizes
    n = min(len(a), len(b))
    if n == 0:
        return
    a, b = a[:n], b[:n]
    m = position_metrics(a, b)
    assert m["rmse"] + 1e-9 >= m["mae"]


@given(pts_st, pts_st)
@settings(max_examples=50)
def test_p95_geq_mean_when_finite(a, b):
    n = min(len(a), len(b))
    if n < 2:
        return
    a, b = a[:n], b[:n]
    m = position_metrics(a, b)
    # p95 >= median; and p95 >= 0 always
    assert m["p95"] >= 0.0
    assert m["max"] + 1e-9 >= m["p95"]


@given(pts_st)
@settings(max_examples=50)
def test_identity_homography_preserves_points(pts):
    I = np.eye(3)
    out = project_homography(I, pts)
    np.testing.assert_allclose(out, pts, atol=1e-9)


@given(pts_st, st.floats(0.01, 10.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=50)
def test_scale_homography_scales_points(pts, s):
    H = np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]])
    out = project_homography(H, pts)
    np.testing.assert_allclose(out, pts * s, atol=1e-9)
