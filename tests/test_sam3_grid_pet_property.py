"""Property tests for validators in src.analysis.grid_trajectory.sam3_grid_pet."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.analysis.grid_trajectory import sam3_grid_pet as sg


@given(c=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False))
def test_validate_conf_accepts_unit_interval(c):
    sg._validate_conf(c)


@given(
    c=st.one_of(
        st.floats(max_value=-1e-9, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.0 + 1e-9, allow_nan=False, allow_infinity=False),
    )
)
def test_validate_conf_rejects_outside_unit_interval(c):
    with pytest.raises(ValueError):
        sg._validate_conf(c)


def test_validate_conf_accepts_boundaries():
    sg._validate_conf(0.0)
    sg._validate_conf(1.0)


def test_validate_conf_rejects_nan():
    with pytest.raises(ValueError):
        sg._validate_conf(float("nan"))


@given(n=st.integers(1, 10_000))
def test_validate_max_frames_accepts_positive(n):
    sg._validate_max_frames(n)


def test_validate_max_frames_accepts_none():
    sg._validate_max_frames(None)


@given(n=st.integers(-10_000, 0))
def test_validate_max_frames_rejects_non_positive(n):
    with pytest.raises(ValueError):
        sg._validate_max_frames(n)


@given(n=st.integers(1, 1000))
def test_validate_frame_stride_accepts_positive(n):
    sg._validate_frame_stride(n)


@given(n=st.integers(-1000, 0))
def test_validate_frame_stride_rejects_non_positive(n):
    with pytest.raises(ValueError):
        sg._validate_frame_stride(n)


def test_validate_bev_config_accepts_full_config():
    sg._validate_bev_config({
        "H_pixel_to_world": [[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]],
        "bev_bounds": [0.0, 0.0, 100.0, 100.0],
        "bev_resolution": [1000, 800],
    })


def test_validate_bev_config_rejects_empty():
    with pytest.raises(KeyError):
        sg._validate_bev_config({})


def test_validate_bev_config_rejects_partial():
    with pytest.raises(KeyError):
        sg._validate_bev_config({"H_pixel_to_world": []})
