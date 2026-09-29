import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis.grid_overlay import (
    OUT_OF_BOUNDS,
    GridDims,
    cell_bounds_px,
    cell_name,
    grid_from_config,
    parse_cell_name,
    pixel_to_cell,
)

STYLE = "CELL_{col}_{row}"


def _dims() -> GridDims:
    return grid_from_config(
        {
            "corners": {
                "top_left": [0, 0],
                "top_right": [1600, 0],
                "bottom_left": [0, 720],
                "bottom_right": [1600, 720],
            },
            "configuration": {"cell_size": 50, "naming_style": STYLE},
        }
    )


@given(
    st.integers(0, 31),
    st.integers(1, 14),
)
@settings(max_examples=100)
def test_name_roundtrip(col, row):
    name = cell_name(col, row, STYLE)
    assert parse_cell_name(name, STYLE) == (col, row)


@given(
    st.floats(0, 1599.999, allow_nan=False, allow_infinity=False),
    st.floats(0, 719.999, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_in_bounds_pixel_never_returns_oob(px, py):
    d = _dims()
    name = pixel_to_cell(px, py, d)
    assert name != OUT_OF_BOUNDS


@given(
    st.floats(1600, 5000, allow_nan=False, allow_infinity=False),
    st.floats(0, 719.999, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50)
def test_x_out_of_bounds(px, py):
    d = _dims()
    assert pixel_to_cell(px, py, d) == OUT_OF_BOUNDS


@given(
    st.integers(0, 31),
    st.integers(1, 14),
)
@settings(max_examples=50)
def test_cell_bounds_inside_frame(col, row):
    d = _dims()
    x1, y1, x2, y2 = cell_bounds_px(col, row, d)
    assert 0 <= x1 < x2 <= 1600
    assert 0 <= y1 < y2 <= 720
    assert x2 - x1 == d.cell_size
    assert y2 - y1 == d.cell_size


@given(
    st.integers(0, 31),
    st.integers(1, 14),
)
@settings(max_examples=50)
def test_cell_center_maps_back_to_same_cell(col, row):
    d = _dims()
    x1, y1, x2, y2 = cell_bounds_px(col, row, d)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    assert pixel_to_cell(cx, cy, d) == cell_name(col, row, STYLE)


@given(
    st.integers(0, 31),
    st.integers(1, 14),
    st.floats(0.0, 0.99, allow_nan=False, allow_infinity=False),
    st.floats(0.0, 0.99, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_every_interior_pixel_inside_bounds_cell(col, row, fx, fy):
    d = _dims()
    x1, y1, x2, y2 = cell_bounds_px(col, row, d)
    # point strictly inside the cell (up to cell_size-1 offset)
    px = x1 + fx * (x2 - x1 - 1e-6)
    py = y1 + fy * (y2 - y1 - 1e-6)
    assert pixel_to_cell(px, py, d) == cell_name(col, row, STYLE)
