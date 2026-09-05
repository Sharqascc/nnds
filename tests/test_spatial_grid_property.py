
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis.grid_trajectory.spatial_grid import (
    SpatialGrid,
    _col_to_letters,
    _letters_to_col,
)


def make_grid():
    tmpdir = tempfile.TemporaryDirectory()
    config = {
        "corners": {
            "top_left": [0, 0],
            "top_right": [200, 0],
            "bottom_left": [0, 200],
            "bottom_right": [200, 200],
        },
        "configuration": {
            "cell_size": 40,
            "naming_style": "G_{col}_{row}",
        },
    }
    config_path = Path(tmpdir.name) / "grid_config.json"
    config_path.write_text(json.dumps(config))
    grid = SpatialGrid(config_path)
    return grid, tmpdir

@given(st.integers(min_value=0, max_value=1000))
def test_col_to_letters_roundtrip(col_idx):
    letters = _col_to_letters(col_idx)
    assert letters.isalpha()
    assert _letters_to_col(letters) == col_idx

@given(st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=1, max_size=3))
def test_letters_to_col_roundtrip(letters):
    col_idx = _letters_to_col(letters)
    assert _col_to_letters(col_idx) == letters

@given(st.integers(min_value=0, max_value=199), st.integers(min_value=0, max_value=199))
@settings(deadline=None)
def test_get_cell_from_pixels_inside_grid(x, y):
    grid, tmpdir = make_grid()
    try:
        cell = grid.get_cell_from_pixels(x, y)
        assert cell != "OUT_OF_BOUNDS"
        assert cell.startswith("G_")
    finally:
        tmpdir.cleanup()

@given(st.integers(min_value=-100, max_value=299), st.integers(min_value=-100, max_value=299))
@settings(deadline=None)
def test_get_cell_from_pixels_returns_string(x, y):
    grid, tmpdir = make_grid()
    try:
        cell = grid.get_cell_from_pixels(x, y)
        assert isinstance(cell, str)
    finally:
        tmpdir.cleanup()
