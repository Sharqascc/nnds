import numpy as np
import pytest

from src.analysis.grid_overlay import (
    OUT_OF_BOUNDS,
    GridDims,
    cell_bounds_px,
    cell_center_px,
    cell_name,
    draw_grid_lines,
    grid_from_config,
    highlight_cell,
    parse_cell_name,
    pixel_to_cell,
)

GITI_CFG = {
    "corners": {
        "top_left": [0, 0],
        "top_right": [1600, 0],
        "bottom_left": [0, 720],
        "bottom_right": [1600, 720],
    },
    "configuration": {"cell_size": 50, "naming_style": "CELL_{col}_{row}"},
}


def _dims() -> GridDims:
    return grid_from_config(GITI_CFG)


def test_grid_from_config_basic():
    d = _dims()
    assert d.x_min == 0 and d.x_max == 1600
    assert d.y_min == 0 and d.y_max == 720
    assert d.cell_size == 50
    assert d.n_cols == 32
    assert d.n_rows == 14


def test_grid_from_config_invalid_corners():
    bad = dict(GITI_CFG)
    bad["corners"] = {**GITI_CFG["corners"], "top_right": [0, 0]}
    with pytest.raises(ValueError):
        grid_from_config(bad)


def test_grid_from_config_invalid_cell_size():
    bad = dict(GITI_CFG)
    bad["configuration"] = {"cell_size": 0, "naming_style": "CELL_{col}_{row}"}
    with pytest.raises(ValueError):
        grid_from_config(bad)


def test_cell_name_letters():
    assert cell_name(0, 1, "CELL_{col}_{row}") == "CELL_A_1"
    assert cell_name(1, 3, "CELL_{col}_{row}") == "CELL_B_3"
    assert cell_name(25, 1, "CELL_{col}_{row}") == "CELL_Z_1"
    assert cell_name(26, 1, "CELL_{col}_{row}") == "CELL_AA_1"


def test_parse_cell_name_roundtrip():
    style = "CELL_{col}_{row}"
    for c in (0, 1, 25, 26, 31):
        for r in (1, 4, 14):
            name = cell_name(c, r, style)
            assert parse_cell_name(name, style) == (c, r)


def test_parse_cell_name_invalid():
    assert parse_cell_name("GARBAGE", "CELL_{col}_{row}") is None
    assert parse_cell_name("CELL_X", "CELL_{col}_{row}") is None


def test_pixel_to_cell_origin():
    d = _dims()
    assert pixel_to_cell(0, 0, d) == "CELL_A_1"


def test_pixel_to_cell_mid():
    d = _dims()
    # x=75 is in col 1 (50..100), y=120 is in row 2 (100..150)
    assert pixel_to_cell(75, 120, d) == "CELL_B_3"


def test_pixel_to_cell_out_of_bounds():
    d = _dims()
    assert pixel_to_cell(-1, 0, d) == OUT_OF_BOUNDS
    assert pixel_to_cell(1600, 0, d) == OUT_OF_BOUNDS
    assert pixel_to_cell(0, 720, d) == OUT_OF_BOUNDS


def test_cell_bounds_px():
    d = _dims()
    assert cell_bounds_px(0, 1, d) == (0, 0, 50, 50)
    assert cell_bounds_px(1, 3, d) == (50, 100, 100, 150)


def test_cell_center_px():
    d = _dims()
    assert cell_center_px(0, 1, d) == (25, 25)
    assert cell_center_px(1, 3, d) == (75, 125)


def test_draw_grid_lines_copies_frame():
    d = _dims()
    frame = np.zeros((720, 1600, 3), dtype=np.uint8)
    out = draw_grid_lines(frame, d, color=(200, 200, 200), thickness=1)
    assert out is not frame
    assert frame.sum() == 0  # original untouched
    assert out.sum() > 0  # something drawn


def test_highlight_cell_visible_inside():
    d = _dims()
    frame = np.zeros((720, 1600, 3), dtype=np.uint8)
    out = highlight_cell(frame, "CELL_B_3", d, color=(0, 255, 255), alpha=0.5)
    # inside the cell -> nonzero
    x1, y1, x2, y2 = cell_bounds_px(1, 3, d)
    assert out[y1:y2, x1:x2].sum() > 0
    # outside -> zero
    assert out[0:50, 200:300].sum() == 0


def test_highlight_cell_out_of_bounds_no_change():
    d = _dims()
    frame = np.zeros((720, 1600, 3), dtype=np.uint8)
    out = highlight_cell(frame, "CELL_ZZ_99", d, alpha=0.5)
    assert out.sum() == 0
