import json

import numpy as np
import pytest

from src.analysis.grid_trajectory.spatial_grid import (
    OUT_OF_BOUNDS_CELL,
    SpatialGrid,
    _col_to_letters,
    _letters_to_col,
)


def make_config(tmp_path, **overrides):
    config = {
        "corners": {
            "top_left": [0, 0],
            "top_right": [100, 0],
            "bottom_left": [0, 100],
            "bottom_right": [100, 100],
        },
        "configuration": {
            "cell_size": 20,
            "naming_style": "G_{col}_{row}",
        },
    }
    # Apply overrides
    for key, value in overrides.items():
        if key in config:
            if isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
        else:
            config[key] = value
    path = tmp_path / "grid_config.json"
    path.write_text(json.dumps(config))
    return path


def test_col_to_letters_single():
    assert _col_to_letters(0) == "A"
    assert _col_to_letters(25) == "Z"


def test_col_to_letters_multi():
    assert _col_to_letters(26) == "AA"
    assert _col_to_letters(27) == "AB"
    assert _col_to_letters(51) == "AZ"


def test_letters_to_col():
    assert _letters_to_col("A") == 0
    assert _letters_to_col("Z") == 25
    assert _letters_to_col("AA") == 26
    assert _letters_to_col("AB") == 27


def test_constructor_missing_file():
    with pytest.raises(FileNotFoundError):
        SpatialGrid("/nonexistent/path.json")


def test_constructor_invalid_corners_type(tmp_path):
    path = tmp_path / "grid_config.json"
    path.write_text(json.dumps({"corners": "not_dict", "configuration": {}}))
    with pytest.raises(ValueError):
        SpatialGrid(path)


def test_constructor_missing_corner_keys(tmp_path):
    cfg = {"corners": {"top_left": [0, 0]}, "configuration": {}}
    path = tmp_path / "grid_config.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(KeyError):
        SpatialGrid(path)


def test_constructor_invalid_cell_size(tmp_path):
    path = make_config(tmp_path, configuration={"cell_size": 0, "naming_style": "G_{col}_{row}"})
    with pytest.raises(ValueError):
        SpatialGrid(path)


def test_constructor_invalid_naming_style(tmp_path):
    path = make_config(tmp_path, configuration={"cell_size": 20, "naming_style": "G"})
    with pytest.raises(ValueError):
        SpatialGrid(path)


def test_constructor_valid(tmp_path):
    path = make_config(tmp_path)
    grid = SpatialGrid(path)
    assert grid.cell_size == 20
    assert grid.x_min == 0
    assert grid.x_max == 100
    assert grid.y_min == 0
    assert grid.y_max == 100


def test_get_cell_from_pixels_out_of_bounds():
    # Use a valid grid
    # We need to create a grid without calling constructor? Better use a temp config.
    import tempfile

    cfg = {
        "corners": {
            "top_left": [0, 0],
            "top_right": [100, 0],
            "bottom_left": [0, 100],
            "bottom_right": [100, 100],
        },
        "configuration": {"cell_size": 20, "naming_style": "G_{col}_{row}"},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        f.flush()
        grid = SpatialGrid(f.name)
    assert grid.get_cell_from_pixels(-1, 50) == OUT_OF_BOUNDS_CELL
    assert grid.get_cell_from_pixels(101, 50) == OUT_OF_BOUNDS_CELL
    assert grid.get_cell_from_pixels(50, -1) == OUT_OF_BOUNDS_CELL
    assert grid.get_cell_from_pixels(50, 101) == OUT_OF_BOUNDS_CELL


def test_get_cell_from_pixels_valid():
    cfg = {
        "corners": {
            "top_left": [0, 0],
            "top_right": [100, 0],
            "bottom_left": [0, 100],
            "bottom_right": [100, 100],
        },
        "configuration": {"cell_size": 20, "naming_style": "G_{col}_{row}"},
    }
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        f.flush()
        grid = SpatialGrid(f.name)
    # At (10,10) -> col 0, row 0 => G_A_1
    assert grid.get_cell_from_pixels(10, 10) == "G_A_1"
    # At (30, 30) -> col 1, row 1 => G_B_2
    assert grid.get_cell_from_pixels(30, 30) == "G_B_2"
    # Boundary x=100, y=100 (max) -> col 5, row 5 => G_F_6? x_max=100, cell_size=20 => (100-0)//20=5; same for row
    assert grid.get_cell_from_pixels(100, 100) == "G_F_6"


def test_get_cell_center_cache_and_valid():
    cfg = {
        "corners": {
            "top_left": [0, 0],
            "top_right": [100, 0],
            "bottom_left": [0, 100],
            "bottom_right": [100, 100],
        },
        "configuration": {"cell_size": 20, "naming_style": "G_{col}_{row}"},
    }
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        f.flush()
        grid = SpatialGrid(f.name)
    # Valid cell
    center = grid.get_cell_center("G_B_3")
    assert center == (30, 50)  # col_idx=1 -> x=0+20+10=30, row_idx=2 -> y=0+40+10=50
    # Test cache
    assert grid.get_cell_center("G_B_3") == center
    # Malformed
    assert grid.get_cell_center("bad") is None
    assert grid.get_cell_center(123) is None
    # Non-alpha col
    assert grid.get_cell_center("G_1_2") is None
    # Negative col/row
    assert grid.get_cell_center("G_A_0") is None


def test_get_cell_bounds():
    cfg = {
        "corners": {
            "top_left": [0, 0],
            "top_right": [100, 0],
            "bottom_left": [0, 100],
            "bottom_right": [100, 100],
        },
        "configuration": {"cell_size": 20, "naming_style": "G_{col}_{row}"},
    }
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        f.flush()
        grid = SpatialGrid(f.name)
    bounds = grid.get_cell_bounds("G_B_3")
    assert bounds == (20, 40, 40, 60)  # center (30,50) half=10 -> x1=20, y1=40, x2=40, y2=60


def test_get_stats():
    cfg = {
        "corners": {
            "top_left": [0, 0],
            "top_right": [100, 0],
            "bottom_left": [0, 100],
            "bottom_right": [100, 100],
        },
        "configuration": {"cell_size": 20, "naming_style": "G_{col}_{row}"},
    }
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        f.flush()
        grid = SpatialGrid(f.name)
    stats = grid.get_stats()
    assert stats["cell_size"] == 20
    assert stats["n_cols"] == 5
    assert stats["n_rows"] == 5
    assert stats["total_cells"] == 25
    assert stats["x_range"] == (0, 100)
    assert stats["y_range"] == (0, 100)


def test_draw_overlay_no_highlight():
    cfg = {
        "corners": {
            "top_left": [0, 0],
            "top_right": [100, 0],
            "bottom_left": [0, 100],
            "bottom_right": [100, 100],
        },
        "configuration": {"cell_size": 20, "naming_style": "G_{col}_{row}"},
    }
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        f.flush()
        grid = SpatialGrid(f.name)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = grid.draw_overlay(frame)
    assert result.shape == frame.shape
    assert isinstance(result, np.ndarray)


def test_draw_overlay_with_highlight():
    cfg = {
        "corners": {
            "top_left": [0, 0],
            "top_right": [100, 0],
            "bottom_left": [0, 100],
            "bottom_right": [100, 100],
        },
        "configuration": {"cell_size": 20, "naming_style": "G_{col}_{row}"},
    }
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        f.flush()
        grid = SpatialGrid(f.name)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = grid.draw_overlay(frame, highlight_cells=["G_B_3"])
    assert result.shape == frame.shape


def test_repr():
    cfg = {
        "corners": {
            "top_left": [0, 0],
            "top_right": [100, 0],
            "bottom_left": [0, 100],
            "bottom_right": [100, 100],
        },
        "configuration": {"cell_size": 20, "naming_style": "G_{col}_{row}"},
    }
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        f.flush()
        grid = SpatialGrid(f.name)
    r = repr(grid)
    assert "SpatialGrid" in r
    assert "cell_size=20" in r


def test_get_cell_center_exception():
    """Cover ValueError/IndexError exception (lines 183-184)."""
    import tempfile

    cfg = {
        "corners": {
            "top_left": [0, 0],
            "top_right": [100, 0],
            "bottom_left": [0, 100],
            "bottom_right": [100, 100],
        },
        "configuration": {"cell_size": 20, "naming_style": "G_{col}_{row}"},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        f.flush()
        grid = SpatialGrid(f.name)
    # "G_A_X" causes int("X") to raise ValueError inside get_cell_center
    assert grid.get_cell_center("G_A_X") is None


def test_get_cell_bounds_none():
    """Cover get_cell_bounds returning None when center is None (line 197)."""
    import tempfile

    cfg = {
        "corners": {
            "top_left": [0, 0],
            "top_right": [100, 0],
            "bottom_left": [0, 100],
            "bottom_right": [100, 100],
        },
        "configuration": {"cell_size": 20, "naming_style": "G_{col}_{row}"},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        f.flush()
        grid = SpatialGrid(f.name)
    assert grid.get_cell_bounds("bad") is None


def test_draw_overlay_highlight_none():
    """Cover draw_overlay continue when bounds is None (line 277)."""
    import tempfile

    cfg = {
        "corners": {
            "top_left": [0, 0],
            "top_right": [100, 0],
            "bottom_left": [0, 100],
            "bottom_right": [100, 100],
        },
        "configuration": {"cell_size": 20, "naming_style": "G_{col}_{row}"},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        f.flush()
        grid = SpatialGrid(f.name)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = grid.draw_overlay(frame, highlight_cells=["bad"])
    assert result.shape == frame.shape
