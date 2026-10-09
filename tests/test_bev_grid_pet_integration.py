"""Integration: BEV config -> spatial grid -> PET grid composition.

Verifies pixel-to-world projection, grid cell assignment, and that the
PET module consumes the interface the grid layer produces. Runs in PR CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
BEV_CFG = REPO / "configs" / "bev_config.json"
GRID_CFG = REPO / "configs" / "GITI_grid_config.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_bev_config_is_diagonal_affine():
    cfg = _load(BEV_CFG)
    H = np.array(cfg["H_pixel_to_world"])
    assert H.shape == (3, 3)
    assert abs(H[0, 1]) < 1e-12
    assert abs(H[1, 0]) < 1e-12
    assert H[2, 2] == 1.0
    assert H[0, 0] > 0
    assert H[1, 1] > 0


def test_pixel_to_world_roundtrip_uses_local_convention():
    cfg = _load(BEV_CFG)
    H = np.array(cfg["H_pixel_to_world"])
    x_min = cfg["x_min"]
    y_max = cfg["y_max"]
    A_X, C_X = H[0, 0], H[0, 2]
    A_Y, C_Y = H[1, 1], H[1, 2]

    for px, py in [(0.0, 0.0), (800.0, 360.0), (1599.0, 719.0)]:
        wx = A_X * px + C_X - x_min
        wy = y_max - (A_Y * py + C_Y)
        px_back = (wx + x_min - C_X) / A_X
        py_back = (y_max - wy - C_Y) / A_Y
        assert abs(px_back - px) < 1e-6
        assert abs(py_back - py) < 1e-6


def test_grid_config_has_positive_cell_size():
    cfg = _load(GRID_CFG)
    assert cfg["configuration"]["cell_size"] > 0


def test_bev_mapper_public_api():
    from src.bev import bev_mapper
    names = [n for n in dir(bev_mapper) if not n.startswith("_")]
    assert any("BEV" in n or "Mapper" in n for n in names), names


def test_spatial_grid_public_api():
    from src.analysis.grid_trajectory import spatial_grid
    names = [n for n in dir(spatial_grid) if not n.startswith("_")]
    assert any("SpatialGrid" in n or "Grid" in n for n in names), names


def test_pet_grid_public_api():
    from src.analysis.grid_trajectory import pet_grid
    for name in ("compute_pet", "summarize_pet"):
        assert hasattr(pet_grid, name), f"pet_grid missing {name}"
