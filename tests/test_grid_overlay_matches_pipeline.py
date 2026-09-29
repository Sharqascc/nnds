"""Regression: grid_overlay.pixel_to_cell must match SpatialGrid's numbering.

The pipeline (src/analysis/grid_trajectory/spatial_grid.py) uses 1-based rows.
An earlier version of grid_overlay used 0-based rows, so review-strip overlays
highlighted a cell one row off from the pipeline's own grid_cell column.
This test pins the two implementations together.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analysis.grid_overlay import GridDims, grid_from_config, pixel_to_cell
from src.analysis.grid_trajectory.spatial_grid import SpatialGrid

REPO = Path(__file__).resolve().parents[1]
GRID_CONFIG = REPO / "configs" / "GITI_grid_config.json"


@pytest.fixture(scope="module")
def pipeline_grid() -> SpatialGrid:
    return SpatialGrid(str(GRID_CONFIG))


@pytest.fixture(scope="module")
def overlay_grid() -> GridDims:
    return grid_from_config(json.loads(GRID_CONFIG.read_text()))


SAMPLE_POINTS = [
    (0, 0),
    (25, 25),
    (75, 75),
    (100, 100),
    (150, 175),
    (200, 300),
    (400, 400),
    (751, 221),
    (1065, 174),
    (1100, 200),
    (1200, 500),
    (1500, 700),
    (1599, 719),
    (500, 50),
    (50, 500),
]


@pytest.mark.parametrize("pt", SAMPLE_POINTS)
def test_overlay_matches_pipeline(pipeline_grid, overlay_grid, pt):
    sg_cell = pipeline_grid.get_cell_from_pixels(*pt)
    ov_cell = pixel_to_cell(pt[0], pt[1], overlay_grid)
    assert sg_cell == ov_cell, f"mismatch at {pt}: pipeline={sg_cell} overlay={ov_cell}"


def test_regression_p6_event0_known_coordinate(pipeline_grid, overlay_grid):
    """A coordinate the pipeline recently produced for event 0."""
    x, y = 1065.1, 174.0
    assert pipeline_grid.get_cell_from_pixels(x, y) == pixel_to_cell(x, y, overlay_grid)
