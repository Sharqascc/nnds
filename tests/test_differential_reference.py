"""Differential tests: optimised path vs naive reference."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.differential


def _naive_cell(px, py, x_min, x_max, y_min, y_max, cell_size):
    if not (x_min <= px < x_max and y_min <= py < y_max):
        return "OUT_OF_BOUNDS"
    col = int((px - x_min) // cell_size)
    row = int((py - y_min) // cell_size)
    return f"G_{chr(65 + col)}_{row + 1}"


def test_spatial_grid_matches_naive_reference():
    from src.analysis.grid_trajectory.spatial_grid import SpatialGrid

    cfg = {
        "corners": {
            "top_left": [0, 0],
            "top_right": [200, 0],
            "bottom_left": [0, 200],
            "bottom_right": [200, 200],
        },
        "configuration": {"cell_size": 40, "naming_style": "G_{col}_{row}"},
    }
    tmp = Path(tempfile.mkdtemp()) / "g.json"
    tmp.write_text(json.dumps(cfg))
    grid = SpatialGrid(tmp)

    import numpy as np

    rng = np.random.default_rng(0)
    for _ in range(500):
        px = float(rng.uniform(-10, 210))
        py = float(rng.uniform(-10, 210))
        got = grid.get_cell_from_pixels(px, py)
        want = _naive_cell(px, py, 0, 200, 0, 200, 40)
        if 0 < px < 200 and 0 < py < 200:
            assert got == want, f"({px:.2f},{py:.2f}): grid={got} naive={want}"


def test_iou_matches_naive_reference():
    from src.analysis.detection_metrics import iou

    def naive(a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        u = aa + ab - inter
        return inter / u if u > 0 else 0.0

    import numpy as np

    rng = np.random.default_rng(1)
    for _ in range(1000):
        x1a, x2a = sorted(rng.uniform(0, 100, 2))
        y1a, y2a = sorted(rng.uniform(0, 100, 2))
        x1b, x2b = sorted(rng.uniform(0, 100, 2))
        y1b, y2b = sorted(rng.uniform(0, 100, 2))
        a = (float(x1a), float(y1a), float(x2a) + 1e-9, float(y2a) + 1e-9)
        b = (float(x1b), float(y1b), float(x2b) + 1e-9, float(y2b) + 1e-9)
        assert iou(a, b) == pytest.approx(naive(a, b), abs=1e-12)


def test_pet_grid_matches_standalone_reference():
    """Non-degenerate cases: grid and standalone interval PET agree."""
    from pet_interval import compute_pet_from_intervals
    from src.analysis.grid_trajectory.pet_grid import (
        Interval,
        WorldSample,
        compute_pet,
    )

    cases = [
        (0.0, 1.0, 2.0, 3.0),  # A then B
        (2.0, 3.0, 0.0, 1.0),  # B then A
        (0.0, 2.0, 1.0, 3.0),  # overlap
    ]
    ws = [WorldSample(t=0.0, x=0.0, y=0.0)]
    for a_e, a_x, b_e, b_x in cases:
        a = Interval(obj_id=1, cell_id="G", t_enter=a_e, t_exit=a_x, world_samples=ws)
        b = Interval(obj_id=2, cell_id="G", t_enter=b_e, t_exit=b_x, world_samples=ws)
        grid_events = compute_pet([a, b], pet_threshold=100.0)
        standalone = compute_pet_from_intervals(a_e, a_x, b_e, b_x)

        if standalone.pet_status == "overlap":
            assert len(grid_events) == 0
        else:
            assert len(grid_events) == 1
            assert grid_events[0].pet == pytest.approx(standalone.pet_s)


def test_pet_grid_and_interval_diverge_on_zero_gap_by_design():
    """Documented divergence: the two PET implementations disagree on
    zero-gap (t_exit_a == t_enter_b).

    pet_grid.compute_pet: strict `0.0 < pet`, so zero-gap produces no event.
    pet_interval.compute_pet_from_intervals: treats zero-gap as sequential
      with pet_s=0.0.

    Neither is wrong in isolation; they were written for different
    consumers. This test pins both behaviors so that if either changes
    the difference is visible. Unifying them is a semantic decision, not
    a bug fix; see docs/VALIDATION.md.
    """
    from pet_interval import compute_pet_from_intervals
    from src.analysis.grid_trajectory.pet_grid import (
        Interval,
        WorldSample,
        compute_pet,
    )

    ws = [WorldSample(t=0.0, x=0.0, y=0.0)]
    a = Interval(obj_id=1, cell_id="G", t_enter=0.0, t_exit=1.0, world_samples=ws)
    b = Interval(obj_id=2, cell_id="G", t_enter=1.0, t_exit=2.0, world_samples=ws)

    grid_events = compute_pet([a, b], pet_threshold=100.0)
    assert len(grid_events) == 0, "pet_grid excludes zero-gap PET (0.0 < pet)"

    standalone = compute_pet_from_intervals(0.0, 1.0, 1.0, 2.0)
    assert standalone.pet_status == "sequential"
    assert standalone.pet_s == pytest.approx(0.0)
    assert standalone.first_actor == "a"
    assert standalone.second_actor == "b"
