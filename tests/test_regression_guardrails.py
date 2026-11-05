"""Regression guardrails for paper-critical invariants.

Each test codifies one invariant the published numbers depend on.
A failure means the tree no longer reproduces the paper.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BEV_CFG = REPO / "configs" / "bev_config.json"
GRID_CFG = REPO / "configs" / "GITI_grid_config.json"
GRID_PET_SRC = REPO / "src" / "analysis" / "grid_trajectory" / "uvh_coco_fused_grid_pet.py"
RAW_CSV = REPO / "outputs" / "giti_raw.csv"
FROZEN_SUMMARY = REPO / "outputs" / "final_screened_summary.json"


def test_pet_threshold_is_enforced_in_source():
    """The threshold gate must appear in the source.

    Regression: if someone removes or bypasses the `pet <= pet_threshold`
    comparison, events above the threshold silently enter the output
    and every count in the paper becomes wrong.
    """
    src = GRID_PET_SRC.read_text()
    assert "pet <= pet_threshold" in src, (
        "PET threshold gate removed or renamed in "
        f"{GRID_PET_SRC.name}. The paper's screening rule requires "
        "a `pet <= pet_threshold` comparison."
    )


def test_bev_config_is_diagonal_affine():
    """The BEV projection must be a diagonal affine, not a full homography.

    Every coordinate transform downstream assumes
        X = H[0,0] * px + H[0,2]
        Y = H[1,1] * py + H[1,2]
    If off-diagonals become nonzero, the local<->world roundtrip
    used for conflict-zone rendering and PET grid assignment breaks.
    """
    cfg = json.loads(BEV_CFG.read_text())
    H = cfg["H_pixel_to_world"]
    assert abs(H[0][1]) < 1e-12, f"H[0][1]={H[0][1]} not zero"
    assert abs(H[1][0]) < 1e-12, f"H[1][0]={H[1][0]} not zero"
    assert H[2][2] == 1.0, f"H[2][2]={H[2][2]} != 1"
    assert H[0][0] > 0 and H[1][1] > 0, "negative or zero scale"
    for k in ("x_min", "x_max", "y_min", "y_max"):
        assert k in cfg, f"missing {k} in BEV config"


def test_grid_config_matches_video_dimensions():
    """Grid corners must span the full 1600x720 video frame.

    Regression: if corners are changed to a subregion, most detections
    fall outside the grid and PET pairing silently drops events.
    """
    cfg = json.loads(GRID_CFG.read_text())
    c = cfg["corners"]
    assert c["top_left"] == [0, 0]
    assert c["bottom_right"] == [1600, 720]
    assert c["top_right"] == [1600, 0]
    assert c["bottom_left"] == [0, 720]
    cell = cfg["configuration"]["cell_size"]
    assert isinstance(cell, int) and cell >= 10, f"cell_size={cell} invalid"


def test_trajectory_json_coordinate_convention():
    """Lock in the exact pixel -> world mapping used by the trajectory JSONs.

    Convention (verified against outputs/giti_raw.csv):
        world_x = (A_X * px + C_X) - x_min          X local to x_min
        world_y = y_max - (A_Y * py + C_Y)          Y flipped, local to y_max

    The Y axis is intentionally flipped: bottom of the camera image maps
    to world_y ~ 0, top maps to world_y ~ y_max - y_min. Every downstream
    consumer (grid cell assignment, conflict-zone rendering, PET grid)
    relies on this being consistent. If the convention changes silently,
    distances still compute but the grid cells move and every count in
    the paper becomes wrong.

    Regression: guard against
      - switching to absolute UTM coordinates,
      - dropping the Y flip,
      - changing x_min / y_max in bev_config.json without re-running.
    """
    import csv as _csv

    cfg = json.loads(BEV_CFG.read_text())
    x_min = cfg["x_min"]
    y_max = cfg["y_max"]
    A_X = cfg["H_pixel_to_world"][0][0]
    C_X = cfg["H_pixel_to_world"][0][2]
    A_Y = cfg["H_pixel_to_world"][1][1]
    C_Y = cfg["H_pixel_to_world"][1][2]

    with RAW_CSV.open() as fh:
        reader = _csv.DictReader(fh)
        row = next(reader)
    traj = json.loads(row["traj_a_json"])

    # Check the formula holds for several points, not just the first.
    for pt in traj[: min(10, len(traj))]:
        px, py = float(pt["x_pixel"]), float(pt["y_pixel"])
        wx, wy = float(pt["world_x"]), float(pt["world_y"])

        expected_wx = (A_X * px + C_X) - x_min
        expected_wy = y_max - (A_Y * py + C_Y)

        assert abs(wx - expected_wx) < 1e-3, (
            f"world_x={wx} != (A_X*px + C_X) - x_min = {expected_wx}"
        )
        assert abs(wy - expected_wy) < 1e-3, (
            f"world_y={wy} != y_max - (A_Y*py + C_Y) = {expected_wy}. "
            "Either the Y-flip convention changed or trajectories switched "
            "to absolute coordinates. Both break the paper's grid cells."
        )


def test_frozen_artifact_counts_unchanged():
    """Paper headline counts must match the frozen summary.

    If any of these change without a re-run and a re-freeze, the
    numbers in the paper no longer trace to the tree.
    """
    summary = json.loads(FROZEN_SUMMARY.read_text())
    assert summary["GITI"]["screened_events"] == 153, (
        f"GITI screened = {summary['GITI']['screened_events']}, paper says 153"
    )
    assert summary["MRC"]["screened_events"] == 34, (
        f"MRC screened = {summary['MRC']['screened_events']}, paper says 34"
    )
    total = summary["GITI"]["screened_events"] + summary["MRC"]["screened_events"]
    assert total == 187, f"combined = {total}, paper says 187"
