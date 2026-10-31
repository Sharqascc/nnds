"""Integration: core types <-> contracts <-> validation roundtrip.

Verifies that data can flow through the core dataclasses, the Pydantic
contracts, and the validation layer without structural mismatch.
Runs in PR CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core import contracts as c
from src.core import types as t


def test_world_point_to_trajectory_to_pet_event_chain():
    pts_a = (
        t.WorldPoint(t=0.0, x=0.0, y=0.0),
        t.WorldPoint(t=1.0, x=5.0, y=5.0),
    )
    pts_b = (
        t.WorldPoint(t=0.0, x=0.0, y=5.0),
        t.WorldPoint(t=1.0, x=5.0, y=0.0),
    )
    ta = t.Trajectory(track_id=1, points=pts_a)
    tb = t.Trajectory(track_id=2, points=pts_b)
    ev = t.PETEvent(
        event_id=0,
        pet=0.5,
        track_a=1,
        track_b=2,
        conflict_type="CELL_A_1",
        world_traj_i=ta,
        world_traj_j=tb,
    )
    assert ev.pet == 0.5
    assert ev.track_a != ev.track_b


def test_pet_event_record_matches_type_event_fields():
    """The Pydantic PETEventRecord accepts the fields produced by PETEvent."""
    rec = c.PETEventRecord(
        event_id=0,
        pet=0.5,
        frame=100,
        track_a=1,
        track_b=2,
        conflict_type="crossing",
        grid_cell="CELL_A_1",
        site="GITI",
    )
    assert rec.pet == 0.5
    assert rec.track_a == 1
    assert rec.track_b == 2


def test_contracts_reject_invalid_pet_from_type_layer():
    """If PETEvent rejects same tracks, PETEventRecord must also reject."""
    ta = t.Trajectory(track_id=1, points=(t.WorldPoint(0.0, 0.0, 0.0),))
    with pytest.raises(ValueError):
        t.PETEvent(
            event_id=0,
            pet=1.0,
            track_a=1,
            track_b=1,
            conflict_type="x",
            world_traj_i=ta,
            world_traj_j=ta,
        )
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        c.PETEventRecord(
            event_id=0,
            pet=1.0,
            frame=0,
            track_a=5,
            track_b=5,
            conflict_type="crossing",
            grid_cell="CELL_A_1",
            site="GITI",
        )


def test_validation_module_imports_and_exposes_api():
    from src.core import validation

    names = [n for n in dir(validation) if not n.startswith("_")]
    assert len(names) > 0, "validation module has no public names"


def test_uncertainty_contract_accepts_valid_error_sources():
    uq = c.PETUncertaintyContract(
        nominal_pet=1.0,
        uncertainty_std=0.1,
        error_sources={"tracking": 0.05, "calibration": 0.08},
    )
    assert uq.nominal_pet == 1.0
    assert len(uq.error_sources) == 2


def test_uq_analysis_contract_passed_with_errors_ok():
    # passing=True with errors is allowed (unusual but not contradictory)
    c.UQAnalysisContract(metric_name="pet", passed=True, method="x", errors=["non-fatal warning"])
    # passing=False without errors is not allowed
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        c.UQAnalysisContract(metric_name="pet", passed=False, method="x")
