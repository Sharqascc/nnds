"""Property tests for src.core.contracts Pydantic models."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from src.core import contracts as c


def _valid_event(**overrides):
    base = {
        "event_id": 0,
        "pet": 1.0,
        "frame": 100,
        "track_a": 1,
        "track_b": 2,
        "conflict_type": "crossing",
        "grid_cell": "CELL_A_1",
        "site": "GITI",
    }
    base.update(overrides)
    return base


def test_minimal_valid_event():
    rec = c.PETEventRecord(**_valid_event())
    assert rec.pet == 1.0


def test_negative_pet_rejected():
    with pytest.raises(ValidationError):
        c.PETEventRecord(**_valid_event(pet=-0.1))


def test_same_tracks_rejected():
    with pytest.raises(ValidationError):
        c.PETEventRecord(**_valid_event(track_a=5, track_b=5))


@given(
    pet=st.floats(0.0, 10.0, allow_nan=False, allow_infinity=False),
    event_id=st.integers(0, 10_000),
    frame=st.integers(0, 100_000),
    ta=st.integers(0, 1000),
    tb=st.integers(0, 1000),
)
def test_valid_events_roundtrip(pet, event_id, frame, ta, tb):
    if ta == tb:
        tb = tb + 1
    rec = c.PETEventRecord(**_valid_event(
        event_id=event_id, pet=pet, frame=frame,
        track_a=ta, track_b=tb,
    ))
    assert rec.pet == pet
    assert rec.track_a == ta
    assert rec.track_b == tb


@given(
    pct=st.floats(-1.0, -0.01, allow_nan=False, allow_infinity=False)
)
def test_negative_percentage_rejected(pct):
    with pytest.raises(ValidationError):
        c.RiskLevelSummary(count=1, percentage=pct)


def test_uncertainty_rejects_empty_error_sources():
    with pytest.raises(ValidationError):
        c.PETUncertaintyContract(
            nominal_pet=1.0, uncertainty_std=0.1, error_sources={}
        )


def test_uncertainty_rejects_negative_error_source():
    with pytest.raises(ValidationError):
        c.PETUncertaintyContract(
            nominal_pet=1.0, uncertainty_std=0.1,
            error_sources={"a": -0.1},
        )


def test_uq_analysis_passed_without_errors():
    c.UQAnalysisContract(metric_name="pet", passed=True, method="x")
    with pytest.raises(ValidationError):
        c.UQAnalysisContract(metric_name="pet", passed=False, method="x")
