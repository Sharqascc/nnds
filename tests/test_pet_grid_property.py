
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.analysis.grid_trajectory.pet_grid import (
    Interval,
    PETEvent,
    WorldSample,
    compute_pet,
    summarize_pet,
)


# Helper to build valid intervals
def interval_strategy():
    return st.builds(
        Interval,
        obj_id=st.integers(min_value=0, max_value=10),
        cell_id=st.text(min_size=1, max_size=5),
        t_enter=st.floats(min_value=0, max_value=50, allow_nan=False, allow_infinity=False),
        t_exit=st.floats(min_value=0.1, max_value=50, allow_nan=False, allow_infinity=False),
        world_samples=st.lists(st.builds(WorldSample, t=st.floats(), x=st.floats(), y=st.floats()), max_size=5)
    ).filter(lambda iv: iv.t_enter < iv.t_exit)

@given(interval_strategy(), interval_strategy(), st.floats(min_value=0.1, max_value=10))
def test_compute_pet_non_negative(iv_a, iv_b, pet_threshold):
    events = compute_pet([iv_a, iv_b], pet_threshold=pet_threshold)
    for ev in events:
        assert ev.pet >= 0

@given(st.lists(st.floats(min_value=0.01, max_value=5), min_size=1, max_size=20))
def test_summarize_pet_positive(pet_values):
    events = [
        PETEvent(
            obj_i=1, obj_j=2, cell_id="C", t_exit_i=0, t_enter_j=0.1,
            pet=val, world_traj_i=[], world_traj_j=[], severity="critical"
        )
        for val in pet_values
    ]
    summary = summarize_pet(events)
    assert summary.count == len(events)
    assert summary.min_pet >= 0
    assert summary.max_pet <= max(pet_values) if pet_values else True
