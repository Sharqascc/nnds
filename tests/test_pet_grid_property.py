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
def _make_interval(obj_id, cell_id, t1, t2, world_samples):
    """Build an Interval whose t_enter <= t_exit by sorting the two draws.

    The previous strategy used .filter(t_enter < t_exit), which cannot work
    once Interval.__post_init__ rejects malformed input: Hypothesis calls
    the filter AFTER construction, so bad draws raise instead of being
    discarded.
    """
    t_enter, t_exit = (t1, t2) if t1 <= t2 else (t2, t1)
    return Interval(
        obj_id=obj_id,
        cell_id=cell_id,
        t_enter=t_enter,
        t_exit=t_exit,
        world_samples=world_samples,
    )


def interval_strategy():
    return st.builds(
        _make_interval,
        obj_id=st.integers(min_value=0, max_value=10),
        cell_id=st.text(min_size=1, max_size=5),
        t1=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        t2=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        world_samples=st.lists(
            st.builds(WorldSample, t=st.floats(), x=st.floats(), y=st.floats()), max_size=5
        ),
    )


@given(interval_strategy(), interval_strategy(), st.floats(min_value=0.1, max_value=10))
def test_compute_pet_non_negative(iv_a, iv_b, pet_threshold):
    events = compute_pet([iv_a, iv_b], pet_threshold=pet_threshold)
    for ev in events:
        assert ev.pet >= 0


@given(st.lists(st.floats(min_value=0.01, max_value=5), min_size=1, max_size=20))
def test_summarize_pet_positive(pet_values):
    events = [
        PETEvent(
            obj_i=1,
            obj_j=2,
            cell_id="C",
            t_exit_i=0,
            t_enter_j=0.1,
            pet=val,
            world_traj_i=[],
            world_traj_j=[],
            severity="critical",
        )
        for val in pet_values
    ]
    summary = summarize_pet(events)
    assert summary.count == len(events)
    assert summary.min_pet >= 0
    assert summary.max_pet <= max(pet_values) if pet_values else True
