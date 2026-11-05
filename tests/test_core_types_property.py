"""Property tests for src.core.types dataclasses."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.core import types as t

_FINITE = st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False)


@given(x=_FINITE, y=_FINITE, time_=_FINITE)
def test_world_point_accepts_finite(x, y, time_):
    wp = t.WorldPoint(t=time_, x=x, y=y)
    assert wp.x == x
    assert wp.y == y


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_world_point_rejects_non_finite(bad):
    with pytest.raises(ValueError):
        t.WorldPoint(t=0.0, x=bad, y=0.0)
    with pytest.raises(ValueError):
        t.WorldPoint(t=bad, x=0.0, y=0.0)


def test_trajectory_empty_allowed():
    tr = t.Trajectory(track_id=1, points=())
    assert tr.duration == 0.0


@given(
    n=st.integers(1, 20),
)
def test_trajectory_strictly_increasing_ok(n):
    pts = tuple(t.WorldPoint(t=float(i), x=0.0, y=0.0) for i in range(n))
    tr = t.Trajectory(track_id=1, points=pts)
    assert tr.duration == float(n - 1)


def test_trajectory_non_increasing_rejected():
    pts = (
        t.WorldPoint(t=0.0, x=0.0, y=0.0),
        t.WorldPoint(t=0.0, x=0.0, y=0.0),
    )
    with pytest.raises(ValueError):
        t.Trajectory(track_id=1, points=pts)


def test_pet_event_rejects_same_tracks():
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


@given(pet=st.floats(0.0, 10.0, allow_nan=False, allow_infinity=False))
def test_pet_event_accepts_valid(pet):
    ta = t.Trajectory(track_id=1, points=(t.WorldPoint(0.0, 0.0, 0.0),))
    tb = t.Trajectory(track_id=2, points=(t.WorldPoint(0.0, 1.0, 0.0),))
    ev = t.PETEvent(
        event_id=0,
        pet=pet,
        track_a=1,
        track_b=2,
        conflict_type="cell",
        world_traj_i=ta,
        world_traj_j=tb,
    )
    assert ev.pet == pet


def test_trajectory_batch_valid_shapes():
    batch = t.TrajectoryBatch(
        inputs=np.zeros((2, 3, 4)),
        targets=np.zeros((2, 5, 4)),
        meta={},
        fps=30.0,
    )
    assert batch.batch_size == 2
    assert batch.input_length == 3
    assert batch.target_length == 5


def test_trajectory_batch_rejects_mismatched_batch():
    with pytest.raises(ValueError):
        t.TrajectoryBatch(
            inputs=np.zeros((2, 3, 4)),
            targets=np.zeros((3, 5, 4)),
            meta={},
            fps=30.0,
        )


def test_trajectory_batch_rejects_bad_fps():
    with pytest.raises(ValueError):
        t.TrajectoryBatch(
            inputs=np.zeros((1, 3, 4)),
            targets=np.zeros((1, 3, 4)),
            meta={},
            fps=0.0,
        )


def test_trajectory_batch_rejects_non_3d():
    with pytest.raises(ValueError):
        t.TrajectoryBatch(
            inputs=np.zeros((2, 3)),
            targets=np.zeros((2, 3)),
            meta={},
            fps=30.0,
        )
