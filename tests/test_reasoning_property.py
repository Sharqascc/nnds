
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from src.core.reasoning import (
    Detection,
    PETEvent,
    TrackingState,
    TrackingStateMachine,
    Trajectory,
    WorldPoint,
)


# ---------- WorldPoint ----------
@given(st.floats(min_value=0, max_value=100), st.floats(), st.floats())
def test_world_point_accepts_non_negative_time(t, x, y):
    p = WorldPoint(t=t, x=x, y=y)
    assert p.t >= 0

@given(st.floats(max_value=-0.001), st.floats(), st.floats())
def test_world_point_rejects_negative_time(t, x, y):
    with pytest.raises(ValidationError):
        WorldPoint(t=t, x=x, y=y)


# ---------- Trajectory ----------
@given(st.lists(st.builds(WorldPoint, t=st.floats(min_value=0, max_value=10), x=st.floats(), y=st.floats()), min_size=2, max_size=20))
def test_trajectory_accepts_ordered_points(points):
    # sort by t to ensure valid
    points = sorted(points, key=lambda p: p.t)
    traj = Trajectory(track_id=0, points=points)
    assert len(traj.points) >= 2

@given(st.sets(st.floats(min_value=0, max_value=10), min_size=2, max_size=20))
def test_trajectory_rejects_unordered_points(times_set):
    times = sorted(times_set)
    points = [WorldPoint(t=t, x=0, y=0) for t in times]
    # Swap first two points to break temporal ordering
    points[0], points[1] = points[1], points[0]
    with pytest.raises(ValidationError):
        Trajectory(track_id=0, points=points)


# ---------- Detection ----------
@given(
    st.floats(min_value=0, max_value=49, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0, max_value=49, allow_nan=False, allow_infinity=False),
    st.floats(min_value=50, max_value=100, allow_nan=False, allow_infinity=False),
    st.floats(min_value=50, max_value=100, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
)
def test_detection_accepts_valid_conf(x1, y1, x2, y2, conf):
    # x1,y1 < 50 <= x2,y2 guarantees a valid box
    det = Detection(frame=1, x1=x1, y1=y1, x2=x2, y2=y2,
                    cx=(x1+x2)/2, cy=(y1+y2)/2, cls_id=2,
                    cls_name="car", conf=conf, source="uvh")
    assert 0 <= det.conf <= 1

@given(st.floats(min_value=1.1, max_value=10))
def test_detection_rejects_conf_out_of_range(conf):
    with pytest.raises(ValidationError):
        Detection(frame=1, x1=0, y1=0, x2=10, y2=10, cx=5, cy=5,
                  cls_id=2, cls_name="car", conf=conf, source="uvh")


# ---------- PETEvent ----------
@given(st.floats(min_value=0, max_value=10))
def test_pet_event_accepts_non_negative_pet(pet):
    ev = PETEvent(event_id=1, pet=pet, frame=100, track_a=1, track_b=2,
                  conflict_type="crossing", grid_cell="G_A_1",
                  track_a_exit_frame=90, track_b_entry_frame=110, site="GITI")
    assert ev.pet >= 0

@given(st.floats(max_value=-0.01))
def test_pet_event_rejects_negative_pet(pet):
    with pytest.raises(ValidationError):
        PETEvent(event_id=1, pet=pet, frame=100, track_a=1, track_b=2,
                 conflict_type="crossing", grid_cell="G_A_1",
                 track_a_exit_frame=90, track_b_entry_frame=110, site="GITI")


# ---------- Tracking state machine ----------
VALID_TRANSITIONS = {
    TrackingState.DETECTED: [TrackingState.TRACKING, TrackingState.EXITED],
    TrackingState.TRACKING: [TrackingState.TRACKING, TrackingState.MISSED, TrackingState.CONFLICT, TrackingState.EXITED],
    TrackingState.MISSED: [TrackingState.TRACKING, TrackingState.EXITED],
    TrackingState.CONFLICT: [TrackingState.TRACKING, TrackingState.EXITED],
    TrackingState.EXITED: [],
}

@given(st.sampled_from(list(TrackingState)))
def test_state_machine_all_valid_transitions(initial_state):
    sm = TrackingStateMachine(initial_state)
    allowed = VALID_TRANSITIONS[initial_state]
    for target in allowed:
        sm.transition(target)
        assert sm.state == target
        sm.state = initial_state  # reset

@given(st.sampled_from(list(TrackingState)))
def test_state_machine_illegal_transitions_raise(initial_state):
    sm = TrackingStateMachine(initial_state)
    illegal = [s for s in TrackingState if s not in VALID_TRANSITIONS[initial_state]]
    for target in illegal:
        with pytest.raises(ValueError):
            sm.transition(target)
