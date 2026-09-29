import pytest
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
def test_world_point_valid():
    p = WorldPoint(t=0.0, x=1.0, y=2.0)
    assert p.t == 0.0


def test_world_point_negative_time():
    with pytest.raises(ValidationError):
        WorldPoint(t=-1.0, x=0, y=0)


# ---------- Trajectory ----------
def test_trajectory_valid():
    pts = [WorldPoint(t=0, x=0, y=0), WorldPoint(t=1, x=1, y=1)]
    traj = Trajectory(track_id=1, points=pts)
    assert len(traj.points) == 2


def test_trajectory_too_few_points():
    with pytest.raises(ValidationError):
        Trajectory(track_id=1, points=[WorldPoint(t=0, x=0, y=0)])


def test_trajectory_unordered_points():
    pts = [WorldPoint(t=1, x=0, y=0), WorldPoint(t=0, x=1, y=1)]
    with pytest.raises(ValidationError):
        Trajectory(track_id=1, points=pts)


# ---------- Detection ----------
def test_detection_valid():
    det = Detection(
        frame=1,
        x1=0,
        y1=0,
        x2=10,
        y2=10,
        cx=5,
        cy=5,
        cls_id=2,
        cls_name="car",
        conf=0.9,
        source="uvh",
    )
    assert det.conf == 0.9


def test_detection_invalid_conf():
    with pytest.raises(ValidationError):
        Detection(
            frame=1,
            x1=0,
            y1=0,
            x2=10,
            y2=10,
            cx=5,
            cy=5,
            cls_id=2,
            cls_name="car",
            conf=1.5,
            source="uvh",
        )


def test_detection_invalid_box():
    with pytest.raises(ValidationError):
        Detection(
            frame=1,
            x1=10,
            y1=0,
            x2=0,
            y2=10,
            cx=5,
            cy=5,
            cls_id=2,
            cls_name="car",
            conf=0.9,
            source="uvh",
        )


# ---------- PETEvent ----------
def test_pet_event_valid():
    ev = PETEvent(
        event_id=1,
        pet=0.5,
        frame=100,
        track_a=1,
        track_b=2,
        conflict_type="crossing",
        grid_cell="G_A_1",
        track_a_exit_frame=90,
        track_b_entry_frame=110,
        site="GITI",
    )
    assert ev.pet == 0.5


def test_pet_event_negative_pet():
    with pytest.raises(ValidationError):
        PETEvent(
            event_id=1,
            pet=-0.1,
            frame=100,
            track_a=1,
            track_b=2,
            conflict_type="crossing",
            grid_cell="G_A_1",
            track_a_exit_frame=90,
            track_b_entry_frame=110,
            site="GITI",
        )


# ---------- TrackingStateMachine ----------
def test_state_machine_valid_transitions():
    sm = TrackingStateMachine(TrackingState.DETECTED)
    sm.transition(TrackingState.TRACKING)
    assert sm.state == TrackingState.TRACKING
    sm.transition(TrackingState.CONFLICT)
    assert sm.state == TrackingState.CONFLICT


def test_state_machine_illegal_transition():
    sm = TrackingStateMachine(TrackingState.EXITED)
    with pytest.raises(ValueError):
        sm.transition(TrackingState.TRACKING)


def test_state_machine_can_transition():
    sm = TrackingStateMachine(TrackingState.TRACKING)
    assert sm.can_transition(TrackingState.MISSED) is True
    assert sm.can_transition(TrackingState.CONFLICT) is True
    assert sm.can_transition(TrackingState.EXITED) is True
