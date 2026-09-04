import pytest

from src.analysis.grid_trajectory.pet_grid import (
    Interval,
    PETEvent,
    PETSummary,
    TrajectoryLogger,
    WorldSample,
    classify_pet,
    compute_pet,
    summarize_pet,
)

# ---------- Dataclasses & properties ----------


def test_world_sample_repr():
    ws = WorldSample(t=1.5, x=3.2, y=4.8)
    assert "WorldSample(t=1.50" in repr(ws)


def test_interval_and_pet_event_repr():
    ws = WorldSample(t=0, x=1, y=2)
    interval = Interval(obj_id=1, cell_id="G_A_1", t_enter=0.0, t_exit=1.0, world_samples=[ws])
    assert isinstance(interval, Interval)
    ev = PETEvent(
        obj_i=1,
        obj_j=2,
        cell_id="G_A_1",
        t_exit_i=1.0,
        t_enter_j=1.5,
        pet=0.5,
        world_traj_i=[ws],
        world_traj_j=[ws],
        severity="critical",
    )
    assert ev.is_critical is True
    assert ev.is_conflict is True
    assert ev.time_gap == 0.5
    assert "PETEvent(1->2" in repr(ev)


def test_pet_event_not_critical_and_conflict():
    ws = WorldSample(t=0, x=0, y=0)
    ev = PETEvent(
        obj_i=1,
        obj_j=2,
        cell_id="G_A_1",
        t_exit_i=1.0,
        t_enter_j=2.0,
        pet=1.0,
        world_traj_i=[ws],
        world_traj_j=[ws],
        severity="safe",
    )
    assert ev.is_critical is False
    assert ev.is_conflict is False


def test_pet_summary_dataclass():
    s = PETSummary(
        count=0,
        min_pet=None,
        max_pet=None,
        mean_pet=None,
        p5=None,
        p50=None,
        p95=None,
        n_critical=0,
        n_moderate=0,
        n_safe=0,
    )
    assert s.count == 0


# ---------- TrajectoryLogger ----------


def test_trajectory_logger_init_errors():
    with pytest.raises(ValueError):
        TrajectoryLogger(fps=0)
    with pytest.raises(ValueError):
        TrajectoryLogger(fps=30, downsample_every=0)


def test_trajectory_logger_log_and_stats():
    logger = TrajectoryLogger(fps=10.0, downsample_every=2)
    logger.log(1, 0, "A", 10.0, 20.0)
    logger.log(1, 1, "A", 11.0, 21.0)
    logger.log(2, 0, "B", 30.0, 40.0)
    stats = logger.get_stats()
    assert stats["num_tracks"] == 2
    assert stats["total_samples"] == 3
    assert stats["fps"] == 10.0
    assert stats["downsample_every"] == 2
    assert stats["avg_samples_per_track"] == 1.5


def test_build_intervals_single_cell_single_interval():
    logger = TrajectoryLogger(fps=2.0, downsample_every=1)
    logger.log(1, 0, "A", 0.0, 0.0)
    logger.log(1, 1, "A", 10.0, 10.0)
    logger.log(1, 2, "A", 20.0, 20.0)
    intervals = logger.build_intervals()
    assert len(intervals) == 1
    iv = intervals[0]
    assert iv.obj_id == 1
    assert iv.cell_id == "A"
    assert iv.t_enter == 0.0
    assert iv.t_exit == 1.0
    assert len(iv.world_samples) == 3


def test_build_intervals_multiple_cells():
    logger = TrajectoryLogger(fps=1.0, downsample_every=1)
    logger.log(1, 0, "A", 0.0, 0.0)
    logger.log(1, 1, "A", 1.0, 1.0)
    logger.log(1, 2, "B", 2.0, 2.0)
    logger.log(1, 3, "B", 3.0, 3.0)
    intervals = logger.build_intervals()
    assert len(intervals) == 2
    first = intervals[0]
    second = intervals[1]
    assert first.cell_id == "A"
    assert first.t_enter == 0.0
    assert first.t_exit == 1.0
    assert second.cell_id == "B"
    assert second.t_enter == 2.0
    assert second.t_exit == 3.0


def test_build_intervals_no_world_coords():
    logger = TrajectoryLogger(fps=2.0, downsample_every=1)
    logger.log(1, 0, "A")  # no world coords
    logger.log(1, 1, "A")
    intervals = logger.build_intervals()
    assert len(intervals) == 1
    assert intervals[0].world_samples == []


def test_build_intervals_downsampling():
    logger = TrajectoryLogger(fps=5.0, downsample_every=2)
    for i in range(5):
        logger.log(1, i, "A", float(i), float(i))
    intervals = logger.build_intervals()
    assert len(intervals) == 1
    samples = intervals[0].world_samples
    # downsampling every 2: samples at 0,2,4 -> 3 samples
    assert len(samples) == 3


# ---------- classify_pet ----------


def test_classify_pet():
    assert classify_pet(1.0, critical_threshold=1.5, moderate_threshold=3.0) == "critical"
    assert classify_pet(2.0, critical_threshold=1.5, moderate_threshold=3.0) == "moderate"
    assert classify_pet(4.0, critical_threshold=1.5, moderate_threshold=3.0) == "safe"


# ---------- summarize_pet ----------


def test_summarize_pet_empty():
    summary = summarize_pet([])
    assert summary.count == 0
    assert summary.min_pet is None


def test_summarize_pet_nonempty():
    samples = [WorldSample(t=0, x=0, y=0)]  # dummy for events
    events = [
        PETEvent(
            obj_i=1,
            obj_j=2,
            cell_id="A",
            t_exit_i=0,
            t_enter_j=0.5,
            pet=0.5,
            world_traj_i=samples,
            world_traj_j=samples,
            severity="critical",
        ),
        PETEvent(
            obj_i=2,
            obj_j=3,
            cell_id="A",
            t_exit_i=1,
            t_enter_j=2,
            pet=1.0,
            world_traj_i=samples,
            world_traj_j=samples,
            severity="moderate",
        ),
        PETEvent(
            obj_i=3,
            obj_j=4,
            cell_id="A",
            t_exit_i=2,
            t_enter_j=5,
            pet=3.0,
            world_traj_i=samples,
            world_traj_j=samples,
            severity="safe",
        ),
    ]
    summary = summarize_pet(events, critical_threshold=1.5, moderate_threshold=3.0)
    assert summary.count == 3
    assert summary.min_pet == 0.5
    assert summary.max_pet == 3.0
    assert summary.mean_pet == 1.5
    assert summary.n_critical == 2
    assert summary.n_moderate == 0
    assert summary.n_safe == 1


# ---------- compute_pet ----------


def test_compute_pet_invalid_threshold():
    with pytest.raises(ValueError):
        compute_pet([], pet_threshold=0)


def test_compute_pet_empty():
    events = compute_pet([])
    assert events == []


def test_compute_pet_case_A_exits_before_B():
    # interval A: 0-2, interval B: 3-5 -> pet = 1.0
    ws = []
    A = Interval(obj_id=1, cell_id="G", t_enter=0.0, t_exit=2.0, world_samples=ws)
    B = Interval(obj_id=2, cell_id="G", t_enter=3.0, t_exit=5.0, world_samples=ws)
    events = compute_pet([A, B], pet_threshold=2.0)
    assert len(events) == 1
    assert events[0].obj_i == 1
    assert events[0].obj_j == 2
    assert events[0].pet == 1.0
    assert events[0].severity == "critical"


def test_compute_pet_case_B_exits_before_A_no_event():
    # Sorted order: B (enter 0) first, A (enter 5) second.
    # First case A.t_exit <= B.t_enter? 1 <= 5 -> true, pet = 5-1 = 4.0 > threshold -> no event.
    ws = []
    A = Interval(obj_id=1, cell_id="G", t_enter=5.0, t_exit=8.0, world_samples=ws)
    B = Interval(obj_id=2, cell_id="G", t_enter=0.0, t_exit=1.0, world_samples=ws)
    events = compute_pet([A, B], pet_threshold=3.0)
    assert len(events) == 0


def test_compute_pet_no_event_if_gap_zero():
    # intervals touching: t_exit = t_enter => pet=0 not included
    ws = []
    A = Interval(obj_id=1, cell_id="G", t_enter=0.0, t_exit=2.0, world_samples=ws)
    B = Interval(obj_id=2, cell_id="G", t_enter=2.0, t_exit=4.0, world_samples=ws)
    events = compute_pet([A, B], pet_threshold=2.0)
    assert len(events) == 0


def test_compute_pet_no_event_if_exceeds_threshold():
    ws = []
    A = Interval(obj_id=1, cell_id="G", t_enter=0.0, t_exit=1.0, world_samples=ws)
    B = Interval(obj_id=2, cell_id="G", t_enter=5.0, t_exit=6.0, world_samples=ws)
    events = compute_pet([A, B], pet_threshold=2.0)
    assert len(events) == 0


def test_compute_pet_case_B_exits_before_A_event():
    """Cover B->A direction when B exits before A enters and gap within threshold."""
    ws = []
    A = Interval(obj_id=1, cell_id="G", t_enter=4.0, t_exit=6.0, world_samples=ws)
    B = Interval(obj_id=2, cell_id="G", t_enter=0.0, t_exit=2.0, world_samples=ws)
    events = compute_pet([A, B], pet_threshold=3.0, critical_threshold=1.5, moderate_threshold=3.0)
    assert len(events) == 1
    ev = events[0]
    assert ev.obj_i == 2
    assert ev.obj_j == 1
    assert ev.pet == 2.0
    assert ev.severity == "moderate"


def test_compute_pet_case2_degenerate():
    """Cover B->A branch using a degenerate interval where exit < enter."""
    ws = []
    A = Interval(obj_id=1, cell_id="G", t_enter=10.0, t_exit=20.0, world_samples=ws)
    B = Interval(obj_id=2, cell_id="G", t_enter=15.0, t_exit=5.0, world_samples=ws)  # degenerate
    events = compute_pet([A, B], pet_threshold=10.0, critical_threshold=1.5, moderate_threshold=3.0)
    assert len(events) == 1
    ev = events[0]
    assert ev.obj_i == 2
    assert ev.obj_j == 1
    assert ev.pet == 5.0


def test_reload_module_for_import_lines():
    """Reload module to ensure import-conditional lines are covered."""
    import importlib

    import src.analysis.grid_trajectory.pet_grid as pg

    importlib.reload(pg)
