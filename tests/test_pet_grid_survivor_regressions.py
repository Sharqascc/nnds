"""Behavioural regressions confirmed by PET-grid survivor replay."""

import pytest

import src.analysis.grid_trajectory.pet_grid as pet_grid


def test_critical_severity_uses_value_equality():
    severity = bytearray(b"critical").decode("ascii")
    literal = "critical"
    assert severity == literal
    assert severity is not literal, "Fixture must use a distinct string object"

    event = pet_grid.PETEvent(
        obj_i=1,
        obj_j=2,
        cell_id="cell",
        t_exit_i=0.0,
        t_enter_j=0.5,
        pet=0.5,
        world_traj_i=[],
        world_traj_j=[],
        severity=severity,
    )
    assert event.is_critical is True


def test_default_downsampling_retains_all_world_samples():
    logger = pet_grid.TrajectoryLogger(fps=10)
    for frame in range(4):
        logger.log(
            track_id=7,
            frame_idx=frame,
            cell_id="cell",
            world_x=float(frame),
            world_y=float(frame + 10),
        )

    intervals = logger.build_intervals()
    assert len(intervals) == 1
    actual = [(sample.t, sample.x, sample.y) for sample in intervals[0].world_samples]
    assert actual == [
        (0.0, 0.0, 10.0),
        (0.1, 1.0, 11.0),
        (0.2, 2.0, 12.0),
        (0.3, 3.0, 13.0),
    ]


def test_out_of_order_samples_are_sorted_by_frame():
    logger = pet_grid.TrajectoryLogger(fps=10, downsample_every=1)

    # Same cell keeps the interval valid even under the wrong sort.
    # Arrival order differs from chronological order.
    for frame in (0, 2, 1, 3):
        logger.log(
            track_id=7,
            frame_idx=frame,
            cell_id="cell",
            world_x=float(frame),
            world_y=float(frame + 10),
        )

    intervals = logger.build_intervals()
    assert len(intervals) == 1
    interval = intervals[0]
    assert interval.t_enter == pytest.approx(0.0)
    assert interval.t_exit == pytest.approx(0.3)
    assert [(sample.t, sample.x, sample.y) for sample in interval.world_samples] == [
        (0.0, 0.0, 10.0),
        (0.1, 1.0, 11.0),
        (0.2, 2.0, 12.0),
        (0.3, 3.0, 13.0),
    ]


def test_single_track_average_counts_all_samples():
    logger = pet_grid.TrajectoryLogger(fps=10)
    for frame in range(4):
        logger.log(track_id=7, frame_idx=frame, cell_id="cell")

    stats = logger.get_stats()
    assert stats["num_tracks"] == 1
    assert stats["total_samples"] == 4
    assert stats["avg_samples_per_track"] == pytest.approx(4.0)
