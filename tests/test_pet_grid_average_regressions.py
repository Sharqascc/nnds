"""Three-way average checks derived from an empty-logger survivor."""

import math
import statistics

import numpy as np
import pytest

import src.analysis.grid_trajectory.pet_grid as pg


def loop_mean(values):
    total = 0.0
    count = 0
    for value in values:
        total += value
        count += 1
    return total / count if count else 0.0


def builtin_mean(values):
    return sum(values) / len(values) if values else 0.0


def numpy_mean(values):
    return float(np.mean(values)) if values else 0.0


@pytest.mark.parametrize(
    "counts,expected",
    [
        ([], 0.0),
        ([4], 4.0),
        ([2, 4], 3.0),
        ([1, 2, 3], 2.0),
        ([1, 2], 1.5),
    ],
    ids=[
        "empty",
        "one-track",
        "two-tracks",
        "three-tracks",
        "fractional-average",
    ],
)
def test_average_samples_three_way(counts, expected):
    # Empty input follows the logger's explicit zero convention.
    key = statistics.mean(counts) if counts else 0.0
    references = (
        loop_mean(counts),
        builtin_mean(counts),
        numpy_mean(counts),
        key,
    )
    for observed in references:
        assert math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12)

    logger = pg.TrajectoryLogger(fps=10)
    for track_id, count in enumerate(counts, start=1):
        for frame in range(count):
            logger.log(track_id, frame, "A")

    stats = logger.get_stats()
    assert stats["num_tracks"] == len(counts)
    assert stats["total_samples"] == sum(counts)
    assert stats["avg_samples_per_track"] == pytest.approx(key, rel=1e-12, abs=1e-12)
