"""PET-grid regressions derived from mutation survivor review.

Three independent calculation styles are checked against explicit examples.
statistics.mean supplies a mean reference. Percentiles follow this module's
rounded-index convention, not an interpolated median.

Negative PET inputs below test summary arithmetic, not scientific validity.
"""

import math
import statistics
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

import src.analysis.grid_trajectory.pet_grid as pg

fields = (
    "count",
    "min_pet",
    "max_pet",
    "mean_pet",
    "p5",
    "p50",
    "p95",
    "n_critical",
    "n_moderate",
    "n_safe",
)


def pack(n, low, high, mean, percentiles, counts):
    return (n, low, high, mean, *percentiles, *counts)


empty = (0, None, None, None, None, None, None, 0, 0, 0)


def loop_version(values):
    if not values:
        return empty
    ordered = list(values)
    for i in range(1, len(ordered)):
        value = ordered[i]
        j = i - 1
        while j >= 0 and ordered[j] > value:
            ordered[j + 1] = ordered[j]
            j -= 1
        ordered[j + 1] = value
    total = 0.0
    counts = [0, 0, 0]
    for value in values:
        total += value
        if value < 1.5:
            counts[0] += 1
        elif value < 3.0:
            counts[1] += 1
        else:
            counts[2] += 1
    indices = [round(p * (len(values) - 1)) for p in (0.05, 0.5, 0.95)]
    return pack(
        len(values),
        ordered[0],
        ordered[-1],
        total / len(values),
        [ordered[i] for i in indices],
        counts,
    )


def builtin_version(values):
    if not values:
        return empty
    ordered = sorted(values)
    indices = [
        round(p * (len(values) - 1)) for p in (Fraction(1, 20), Fraction(1, 2), Fraction(19, 20))
    ]
    return pack(
        len(values),
        min(values),
        max(values),
        statistics.mean(values),
        [ordered[i] for i in indices],
        (
            sum(value < 1.5 for value in values),
            sum(1.5 <= value < 3 for value in values),
            sum(value >= 3 for value in values),
        ),
    )


def numpy_version(values):
    if not values:
        return empty
    array = np.asarray(values, dtype=float)
    indices = np.rint(np.array([0.05, 0.5, 0.95]) * (array.size - 1)).astype(int)
    percentiles = np.sort(array)[indices].tolist()
    return pack(
        int(array.size),
        float(array.min()),
        float(array.max()),
        float(array.mean()),
        percentiles,
        (
            int(np.count_nonzero(array < 1.5)),
            int(np.count_nonzero((array >= 1.5) & (array < 3))),
            int(np.count_nonzero(array >= 3)),
        ),
    )


cases = [
    ("empty", [], empty),
    ("one", [4.0], (1, 4, 4, 4, 4, 4, 4, 0, 0, 1)),
    ("two", [2.0, 4.0], (2, 2, 4, 3, 2, 2, 4, 0, 1, 1)),
    ("negative-arithmetic", [-4.0, -2.0], (2, -4, -2, -3, -4, -4, -2, 2, 0, 0)),
    ("boundaries", [0.0, 1.5, 3.0], (3, 0, 3, 1.5, 0, 1.5, 3, 1, 1, 1)),
    ("percentile-spread", list(range(21)), (21, 0, 20, 10, 1, 10, 19, 2, 1, 18)),
    ("large", [1e150, 1e150], (2, 1e150, 1e150, 1e150, 1e150, 1e150, 1e150, 0, 0, 2)),
]


@pytest.mark.parametrize(
    "name,values,expected",
    cases,
    ids=[case[0] for case in cases],
)
def test_summary_three_way(name, values, expected):
    """Check all fields; only mean comparison permits rounding tolerance."""
    events = [SimpleNamespace(pet=value) for value in values]
    summary = pg.summarize_pet(events)
    outputs = {
        "loop": loop_version(values),
        "built-ins": builtin_version(values),
        "numpy": numpy_version(values),
        "repository": tuple(getattr(summary, field) for field in fields),
    }

    print(f"\nSummary case: {name}")
    print("| Implementation | Observed tuple | Expected tuple | Match |")
    print("|---|---|---|---|")
    failures = []

    for label, actual in outputs.items():
        field_failures = []
        if len(actual) != len(expected):
            field_failures.append("field count")
        else:
            for field, observed, wanted in zip(fields, actual, expected, strict=True):
                if observed is None or wanted is None:
                    matched = observed is wanted
                elif field == "mean_pet":
                    matched = math.isclose(observed, wanted, rel_tol=1e-12, abs_tol=1e-12)
                else:
                    matched = observed == wanted
                if not matched:
                    field_failures.append(f"{field}: observed={observed!r}, expected={wanted!r}")

        print(f"| {label} | {actual} | {expected} | {'MISMATCH' if field_failures else 'PASS'} |")
        if field_failures:
            failures.append((label, field_failures))

    assert not failures, f"Summary mismatches in {name}: {failures}"


@pytest.mark.parametrize(
    "value,expected",
    [(0.0, "critical"), (1.5, "moderate"), (3.0, "safe")],
    ids=["zero", "critical-boundary", "moderate-boundary"],
)
def test_severity_boundaries(value, expected):
    actual = pg.classify_pet(value)
    assert actual == expected, f"PET={value}: observed={actual!r}, expected={expected!r}"


def test_downsampled_world_samples_preserve_times_and_coordinates():
    logger = pg.TrajectoryLogger(fps=2, downsample_every=2)
    for frame in range(5):
        logger.log(7, frame, "A", 10 + frame, -frame)

    intervals = logger.build_intervals()
    assert len(intervals) == 1

    actual = [(sample.t, sample.x, sample.y) for sample in intervals[0].world_samples]
    expected = [
        (0.0, 10.0, 0.0),
        (1.0, 12.0, -2.0),
        (2.0, 14.0, -4.0),
    ]
    assert actual == expected


def test_single_zero_duration_interval_has_no_self_event():
    interval = pg.Interval(
        obj_id=1,
        cell_id="A",
        t_enter=2,
        t_exit=2,
        world_samples=[],
    )
    assert pg.compute_pet([interval]) == []
