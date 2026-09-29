import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from src.analysis.visualization.pet_event_plots import (
    COLORS,
    DEFAULT_THRESHOLDS,
    EventPlotter,
    compute_timing_from_traj,
)


def make_traj(times, xs, ys):
    return list(zip(times, xs, ys, strict=False))


@given(
    st.lists(st.floats(0.0, 10.0, allow_nan=False, allow_infinity=False), min_size=2, max_size=10),
    st.lists(
        st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False), min_size=2, max_size=10
    ),
    st.lists(
        st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False), min_size=2, max_size=10
    ),
)
@settings(max_examples=50)
def test_compute_timing_matches_direct_calculation(times, xs, ys):
    # Ensure all lists have same length
    min_len = min(len(times), len(xs), len(ys))
    times = times[:min_len]
    xs = xs[:min_len]
    ys = ys[:min_len]
    assume(min_len >= 2)

    traj_i = make_traj(times, xs, ys)
    # Generate second trajectory with slight offset
    traj_j = make_traj(times, [x + 1.0 for x in xs], [y - 1.0 for y in ys])

    df = pd.DataFrame({"traj_i": [traj_i], "traj_j": [traj_j]})
    result = compute_timing_from_traj(df)

    # Direct computation
    ti = np.array(times)
    xi = np.array(xs)
    yi = np.array(ys)
    xj = np.array([x + 1.0 for x in xs])
    yj = np.array([y - 1.0 for y in ys])

    dist = np.hypot(xi - xj, yi - yj)
    k_min = int(np.argmin(dist))

    assert result.iloc[0]["k_closest"] == k_min
    assert np.isclose(result.iloc[0]["dist_min"], dist[k_min], atol=1e-9)
    assert np.isclose(result.iloc[0]["t_closest"], ti[k_min], atol=1e-9)
    # t_leave_i is max(0, k_min-1), t_enter_j is min(T-1, k_min+1)
    k_leave = max(0, k_min - 1)
    k_enter = min(len(ti) - 1, k_min + 1)
    assert np.isclose(result.iloc[0]["t_leave_i"], ti[k_leave], atol=1e-9)
    assert np.isclose(result.iloc[0]["t_enter_j"], ti[k_enter], atol=1e-9)
    assert np.isclose(result.iloc[0]["pet_approx"], ti[k_enter] - ti[k_leave], atol=1e-9)


@given(st.floats(0.0, 10.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_severity_color_and_label_consistent(pet_value):
    plotter = EventPlotter(thresholds=DEFAULT_THRESHOLDS)
    color = plotter._get_severity_color(pet_value)
    label = plotter._get_severity_label(pet_value)

    if pet_value < DEFAULT_THRESHOLDS["critical"]:
        assert color == COLORS["red"]
        assert label == "Critical"
    elif pet_value < DEFAULT_THRESHOLDS["serious"]:
        assert color == COLORS["orange"]
        assert label == "Serious"
    elif pet_value < DEFAULT_THRESHOLDS["moderate"]:
        assert color == COLORS["yellow"]
        assert label == "Moderate"
    elif pet_value < DEFAULT_THRESHOLDS["safe"]:
        assert color == COLORS["green"]
        assert label == "Slight"
    else:
        assert color == COLORS["blue"]
        assert label == "Safe"


@given(
    st.lists(
        st.floats(0.1, 20.0, allow_nan=False, allow_infinity=False),
        min_size=4,
        max_size=4,
        unique=True,
    ).map(sorted),
    st.floats(0.0, 30.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50)
def test_severity_with_custom_thresholds(threshold_list, pet_value):
    # Build thresholds from sorted unique values (guaranteed strictly increasing)
    thresholds = {
        "critical": threshold_list[0],
        "serious": threshold_list[1],
        "moderate": threshold_list[2],
        "safe": threshold_list[3],
    }
    plotter = EventPlotter(thresholds=thresholds)
    color = plotter._get_severity_color(pet_value)
    label = plotter._get_severity_label(pet_value)

    # Verify mapping for custom thresholds
    if pet_value < thresholds["critical"]:
        assert color == COLORS["red"]
        assert label == "Critical"
    elif pet_value < thresholds["serious"]:
        assert color == COLORS["orange"]
        assert label == "Serious"
    elif pet_value < thresholds["moderate"]:
        assert color == COLORS["yellow"]
        assert label == "Moderate"
    elif pet_value < thresholds["safe"]:
        assert color == COLORS["green"]
        assert label == "Slight"
    else:
        assert color == COLORS["blue"]
        assert label == "Safe"
