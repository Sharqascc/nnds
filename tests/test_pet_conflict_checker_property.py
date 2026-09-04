
import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.analysis.pet_conflict_checker import (
    classify_pet_severity,
    compute_grid_pet,
    compute_pet,
    filter_by_roi,
    get_trajectory_pairs,
)


@given(st.floats(min_value=-1.0, max_value=10.0))
def test_classify_pet_severity_returns_enum(pet):
    severity = classify_pet_severity(pet)
    assert severity.value in {"critical", "serious", "moderate", "minor", "safe"}

@given(st.lists(st.floats(min_value=0, max_value=10), min_size=1), st.lists(st.floats(min_value=0, max_value=10), min_size=1))
def test_compute_pet_non_negative(times_a, times_b):
    pet = compute_pet(times_a, times_b)
    assert pet >= 0 or np.isinf(pet)

@given(st.integers(min_value=1, max_value=10), st.integers(min_value=1, max_value=10), st.integers(min_value=1, max_value=10), st.integers(min_value=1, max_value=10))
def test_compute_grid_pet_shape_and_non_negative(t, h, w, fps):
    grid_a = np.random.randint(0, 2, size=(t, h, w)).astype(bool)
    grid_b = np.random.randint(0, 2, size=(t, h, w)).astype(bool)
    pet = compute_grid_pet(grid_a, grid_b, fps)
    assert pet >= 0 or np.isinf(pet)

@given(st.floats(min_value=0, max_value=100), st.floats(min_value=0, max_value=100),
       st.floats(min_value=0, max_value=100), st.floats(min_value=0, max_value=100))
def test_filter_by_roi_bounds(x, y, x_max, y_max):
    df = pd.DataFrame({"x": [x], "y": [y]})
    roi = {"xmin": 0, "xmax": x_max, "ymin": 0, "ymax": y_max}
    out = filter_by_roi(df, roi)
    if 0 <= x <= x_max and 0 <= y <= y_max:
        assert len(out) == 1
    else:
        assert len(out) == 0

@given(st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=20))
def test_get_trajectory_pairs_sorted_unique(track_ids):
    df = pd.DataFrame({"track_id": track_ids, "frame": range(len(track_ids))})
    pairs = get_trajectory_pairs(df)
    for a, b in pairs:
        assert a < b
    assert len(pairs) == len(set(pairs))
