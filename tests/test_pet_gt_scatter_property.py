"""Property tests for src.analysis.visualization.pet_gt_scatter."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis.visualization import pet_gt_scatter as pgs

_ARR = st.lists(
    st.floats(0.0, 5.0, allow_nan=False, allow_infinity=False),
    min_size=1, max_size=20,
)


def test_shape_mismatch_raises(tmp_path):
    pred = np.array([1.0, 2.0, 3.0])
    gt = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        pgs.plot_pet_agreement(pred, gt, tmp_path / "out.png")


@given(vals=_ARR)
def test_matched_shape_writes_file(tmp_path_factory, vals):
    tmp = tmp_path_factory.mktemp("scatter")
    out = tmp / "out.png"
    arr = np.array(vals)
    result = pgs.plot_pet_agreement(arr, arr, out)
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_empty_arrays_do_not_crash(tmp_path):
    out = tmp_path / "empty.png"
    result = pgs.plot_pet_agreement(np.array([]), np.array([]), out)
    assert result == out
    assert out.exists()


def test_custom_label_and_unit_do_not_crash(tmp_path):
    arr = np.array([0.5, 1.0, 1.5])
    out = tmp_path / "custom.png"
    pgs.plot_pet_agreement(arr, arr, out, field_label="TTC", unit="ms")
    assert out.exists()


def test_parent_directory_created(tmp_path):
    out = tmp_path / "nested" / "deep" / "out.png"
    arr = np.array([1.0, 2.0])
    pgs.plot_pet_agreement(arr, arr, out)
    assert out.exists()
