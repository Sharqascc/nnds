import math

import pytest

from src.analysis.grid_trajectory.uvh_coco_fused_grid_pet import (
    _compute_structured_pet_from_windows,
)


def test_structured_pet_positive():
    result = _compute_structured_pet_from_windows(
        30, 60, 90, 120, fps=30.0
    )

    assert result.pet_s == pytest.approx(1.0)
    assert result.pet_status == "sequential"
    assert result.first_actor == "a"
    assert result.second_actor == "b"
    assert result.overlap_duration_s == 0.0


def test_structured_pet_zero_gap():
    result = _compute_structured_pet_from_windows(
        30, 60, 60, 90, fps=30.0
    )

    assert result.pet_s == pytest.approx(0.0)
    assert result.pet_status == "sequential"
    assert result.first_actor == "a"
    assert result.second_actor == "b"
    assert result.overlap_duration_s == 0.0


def test_structured_pet_reverse_order():
    result = _compute_structured_pet_from_windows(
        90, 120, 30, 60, fps=30.0
    )

    assert result.pet_s == pytest.approx(1.0)
    assert result.pet_status == "sequential"
    assert result.first_actor == "b"
    assert result.second_actor == "a"


def test_structured_pet_overlap():
    result = _compute_structured_pet_from_windows(
        30, 90, 45, 75, fps=30.0
    )

    assert result.pet_s is None
    assert result.pet_status == "overlap"
    assert result.first_actor is None
    assert result.second_actor is None
    assert result.overlap_duration_s == pytest.approx(1.0)


@pytest.mark.parametrize(
    "args",
    [
        (math.nan, 60, 90, 120, 30.0),
        (30, math.inf, 90, 120, 30.0),
        (30, 60, 90, 120, 0.0),
        (30, 60, 90, 120, -30.0),
        (60, 30, 90, 120, 30.0),
        (30, 60, 120, 90, 30.0),
    ],
)
def test_structured_pet_rejects_invalid_input(args):
    with pytest.raises(ValueError):
        _compute_structured_pet_from_windows(*args)
