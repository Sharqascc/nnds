import math

import pytest

from pet_interval import compute_pet_from_intervals


def test_positive_pet():
    result = compute_pet_from_intervals(1.0, 2.0, 3.5, 4.0)

    assert result.pet_s == pytest.approx(1.5)
    assert result.pet_status == "sequential"
    assert result.first_actor == "a"
    assert result.second_actor == "b"
    assert result.overlap_duration_s == 0.0


def test_zero_gap_is_sequential_not_overlap():
    result = compute_pet_from_intervals(1.0, 2.0, 2.0, 3.0)

    assert result.pet_s == pytest.approx(0.0)
    assert result.pet_status == "sequential"
    assert result.overlap_duration_s == 0.0


def test_overlap_has_no_pet():
    result = compute_pet_from_intervals(1.0, 3.0, 1.5, 2.5)

    assert result.pet_s is None
    assert result.pet_status == "overlap"
    assert result.first_actor is None
    assert result.second_actor is None
    assert result.overlap_duration_s == pytest.approx(1.0)


def test_reverse_sequential_order():
    result = compute_pet_from_intervals(3.0, 4.0, 1.0, 2.0)

    assert result.pet_s == pytest.approx(1.0)
    assert result.pet_status == "sequential"
    assert result.first_actor == "b"
    assert result.second_actor == "a"


@pytest.mark.parametrize(
    "args",
    [
        (math.nan, 2.0, 3.0, 4.0),
        (1.0, math.inf, 3.0, 4.0),
        (2.0, 1.0, 3.0, 4.0),
        (1.0, 2.0, 4.0, 3.0),
    ],
)
def test_invalid_intervals_raise(args):
    with pytest.raises(ValueError):
        compute_pet_from_intervals(*args)
