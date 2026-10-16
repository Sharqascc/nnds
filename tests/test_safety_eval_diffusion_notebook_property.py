"""Property tests for pet_to_risk_exponential in
src.analysis.safety_eval_diffusion_notebook.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

sedn = pytest.importorskip(
    "src.analysis.safety_eval_diffusion_notebook",
    reason="module imports heavy diffusion dependencies",
)


_FINITE_POS = st.floats(0.0, 1000.0, allow_nan=False, allow_infinity=False)


@given(pet=_FINITE_POS, half_life=st.floats(0.01, 100.0, allow_nan=False, allow_infinity=False))
def test_risk_in_unit_interval(pet, half_life):
    r = sedn.pet_to_risk_exponential(pet, half_life)
    assert 0.0 <= r <= 1.0


def test_zero_pet_gives_one():
    assert sedn.pet_to_risk_exponential(0.0, 1.0) == pytest.approx(1.0)


@given(
    pet1=st.floats(0.0, 10.0, allow_nan=False, allow_infinity=False),
    pet2=st.floats(0.0, 10.0, allow_nan=False, allow_infinity=False),
    half_life=st.floats(0.1, 5.0, allow_nan=False, allow_infinity=False),
)
def test_monotone_decreasing(pet1, pet2, half_life):
    if pet1 >= pet2:
        return
    r1 = sedn.pet_to_risk_exponential(pet1, half_life)
    r2 = sedn.pet_to_risk_exponential(pet2, half_life)
    assert r1 >= r2 - 1e-12


def test_nan_pet_returns_zero():
    assert sedn.pet_to_risk_exponential(float("nan"), 1.0) == 0.0


def test_inf_pet_returns_zero():
    assert sedn.pet_to_risk_exponential(float("inf"), 1.0) == 0.0


def test_negative_pet_returns_zero():
    assert sedn.pet_to_risk_exponential(-0.5, 1.0) == 0.0


def test_zero_half_life_is_bounded():
    # Code uses max(half_life, 1e-6); zero half_life should not divide-by-zero
    r = sedn.pet_to_risk_exponential(1.0, 0.0)
    assert 0.0 <= r <= 1.0
