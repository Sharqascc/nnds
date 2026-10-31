"""Property tests for src.utils.seed."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from src.utils import seed as s


@given(value=st.integers(0, 2**31 - 1))
def test_set_seed_updates_get_seed(value):
    s.set_seed(value)
    assert s.get_seed() == value


@given(value=st.integers(0, 2**31 - 1))
def test_set_seed_makes_python_random_deterministic(value):
    s.set_seed(value)
    a = random.random()
    s.set_seed(value)
    b = random.random()
    assert a == b


@given(value=st.integers(0, 2**31 - 1))
def test_set_seed_makes_numpy_deterministic(value):
    s.set_seed(value)
    a = float(np.random.rand())
    s.set_seed(value)
    b = float(np.random.rand())
    assert a == b


@given(value=st.integers(0, 2**31 - 1))
def test_set_seed_makes_torch_deterministic(value):
    s.set_seed(value)
    a = float(torch.rand(1).item())
    s.set_seed(value)
    b = float(torch.rand(1).item())
    assert a == b


@given(value=st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False))
def test_non_int_seed_raises(value):
    with pytest.raises(TypeError):
        s.set_seed(value)


def test_string_seed_raises():
    with pytest.raises(TypeError):
        s.set_seed("42")
