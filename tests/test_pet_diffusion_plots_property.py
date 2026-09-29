import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis.visualization.pet_diffusion_plots import (
    plot_bland_altman,
    plot_pet_like_histogram,
    plot_true_vs_pet_like,
    plot_true_vs_sample_delta,
)

# Strategy for pet_pairs: list of (pet_real, pet_sample) where either can be None
pet_pair_st = st.tuples(
    st.one_of(
        st.none(), st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
    ),
    st.one_of(
        st.none(), st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
    ),
)
pet_pairs_st = st.lists(pet_pair_st, min_size=0, max_size=20)

# Strategy for record: (row_idx, true_pet, pet_like_real, pet_like_sample)
record_st = st.tuples(
    st.integers(min_value=0, max_value=100),
    st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    st.one_of(
        st.none(), st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
    ),
    st.one_of(
        st.none(), st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
    ),
)
records_st = st.lists(record_st, min_size=0, max_size=20)


@settings(deadline=None)
@given(pet_pairs_st)
def test_plot_pet_like_histogram_no_raise(pet_pairs):
    plot_pet_like_histogram(pet_pairs)
    import matplotlib.pyplot as plt

    plt.close("all")


@settings(deadline=None)
@given(records_st)
def test_plot_true_vs_pet_like_no_raise(records):
    # Disable regression to avoid linregress on constant data
    plot_true_vs_pet_like(records, add_regression=False)
    import matplotlib.pyplot as plt

    plt.close("all")


@settings(deadline=None)
@given(records_st)
def test_plot_true_vs_sample_delta_no_raise(records):
    plot_true_vs_sample_delta(records)
    import matplotlib.pyplot as plt

    plt.close("all")


@settings(deadline=None)
@given(records_st)
def test_plot_bland_altman_no_raise(records):
    plot_bland_altman(records)
    import matplotlib.pyplot as plt

    plt.close("all")
