import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis.visualization.industry_standard_viz import SSMPlotter


def make_plotter():
    """Create an SSMPlotter instance without calling __init__ (no style overhead)."""
    return SSMPlotter.__new__(SSMPlotter)


@settings(deadline=None)
@given(
    st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=50,
    )
)
def test_validate_ssm_data_valid_1d(data):
    obj = make_plotter()
    arr = np.array(data, dtype=float)
    # Use allow_negative=True to avoid negative warnings affecting validity
    result = obj.validate_ssm_data(arr, metric_name="test", allow_negative=True)
    assert result["valid"] is True
    assert result["clean_data"] is not None
    assert result["n_clean"] <= len(arr)
    assert 0.0 <= result["removal_rate"] <= 100.0


@settings(deadline=None)
@given(
    st.lists(
        st.one_of(
            st.floats(min_value=-10.0, max_value=10.0), st.just(float("nan")), st.just(float("inf"))
        ),
        min_size=1,
        max_size=50,
    )
)
def test_validate_ssm_data_handles_nan_inf(data):
    obj = make_plotter()
    arr = np.array(data, dtype=float)
    result = obj.validate_ssm_data(arr, metric_name="test", allow_negative=True)
    if np.any(np.isfinite(arr)):
        assert result["valid"] is True
        assert result["n_clean"] >= 0
    else:
        assert result["valid"] is False


@given(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
def test_validate_ssm_data_shape_errors(data):
    obj = make_plotter()
    arr_2d = np.array([[data, data], [data, data]])
    result = obj.validate_ssm_data(arr_2d, metric_name="test")
    assert result["valid"] is False
    assert len(result["errors"]) > 0


@given(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
def test_format_p_value_apa_style(p):
    obj = make_plotter()
    s = obj._format_p_value(p)
    if p < 0.001:
        assert s == "p < 0.001***"
    elif p < 0.01:
        assert "**" in s and "p = " in s
    elif p < 0.05:
        assert "*" in s and "p = " in s
    else:
        assert "(ns)" in s and "p = " in s
