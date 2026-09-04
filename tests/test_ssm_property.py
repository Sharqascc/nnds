
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.analysis.ssm.ssm_verification import (
    SSMVerifier,
    verify_drac_calculation,
    verify_pet_calculation,
    verify_ttc_calculation,
)
from src.analysis.ssm.uncertainty_quantifier import UncertaintyQuantifier


@given(st.lists(st.floats(min_value=0, max_value=30), min_size=1, max_size=50))
def test_verify_pet_calculation_non_negative(pet_values):
    verifier = SSMVerifier(min_sample_size=1)
    result = verifier.verify_pet_calculation(np.array(pet_values))
    # Clean data should have no negative PET
    clean = result["data_quality"]["clean_data"]
    if clean is not None and len(clean) > 0:
        assert np.all(clean >= 0)

@given(st.lists(st.floats(min_value=0, max_value=20), min_size=1, max_size=50))
def test_verify_ttc_calculation_non_negative(ttc_values):
    verifier = SSMVerifier(min_sample_size=1)
    result = verifier.verify_ttc_calculation(np.array(ttc_values))
    clean = result["data_quality"]["clean_data"]
    if clean is not None and len(clean) > 0:
        assert np.all(clean >= 0)

@given(st.lists(st.floats(min_value=0, max_value=10), min_size=1, max_size=50))
def test_verify_drac_calculation_non_negative(drac_values):
    verifier = SSMVerifier(min_sample_size=1)
    result = verifier.verify_drac_calculation(np.array(drac_values))
    clean = result["data_quality"]["clean_data"]
    if clean is not None and len(clean) > 0:
        assert np.all(clean >= 0)

@given(st.lists(st.floats(min_value=0.1, max_value=10), min_size=2, max_size=50))
def test_bootstrap_ci_order(data):
    uq = UncertaintyQuantifier(n_bootstrap=100, random_state=42)
    ci = uq.bootstrap_ci(np.array(data), method="percentile")
    assert ci[0] <= ci[1]
