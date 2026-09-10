
import pytest
import numpy as np
from hypothesis import given, strategies as st
from src.analysis.ssm.ssm_verification import SSMVerifier, compare_with_reference

# ------------------------------------------------------------------
# 1. PHYSICALITY TESTS (Invariants)
# ------------------------------------------------------------------

@given(st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=10, max_size=100))
def test_pet_non_negativity_invariant(values):
    """
    PROVE: PET values should be handled such that the verifier 
    flags negative values as errors or cleans them.
    """
    verifier = SSMVerifier(strict_mode=True)
    # We simulate a case where some values are negative (physically impossible)
    results = verifier.verify_pet_calculation(np.array(values))
    
    # If the data contains negatives, the data_quality check must fail or 
    # the clean_data must be non-negative.
    if any(v < 0 for v in values):
        # The check_data_quality should flag this via expected_range
        # We test if the verifier identifies that the data is out of range
        quality = verifier.check_data_quality(np.array(values), "PET", expected_range=(0.0, 100.0))
        # It should either mark as passed=False or put them in warnings/errors
        # If it's strict_mode, it should be an error.
        assert quality["passed"] == False or len(quality["warnings"]) > 0

@given(st.lists(st.floats(min_value=0.0, max_value=1e6), min_size=10, max_size=100))
def test_ttc_distribution_stability(values):
    """
    PROVE: TTC verification should not crash regardless of the 
    magnitude of the input values (Stability Test).
    """
    verifier = SSMVerifier()
    # This test ensures no ZeroDivisionError or OverflowError occurs
    try:
        results = verifier.verify_ttc_calculation(np.array(values))
        assert "summary" in results
    except Exception as e:
        pytest.fail(f"TTC Verification crashed with {type(e).__name__}: {e}")

# ------------------------------------------------------------------
# 2. SINGULARITY & EDGE CASE TESTS
# ------------------------------------------------------------------

def test_drac_extreme_values():
    """
    Check how the verifier handles extreme deceleration 
    (e.g., 100g deceleration, which is physically impossible).
    """
    verifier = SSMVerifier()
    extreme_drac = np.array([10.0, 50.0, 100.0]) # > 9.8 m/s^2
    results = verifier.verify_drac_calculation(extreme_drac)
    
    # It should flag these as 'extreme' in the severity distribution
    stats = results["statistics"]["severity_distribution"]
    assert stats["extreme"]["count"] == 3
    assert "warnings" in results and len(results["warnings"]) > 0

def test_empty_or_nan_data():
    """
    Ensure the system handles empty arrays or arrays of NaNs without crashing.
    """
    verifier = SSMVerifier()
    
    # Case 1: Pure NaNs
    nan_data = np.array([np.nan, np.nan])
    res_nan = verifier.verify_pet_calculation(nan_data)
    assert res_nan["passed"] == False
    assert "no valid data" in res_nan["summary"]
    
    # Case 2: Empty array
    empty_data = np.array([])
    res_empty = verifier.verify_pet_calculation(empty_data)
    assert res_empty["passed"] == False
    assert "no valid data" in res_empty["summary"]

# ------------------------------------------------------------------
# 3. STATISTICAL VALIDATION TESTS
# ------------------------------------------------------------------

def test_statistical_comparison_identical():
    """
    PROVE: If two datasets are identical, the comparison 
    must pass and Cohen's d must be 0.
    """
    data = np.random.normal(1.5, 0.2, 100)
    results = compare_with_reference(data, data, "TTC")
    
    assert results["passed"] == True
    assert results["tests"]["t_test"]["p_value"] > 0.05
    assert abs(results["effect_size"]["cohens_d"]) < 1e-7

def test_statistical_comparison_wildly_different():
    """
    PROVE: If datasets are vastly different, the comparison 
    must fail (Significant p-value and large effect size).
    """
    observed = np.random.normal(1.0, 0.1, 100)
    reference = np.random.normal(5.0, 0.1, 100)
    results = compare_with_reference(observed, reference, "TTC")
    
    assert results["passed"] == False
    assert results["tests"]["t_test"]["p_value"] < 0.05
    assert abs(results["effect_size"]["cohens_d"]) > 1.0
