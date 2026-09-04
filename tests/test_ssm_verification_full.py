from unittest.mock import patch

import numpy as np

from src.analysis.ssm.ssm_verification import (
    SSMVerifier,
    compare_with_reference,
    run_verification_suite,
    verify_drac_calculation,
    verify_pet_calculation,
    verify_ttc_calculation,
)

# ---------- check_data_quality ----------


def test_check_data_quality_type_conversion_success():
    verifier = SSMVerifier(min_sample_size=1)
    result = verifier.check_data_quality([1.0, 2.0, 3.0], name="test")
    assert result["passed"] == True
    assert any(c["check"] == "Type conversion" and c["passed"] for c in result["checks"])


class BadArray:
    def __array__(self, *args, **kwargs):
        raise ValueError("cannot convert")


def test_check_data_quality_type_conversion_failure():
    verifier = SSMVerifier(min_sample_size=1)
    result = verifier.check_data_quality(BadArray(), name="test")
    assert result["passed"] == False
    assert any("Cannot convert" in e for e in result["errors"])


def test_check_data_quality_out_of_range_gt5():
    verifier = SSMVerifier(min_sample_size=1)
    data = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
    )  # 1/11 >9%? Actually 1/11=9.09% <5? need >5: 1/11=9.09% >5 yes
    result = verifier.check_data_quality(data, expected_range=(0, 10))
    assert result["passed"] == False
    assert any("outside expected range" in e for e in result["errors"])


def test_check_data_quality_out_of_range_le5():
    verifier = SSMVerifier(min_sample_size=1)
    data = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 100]
    )  # 1/21=4.76%
    result = verifier.check_data_quality(data, expected_range=(0, 20))
    assert result["passed"] == True  # less than 5% out-of-range is warning, not error
    assert any("outside expected range" in w for w in result["warnings"])


def test_check_data_quality_outliers_gt10():
    verifier = SSMVerifier(min_sample_size=1)
    data = np.array([1] * 17 + [100, 100, 100])  # 3/20=15% outliers
    result = verifier.check_data_quality(data)
    assert any("potential outliers" in w for w in result["warnings"])


def test_check_data_quality_normality_nonnormal():
    verifier = SSMVerifier(min_sample_size=1)
    data = np.arange(20, dtype=float)
    with patch("scipy.stats.shapiro", return_value=(1.0, 0.01)):
        result = verifier.check_data_quality(data)
    assert "normality_p_value" in result["statistics"]
    assert any(c.get("check") == "Normality" for c in result["checks"])


# ---------- verify_pet_calculation ----------


def test_verify_pet_high_critical_rate_warning():
    verifier = SSMVerifier(min_sample_size=1)
    data = np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2])
    result = verifier.verify_pet_calculation(data)
    assert any("High critical event rate" in w for w in result["warnings"])


def test_verify_pet_reference_mean_high_error():
    verifier = SSMVerifier(min_sample_size=1)
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    reference = {"mean": 10.0}
    result = verifier.verify_pet_calculation(data, reference_values=reference)
    assert "reference_comparison" in result
    assert result["reference_comparison"]["mean"]["passed"] == False
    assert any("Mean PET differs" in w for w in result["warnings"])


def test_verify_pet_reference_mean_low_error():
    verifier = SSMVerifier(min_sample_size=1)
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    reference = {"mean": 3.0}
    result = verifier.verify_pet_calculation(data, reference_values=reference)
    assert result["reference_comparison"]["mean"]["passed"] == True


# ---------- verify_ttc_calculation ----------


def test_verify_ttc_no_valid_data():
    verifier = SSMVerifier(min_sample_size=1)
    data = np.array([np.nan, np.nan])
    result = verifier.verify_ttc_calculation(data)
    assert result["passed"] == False
    assert "no valid data" in result["summary"]


def test_verify_ttc_near_collision_strict_mode():
    verifier = SSMVerifier(min_sample_size=1, strict_mode=True)
    data = np.array([0.1, 1.0, 2.0])
    result = verifier.verify_ttc_calculation(data)
    assert result["passed"] == False
    assert any("near-collision" in e for e in result["errors"])


def test_verify_ttc_near_collision_warning():
    verifier = SSMVerifier(min_sample_size=1, strict_mode=False)
    data = np.array(
        [0.1, 0.2, 1.0, 2.0, 3.0]
    )  # 2/5=40% but rate >5 triggers error? Actually condition strict_mode or rate>5 -> error
    # We need rate <=5 and not strict_mode for warning
    verifier = SSMVerifier(min_sample_size=1, strict_mode=False)
    data = np.array(
        [
            0.1,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
            18.0,
            19.0,
        ]
    )  # 1/20=5% -> warning not error
    result = verifier.verify_ttc_calculation(data)
    assert any("near-collision" in w for w in result["warnings"])


# ---------- verify_drac_calculation ----------


def test_verify_drac_no_valid_data():
    verifier = SSMVerifier(min_sample_size=1)
    data = np.array([np.nan, np.inf])
    result = verifier.verify_drac_calculation(data)
    assert result["passed"] == False
    assert "no valid data" in result["summary"]


def test_verify_drac_extreme_events():
    verifier = SSMVerifier(min_sample_size=1)
    data = np.array([1.0, 2.0, 3.0, 10.0, 11.0])
    result = verifier.verify_drac_calculation(data)
    assert any("emergency braking" in w for w in result["warnings"])


# ---------- run_verification_suite ----------


def test_run_verification_suite_all_metrics():
    verifier = SSMVerifier(min_sample_size=1)
    pet = np.array([1.0, 2.0, 3.0, 4.0])
    ttc = np.array([1.0, 2.0, 3.0, 4.0])
    drac = np.array([1.0, 2.0, 3.0])
    result = verifier.run_verification_suite(pet_values=pet, ttc_values=ttc, drac_values=drac)
    assert result["overall_pass"] == True
    assert len(result["tests"]) == 3


def test_run_verification_suite_some_fail():
    verifier = SSMVerifier(min_sample_size=1)
    pet = np.array([1.0, 2.0, 3.0])
    ttc = np.array([np.nan])  # invalid
    result = verifier.run_verification_suite(pet_values=pet, ttc_values=ttc)
    assert result["overall_pass"] == False


def test_run_verification_suite_no_data():
    verifier = SSMVerifier()
    result = verifier.run_verification_suite()
    assert result["overall_pass"] == False
    assert result["summary"] == "Verification Suite: No data provided"


# ---------- compare_with_reference ----------


def test_compare_with_reference():
    observed = np.array([1.0, 2.0, 3.0])
    reference = np.array([1.1, 2.1, 3.1])
    result = compare_with_reference(observed, reference)
    assert "t_test" in result["tests"]
    assert "ks_test" in result["tests"]
    assert "effect_size" in result


# ---------- Additional coverage for missing lines ----------


def test_check_data_quality_low_completeness_and_small_sample():
    verifier = SSMVerifier(min_sample_size=1000)
    data = np.array([1.0, 2.0, 3.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    result = verifier.check_data_quality(data)
    # completeness = 3/9 = 33% <90 -> warning
    assert any("Low completeness" in w for w in result["warnings"])
    # sample size 3 < 1000 -> small sample warning
    assert any("Small sample size" in w for w in result["warnings"])


def test_verify_pet_no_valid_data():
    verifier = SSMVerifier(min_sample_size=1)
    result = verifier.verify_pet_calculation(np.array([np.nan, np.inf]))
    assert result["passed"] == False
    assert "no valid data" in result["summary"]


def test_run_verification_suite_pet_fails():
    verifier = SSMVerifier(min_sample_size=1)
    pet = np.array([np.nan, np.inf])
    result = verifier.run_verification_suite(pet_values=pet)
    assert result["overall_pass"] == False
    assert any(t["passed"] == False for t in result["tests"])


def test_convenience_wrappers():
    pet = np.array([1.0, 2.0, 3.0, 4.0])
    ttc = np.array([1.0, 2.0, 3.0, 4.0])
    drac = np.array([1.0, 2.0, 3.0])
    r1 = verify_pet_calculation(pet)
    assert r1["metric"] == "PET"
    r2 = verify_ttc_calculation(ttc)
    assert r2["metric"] == "TTC"
    r3 = verify_drac_calculation(drac)
    assert r3["metric"] == "DRAC"
    r4 = run_verification_suite(pet_values=pet, ttc_values=ttc, drac_values=drac)
    assert "overall_pass" in r4


def test_run_verification_suite_drac_fails():
    verifier = SSMVerifier(min_sample_size=1)
    drac = np.array([np.nan, np.inf])
    result = verifier.run_verification_suite(drac_values=drac)
    assert result["overall_pass"] == False
    assert any(t["passed"] == False for t in result["tests"])
