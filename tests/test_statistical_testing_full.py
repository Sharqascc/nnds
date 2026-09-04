import numpy as np
import pytest

from src.analysis.verification.statistical_testing import (
    StatisticalTester,
    check_assumptions,
    chi_square_test,
    multiple_comparisons,
    paired_test,
)
from src.analysis.verification.statistical_testing import (
    test_drac_difference as st_test_drac_difference,
)
from src.analysis.verification.statistical_testing import (
    test_pet_difference as st_test_pet_difference,
)
from src.analysis.verification.statistical_testing import (
    test_ttc_difference as st_test_ttc_difference,
)


# ---------- _clean_data ----------
def test_clean_data():
    tester = StatisticalTester()
    arr = np.array([1.0, np.nan, 3.0, np.inf, 4.0])
    cleaned = tester._clean_data(arr)[0]
    assert np.array_equal(cleaned, np.array([1.0, 3.0, 4.0]))


# ---------- check_normality ----------
def test_check_normality_shapiro_small():
    tester = StatisticalTester()
    res = tester.check_normality(np.array([1.0]))
    assert res["passed"] == False
    assert "error" in res


def test_check_normality_shapiro_normal():
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 100)
    tester = StatisticalTester(alpha=0.05)
    res = tester.check_normality(data, method="shapiro")
    assert "statistic" in res
    assert "normal" in res


def test_check_normality_anderson():
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 100)
    tester = StatisticalTester(alpha=0.05)
    res = tester.check_normality(data, method="anderson")
    assert "statistic" in res
    assert "normal" in res


def test_check_normality_unknown_method():
    tester = StatisticalTester()
    with pytest.raises(ValueError):
        tester.check_normality(np.array([1, 2, 3]), method="bogus")


# ---------- check_homoscedasticity ----------
def test_homoscedasticity_need_two_groups():
    tester = StatisticalTester()
    res = tester.check_homoscedasticity(np.array([1, 2, 3]))
    assert "error" in res


def test_homoscedasticity_small_groups():
    tester = StatisticalTester()
    res = tester.check_homoscedasticity(np.array([1]), np.array([2]))
    assert "error" in res


def test_homoscedasticity_valid():
    rng = np.random.default_rng(42)
    g1 = rng.normal(0, 1, 30)
    g2 = rng.normal(0, 1, 30)
    tester = StatisticalTester()
    res = tester.check_homoscedasticity(g1, g2)
    assert "statistic" in res
    assert "equal_variances" in res


# ---------- check_assumptions ----------
def test_check_assumptions_t_test():
    rng = np.random.default_rng(42)
    g1 = rng.normal(0, 1, 30)
    g2 = rng.normal(0, 1, 30)
    tester = StatisticalTester()
    res = tester.check_assumptions(g1, g2, test_type="t-test")
    assert "checks" in res
    assert "all_passed" in res


def test_check_assumptions_anova():
    rng = np.random.default_rng(42)
    groups = [rng.normal(0, 1, 30), rng.normal(0, 1, 30), rng.normal(0, 1, 30)]
    tester = StatisticalTester()
    res = tester.check_assumptions(*groups, test_type="anova")
    assert res["test_type"] == "anova"


# ---------- t_test ----------
def test_t_test_one_sample():
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 30)
    tester = StatisticalTester()
    res = tester.t_test(data)
    assert res["type"] == "one-sample"
    assert "t_statistic" in res["test_statistics"]


def test_t_test_one_sample_insufficient():
    tester = StatisticalTester()
    res = tester.t_test(np.array([1.0]))
    assert res["passed"] == False


def test_t_test_two_sample_welch():
    rng = np.random.default_rng(42)
    g1 = rng.normal(0, 1, 30)
    g2 = rng.normal(0.5, 1, 30)
    tester = StatisticalTester()
    res = tester.t_test(g1, g2, equal_var=False)
    assert res["type"] == "two-sample (Welch)"
    assert "effect_size_cohens_d" in res["test_statistics"]


def test_t_test_two_sample_student():
    rng = np.random.default_rng(42)
    g1 = rng.normal(0, 1, 30)
    g2 = rng.normal(0.5, 1, 30)
    tester = StatisticalTester()
    res = tester.t_test(g1, g2, equal_var=True)
    assert res["type"] == "two-sample (Student)"


def test_t_test_paired_equal_length():
    rng = np.random.default_rng(42)
    before = rng.normal(0, 1, 20)
    after = before + rng.normal(0.5, 0.1, 20)
    tester = StatisticalTester()
    res = tester.t_test(before, after, paired=True)
    assert res["type"] == "paired"


def test_t_test_paired_length_mismatch():
    tester = StatisticalTester()
    res = tester.t_test(np.array([1, 2, 3]), np.array([1, 2]), paired=True)
    assert res["passed"] == False


def test_t_test_insufficient_group2():
    tester = StatisticalTester()
    res = tester.t_test(np.array([1, 2, 3]), np.array([1]))
    assert res["passed"] == False


# ---------- mann_whitney ----------
def test_mann_whitney_insufficient():
    tester = StatisticalTester()
    res = tester.mann_whitney(np.array([1]), np.array([2]))
    assert res["passed"] == False


def test_mann_whitney_valid():
    rng = np.random.default_rng(42)
    g1 = rng.normal(0, 1, 30)
    g2 = rng.normal(0.5, 1, 30)
    tester = StatisticalTester()
    res = tester.mann_whitney(g1, g2)
    assert "statistic" in res
    assert "p_value" in res


# ---------- wilcoxon ----------
def test_wilcoxon_length_mismatch():
    tester = StatisticalTester()
    res = tester.wilcoxon(np.array([1, 2, 3]), np.array([1, 2]))
    assert res["passed"] == False


def test_wilcoxon_insufficient():
    tester = StatisticalTester()
    res = tester.wilcoxon(np.array([1]), np.array([1]))
    assert res["passed"] == False


def test_wilcoxon_valid():
    rng = np.random.default_rng(42)
    g1 = rng.normal(0, 1, 20)
    g2 = g1 + rng.normal(0.5, 0.1, 20)
    tester = StatisticalTester()
    res = tester.wilcoxon(g1, g2)
    assert "statistic" in res


# ---------- anova ----------
def test_anova_need_two_groups():
    tester = StatisticalTester()
    res = tester.anova(np.array([1, 2, 3]))
    assert res["passed"] == False


def test_anova_small_group():
    tester = StatisticalTester()
    res = tester.anova(np.array([1]), np.array([2]))
    assert res["passed"] == False


def test_anova_valid():
    rng = np.random.default_rng(42)
    g1 = rng.normal(0, 1, 30)
    g2 = rng.normal(0.5, 1, 30)
    g3 = rng.normal(1, 1, 30)
    tester = StatisticalTester()
    res = tester.anova(g1, g2, g3)
    assert res["test"] == "One-way ANOVA"
    assert "f_statistic" in res


# ---------- kruskal_wallis ----------
def test_kruskal_need_two_groups():
    tester = StatisticalTester()
    res = tester.kruskal_wallis(np.array([1, 2, 3]))
    assert res["passed"] == False


def test_kruskal_small_group():
    tester = StatisticalTester()
    res = tester.kruskal_wallis(np.array([1]), np.array([2]))
    assert res["passed"] == False


def test_kruskal_valid():
    rng = np.random.default_rng(42)
    g1 = rng.normal(0, 1, 30)
    g2 = rng.normal(0.5, 1, 30)
    g3 = rng.normal(1, 1, 30)
    tester = StatisticalTester()
    res = tester.kruskal_wallis(g1, g2, g3)
    assert res["test"] == "Kruskal-Wallis H"
    assert "h_statistic" in res


# ---------- chi_square_test method ----------
def test_chi_square_method_expected_none():
    tester = StatisticalTester()
    obs = np.array([10, 20, 30])
    res = tester.chi_square_test(obs)
    assert "chi2_statistic" in res


def test_chi_square_method_with_expected():
    tester = StatisticalTester()
    obs = np.array([10, 20, 30])
    exp = np.array([15, 15, 30])
    res = tester.chi_square_test(obs, exp)
    assert "p_value" in res


# ---------- adjust_p_values ----------
def test_adjust_bonferroni():
    tester = StatisticalTester()
    p = np.array([0.01, 0.02, 0.03])
    adj = tester.adjust_p_values(p, method="bonferroni")
    assert len(adj) == 3
    assert adj[0] == pytest.approx(0.03)


def test_adjust_holm():
    tester = StatisticalTester()
    p = np.array([0.01, 0.02, 0.03])
    adj = tester.adjust_p_values(p, method="holm")
    assert len(adj) == 3


def test_adjust_fdr_bh():
    tester = StatisticalTester()
    p = np.array([0.01, 0.02, 0.03])
    adj = tester.adjust_p_values(p, method="fdr_bh")
    assert len(adj) == 3


def test_adjust_unknown():
    tester = StatisticalTester()
    with pytest.raises(ValueError):
        tester.adjust_p_values([0.1], method="bogus")


# ---------- _interpret_effect_size ----------
def test_interpret_effect_size():
    tester = StatisticalTester()
    assert tester._interpret_effect_size(0.1) == "negligible"
    assert tester._interpret_effect_size(0.3) == "small"
    assert tester._interpret_effect_size(0.6) == "medium"
    assert tester._interpret_effect_size(0.9) == "large"


# ---------- convenience functions ----------
def test_convenience_functions_parametric():
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, 30)
    b = rng.normal(0.5, 1, 30)
    res = st_test_pet_difference(a, b, parametric=True)
    assert "test_statistics" in res
    res = st_test_ttc_difference(a, b, parametric=True)
    assert "test_statistics" in res
    res = st_test_drac_difference(a, b, parametric=True)
    assert "test_statistics" in res


def test_convenience_functions_nonparametric():
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, 30)
    b = rng.normal(0.5, 1, 30)
    res = st_test_pet_difference(a, b, parametric=False)
    assert "statistic" in res
    res = st_test_ttc_difference(a, b, parametric=False)
    assert "statistic" in res
    res = st_test_drac_difference(a, b, parametric=False)
    assert "statistic" in res


def test_multiple_comparisons_parametric():
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, 30)
    b = rng.normal(0.5, 1, 30)
    c = rng.normal(1, 1, 30)
    res = multiple_comparisons(a, b, c, parametric=True, correction="holm")
    assert "omnibus_test" in res
    assert len(res["pairwise_comparisons"]) == 3


def test_multiple_comparisons_nonparametric():
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, 30)
    b = rng.normal(0.5, 1, 30)
    c = rng.normal(1, 1, 30)
    res = multiple_comparisons(a, b, c, parametric=False, correction="bonferroni")
    assert "omnibus_test" in res


# ---------- Missing branch coverage ----------


def test_check_assumptions_nonnormal():
    rng = np.random.default_rng(42)
    # Exponential distribution is non-normal
    g1 = rng.exponential(1, 30)
    g2 = rng.exponential(1.5, 30)
    tester = StatisticalTester(alpha=0.05)
    res = tester.check_assumptions(g1, g2, test_type="t-test")
    assert res["all_passed"] == False
    assert any("non-normal" in rec for rec in res["recommendations"])


def test_check_assumptions_unequal_variances_t_test():
    rng = np.random.default_rng(42)
    g1 = rng.normal(0, 1, 30)
    g2 = rng.normal(0, 10, 30)  # much larger variance
    tester = StatisticalTester(alpha=0.05)
    res = tester.check_assumptions(g1, g2, test_type="t-test")
    assert res["all_passed"] == False
    assert any("Welch's t-test" in rec for rec in res["recommendations"])


def test_check_assumptions_unequal_variances_anova():
    rng = np.random.default_rng(42)
    g1 = rng.normal(0, 1, 30)
    g2 = rng.normal(0, 10, 30)
    g3 = rng.normal(0, 10, 30)
    tester = StatisticalTester(alpha=0.05)
    res = tester.check_assumptions(g1, g2, g3, test_type="anova")
    assert res["all_passed"] == False
    assert any("Kruskal-Wallis" in rec for rec in res["recommendations"])


def test_t_test_warning_nonnormal():
    rng = np.random.default_rng(42)
    g1 = rng.exponential(1, 30)
    g2 = rng.exponential(1.5, 30)
    tester = StatisticalTester(alpha=0.05, auto_check_assumptions=True)
    res = tester.t_test(g1, g2, equal_var=False)
    assert any("Mann-Whitney" in w for w in res["warnings"])


def test_convenience_chi_square():
    observed = np.array([10, 20, 30])
    expected = np.array([15, 15, 30])
    res = chi_square_test(observed, expected)
    assert "chi2_statistic" in res


def test_convenience_paired_test_parametric():
    rng = np.random.default_rng(42)
    before = rng.normal(0, 1, 20)
    after = before + rng.normal(0.5, 0.1, 20)
    res = paired_test(before, after, parametric=True)
    assert "test_statistics" in res


def test_convenience_paired_test_nonparametric():
    rng = np.random.default_rng(42)
    before = rng.normal(0, 1, 20)
    after = before + rng.normal(0.5, 0.1, 20)
    res = paired_test(before, after, parametric=False)
    assert "statistic" in res


def test_convenience_check_assumptions():
    rng = np.random.default_rng(42)
    g1 = rng.normal(0, 1, 30)
    g2 = rng.normal(0.5, 1, 30)
    res = check_assumptions(g1, g2, test_type="t-test")
    assert "checks" in res
