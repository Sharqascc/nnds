import numpy as np
import pytest

from src.analysis.ssm.uncertainty_quantifier import (
    UncertaintyQuantifier,
    bootstrap_ci,
    compute_confidence_interval,
    compute_effect_size,
    compute_sample_size,
    monte_carlo_uq,
    sensitivity_analysis,
)


# ---------- analyze ----------
def test_analyze_empty_data():
    uq = UncertaintyQuantifier(n_bootstrap=100)
    res = uq.analyze(np.array([]), name="empty")
    assert not res["passed"]
    assert "No valid data" in res["errors"][0]


def test_analyze_unknown_method():
    uq = UncertaintyQuantifier()
    with pytest.raises(ValueError):
        uq.analyze(np.array([1, 2, 3]), method="bogus")


def test_analyze_small_sample_warning():
    uq = UncertaintyQuantifier(n_bootstrap=50)
    data = np.array([1, 2, 3])
    res = uq.analyze(data, method="bootstrap")
    assert any("Small sample size" in w for w in res["warnings"])


def test_analyze_high_cv_and_wide_ci():
    uq = UncertaintyQuantifier(n_bootstrap=50)
    # Data with large CV and wide CI relative to mean
    data = np.array([0.001, 0.01, 0.1])
    res = uq.analyze(data, method="bootstrap")
    warnings = " ".join(res["warnings"])
    assert "High variability" in warnings
    assert "Wide confidence interval" in warnings


def test_analyze_distribution_when_n_ge_8():
    uq = UncertaintyQuantifier(n_bootstrap=50)
    data = np.linspace(0.5, 2.0, 10)
    res = uq.analyze(data, method="parametric")
    assert "distribution" in res
    assert "normality_p" in res["distribution"]


# ---------- bootstrap_ci ----------
def test_bootstrap_ci_percentile():
    uq = UncertaintyQuantifier(confidence_level=0.95, n_bootstrap=100, random_state=0)
    data = np.random.default_rng(42).normal(0, 1, 30)
    ci = uq.bootstrap_ci(data, method="percentile")
    assert ci[0] < ci[1]


def test_bootstrap_ci_bca():
    uq = UncertaintyQuantifier(confidence_level=0.95, n_bootstrap=100, random_state=0)
    data = np.random.default_rng(42).normal(0, 1, 30)
    ci = uq.bootstrap_ci(data, method="bca")
    assert ci[0] < ci[1]


def test_bootstrap_ci_basic():
    uq = UncertaintyQuantifier(confidence_level=0.95, n_bootstrap=100, random_state=0)
    data = np.random.default_rng(42).normal(0, 1, 30)
    ci = uq.bootstrap_ci(data, method="basic")
    assert ci[0] < ci[1]


def test_bootstrap_ci_unknown_method():
    uq = UncertaintyQuantifier(n_bootstrap=10)
    with pytest.raises(ValueError):
        uq.bootstrap_ci(np.array([1, 2, 3]), method="bogus")


# ---------- monte_carlo_propagation ----------
def test_monte_carlo_unknown_distribution():
    uq = UncertaintyQuantifier(n_bootstrap=10)
    with pytest.raises(ValueError):
        uq.monte_carlo_propagation({"x": ("unknown", (0, 1))}, lambda **kw: 1.0, n_samples=5)


def test_monte_carlo_normal_uniform_lognormal():
    uq = UncertaintyQuantifier(n_bootstrap=50)
    result = uq.monte_carlo_propagation(
        {"a": ("normal", (5, 1)), "b": ("uniform", (1, 2)), "c": ("lognormal", (0, 1))},
        lambda **kw: kw["a"] + kw["b"] + kw["c"],
        n_samples=20,
    )
    assert result["passed"]


# ---------- compute_effect_size ----------
def test_compute_effect_size_one_sample():
    uq = UncertaintyQuantifier(n_bootstrap=50, random_state=0)
    data = np.random.default_rng(42).normal(0, 1, 30)
    res = uq.compute_effect_size(data)
    assert "estimate" in res
    assert "ci_lower" in res


def test_compute_effect_size_hedges_g():
    uq = UncertaintyQuantifier(n_bootstrap=50, random_state=0)
    g1 = np.random.default_rng(42).normal(0, 1, 20)
    g2 = np.random.default_rng(43).normal(0.5, 1, 20)
    res = uq.compute_effect_size(g1, g2, estimator="hedges_g")
    assert res["estimator"] == "hedges_g"


def test_compute_effect_size_glass_delta():
    uq = UncertaintyQuantifier(n_bootstrap=50, random_state=0)
    g1 = np.random.default_rng(42).normal(0, 1, 20)
    g2 = np.random.default_rng(43).normal(0.5, 1, 20)
    res = uq.compute_effect_size(g1, g2, estimator="glass_delta")
    assert res["estimator"] == "glass_delta"


def test_compute_effect_size_unknown_estimator():
    uq = UncertaintyQuantifier(n_bootstrap=10)
    with pytest.raises(ValueError):
        uq.compute_effect_size(np.array([1, 2]), np.array([2, 3]), estimator="bogus")


# ---------- compute_required_sample_size ----------
def test_compute_sample_size_two_sided():
    uq = UncertaintyQuantifier()
    n = uq.compute_required_sample_size(
        effect_size=0.5, power=0.8, alpha=0.05, test_type="two-sided"
    )
    assert n > 0
    assert isinstance(n, int)


def test_compute_sample_size_one_sided():
    uq = UncertaintyQuantifier()
    n = uq.compute_required_sample_size(
        effect_size=0.5, power=0.8, alpha=0.05, test_type="one-sided"
    )
    assert n > 0


# ---------- convenience functions ----------
def test_standalone_bootstrap_ci():
    data = np.random.default_rng(42).normal(0, 1, 30)
    ci = bootstrap_ci(data, confidence=0.95, n_bootstrap=50, method="percentile")
    assert ci[0] < ci[1]


def test_standalone_monte_carlo_uq():
    result = monte_carlo_uq(
        {"x": ("normal", (5, 1))},
        lambda **kw: kw["x"] * 2,
        n_samples=20,
    )
    assert "point_estimates" in result


def test_compute_confidence_interval_parametric():
    data = np.random.default_rng(42).normal(0, 1, 30)
    ci = compute_confidence_interval(data, confidence=0.95, method="parametric")
    assert ci[0] < ci[1]


def test_compute_confidence_interval_bootstrap():
    data = np.random.default_rng(42).normal(0, 1, 30)
    ci = compute_confidence_interval(data, confidence=0.95, method="percentile")
    assert ci[0] < ci[1]


def test_standalone_compute_effect_size():
    g1 = np.random.default_rng(42).normal(0, 1, 20)
    g2 = np.random.default_rng(43).normal(0.5, 1, 20)
    d = compute_effect_size(g1, g2)
    assert isinstance(d, float)


def test_standalone_compute_sample_size():
    n = compute_sample_size(effect_size=0.5, power=0.8)
    assert n > 0


def test_sensitivity_analysis_normal():
    def model(**kw):
        return kw["a"] + 2 * kw["b"]

    result = sensitivity_analysis(
        baseline_params={"a": 1.0, "b": 2.0},
        model_function=model,
        param_ranges={"a": (0, 2), "b": (1, 3)},
        n_steps=5,
    )
    assert set(result.keys()) == {"a", "b"}
    assert "sensitivity" in result["a"]


def test_sensitivity_analysis_zero_mean():
    def model(**kw):
        return 0.0  # mean output zero -> sensitivity guard

    result = sensitivity_analysis(
        baseline_params={"x": 1.0},
        model_function=model,
        param_ranges={"x": (0, 2)},
        n_steps=3,
    )
    assert result["x"]["sensitivity"] == 0


def test_analyze_method_bca():
    uq = UncertaintyQuantifier(confidence_level=0.95, n_bootstrap=100, random_state=0)
    data = np.random.default_rng(42).normal(0, 1, 30)
    res = uq.analyze(data, name="bca_test", method="bca")
    assert res["method"] == "bca"
    assert "confidence_interval" in res
