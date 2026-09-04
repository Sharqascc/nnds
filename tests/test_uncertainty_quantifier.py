"""
Tests for uncertainty quantifier functions.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.ssm.uncertainty_quantifier import (
    bootstrap_ci,
    compute_confidence_interval,
    compute_effect_size,
    compute_sample_size,
    monte_carlo_uq,
    sensitivity_analysis,
)


def test_bootstrap_ci_basic():
    """Test bootstrap confidence interval."""
    np.random.seed(42)
    data = np.random.normal(5, 2, 100)
    ci = bootstrap_ci(data)
    assert ci is not None
    assert len(ci) == 2
    assert ci[0] < ci[1]


def test_bootstrap_ci_contains_mean():
    """Test that bootstrap CI contains the true mean."""
    np.random.seed(42)
    data = np.random.normal(5, 2, 1000)
    ci = bootstrap_ci(data, n_bootstrap=1000)
    assert ci[0] < 5.0 < ci[1]


def test_compute_confidence_interval():
    """Test parametric confidence interval."""
    np.random.seed(42)
    data = np.random.normal(10, 2, 100)
    ci = compute_confidence_interval(data, confidence=0.95)
    assert ci is not None
    assert len(ci) == 2
    assert ci[0] < ci[1]


def test_compute_effect_size():
    """Test effect size computation."""
    np.random.seed(42)
    group1 = np.random.normal(10, 2, 50)
    group2 = np.random.normal(12, 2, 50)
    effect = compute_effect_size(group1, group2)
    assert effect is not None
    assert isinstance(effect, float)


def test_compute_sample_size():
    """Test sample size computation."""
    n = compute_sample_size(effect_size=0.5, power=0.8, alpha=0.05)
    assert n is not None
    assert isinstance(n, int)
    assert n > 0


def test_monte_carlo_uq():
    """Test Monte Carlo uncertainty quantification."""

    def model_fn(x):
        return x**2 + 2 * x

    input_distributions = {
        "x": ("normal", (5.0, 1.0))  # mean=5, std=1
    }
    result = monte_carlo_uq(input_distributions, model_fn, n_samples=100)
    assert result is not None
    assert isinstance(result, dict)


def test_sensitivity_analysis():
    """Test sensitivity analysis."""

    # Model function must accept keyword arguments
    def model_fn(a, b):
        return a * 2 + b * 3

    baseline_params = {"a": 1.0, "b": 2.0}
    param_ranges = {"a": (0.5, 1.5), "b": (1.0, 3.0)}
    result = sensitivity_analysis(baseline_params, model_fn, param_ranges, n_steps=10)
    assert result is not None
    assert isinstance(result, dict)
    assert "a" in result
    assert "b" in result


def test_uncertainty_quantifier_class():
    """Test UncertaintyQuantifier class."""
    from src.analysis.ssm.uncertainty_quantifier import UncertaintyQuantifier

    quantifier = UncertaintyQuantifier()
    assert quantifier is not None
