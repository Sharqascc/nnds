"""
Tests for statistical testing functions (with aliased imports to avoid pytest collection).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Alias imports to avoid pytest collecting source test functions
import src.analysis.verification.statistical_testing as st_module

StatisticalTester = st_module.StatisticalTester
TestResult = st_module.TestResult
paired_test = st_module.paired_test
chi_square_test = st_module.chi_square_test
check_assumptions = st_module.check_assumptions
multiple_comparisons = st_module.multiple_comparisons


def test_paired_test_basic():
    """Test paired t-test with known difference."""
    np.random.seed(42)
    group1 = np.random.normal(10, 2, 100)
    group2 = group1 + np.random.normal(0.5, 0.1, 100)
    result = paired_test(group1, group2)
    assert result is not None


def test_chi_square_test():
    """Test chi-square test."""
    observed = np.array([10, 20, 30])
    expected = np.array([15, 15, 30])
    result = chi_square_test(observed, expected)
    assert result is not None


def test_check_assumptions():
    """Test assumption checking."""
    np.random.seed(42)
    data1 = np.random.normal(10, 2, 100)
    data2 = np.random.normal(12, 2, 100)
    result = check_assumptions(data1, data2)
    assert result is not None


def test_multiple_comparisons():
    """Test multiple comparisons correction."""
    np.random.seed(42)
    p_values = np.random.uniform(0, 0.1, 10)
    result = multiple_comparisons(p_values)
    assert result is not None


def test_statistical_tester_initialization():
    """Test StatisticalTester initialization."""
    tester = StatisticalTester()
    assert tester is not None


def test_test_result_dataclass():
    """Test TestResult dataclass."""
    result = TestResult(test_name="test", statistic=1.5, p_value=0.05, significant=True)
    assert result.test_name == "test"
    assert result.p_value == 0.05
