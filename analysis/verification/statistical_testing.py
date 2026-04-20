"""Statistical Testing Module for SSM Validation.

Provides t-tests, ANOVA, chi-square tests, and specialized
PET/TTC difference tests for safety metric validation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats


@dataclass
class TestResult:
    """Container for statistical test results."""
    test: str
    statistic: float
    p_value: float
    significant: bool
    n_groups: int = 1
    n_observations: int = 0
    summary: str = ""
    error: str = ""
    passed: bool = True


class StatisticalTester:
    """Statistical testing utilities for safety metrics.
    
    Provides t-tests, ANOVA, and chi-square tests
    with safety-metric-specific convenience methods.
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def _clean_data(self, *arrays) -> List[np.ndarray]:
        """Remove NaN values from arrays."""
        return [np.asarray(a)[~np.isnan(np.asarray(a))] for a in arrays]
    
    def t_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        equal_var: bool = True,
    ) -> TestResult:
        """Two-sample t-test between two groups.
        
        Args:
            group1: First group of values
            group2: Second group of values
            equal_var: Assume equal variance (default True)
            
        Returns:
            TestResult with t-statistic and p-value
        """
        g1, g2 = self._clean_data(group1, group2)
        
        if len(g1) < 2 or len(g2) < 2:
            return TestResult(
                test="t-test",
                statistic=np.nan,
                p_value=np.nan,
                significant=False,
                n_groups=2,
                n_observations=len(g1) + len(g2),
                error="Need at least 2 observations per group",
                passed=False,
            )
        
        t_stat, p_value = stats.ttest_ind(g1, g2, equal_var=equal_var)
        significant = p_value < self.alpha
        
        return TestResult(
            test="t-test",
            statistic=float(t_stat),
            p_value=float(p_value),
            significant=significant,
            n_groups=2,
            n_observations=len(g1) + len(g2),
            summary=f"t={t_stat:.3f}, p={p_value:.4f} "
                    f"({'significant' if significant else 'not significant'})",
        )
    
    def anova(self, *groups) -> TestResult:
        """One-way ANOVA across multiple groups.
        
        Args:
            *groups: Variable number of group arrays
            
        Returns:
            TestResult with F-statistic and p-value
        """
        if len(groups) < 2:
            return TestResult(
                test="ANOVA",
                statistic=np.nan,
                p_value=np.nan,
                significant=False,
                n_groups=len(groups),
                error="Need at least 2 groups",
                passed=False,
            )
        
        clean_groups = self._clean_data(*groups)
        if any(len(g) < 2 for g in clean_groups):
            return TestResult(
                test="ANOVA",
                statistic=np.nan,
                p_value=np.nan,
                significant=False,
                n_groups=len(groups),
                error="All groups need at least 2 observations",
                passed=False,
            )
        
        f_stat, p_value = stats.f_oneway(*clean_groups)
        significant = p_value < self.alpha
        
        return TestResult(
            test="One-way ANOVA",
            statistic=float(f_stat),
            p_value=float(p_value),
            significant=significant,
            n_groups=len(groups),
            n_observations=sum(len(g) for g in clean_groups),
            summary=f"F={f_stat:.2f}, p={p_value:.4f} "
                    f"({'significant' if significant else 'not significant'})",
        )
    
    def chi_square_test(
        self,
        observed: np.ndarray,
        expected: Optional[np.ndarray] = None,
    ) -> TestResult:
        """Chi-square goodness-of-fit test.
        
        Args:
            observed: Observed frequencies
            expected: Expected frequencies (optional)
            
        Returns:
            TestResult with chi-square statistic and p-value
        """
        observed = np.asarray(observed)
        
        if len(observed) < 2:
            return TestResult(
                test="chi-square",
                statistic=np.nan,
                p_value=np.nan,
                significant=False,
                n_observations=len(observed),
                error="Need at least 2 categories",
                passed=False,
            )
        
        if expected is not None:
            expected = np.asarray(expected)
            chi2, p_value = stats.chisquare(observed, f_exp=expected)
        else:
            chi2, p_value = stats.chisquare(observed)
        
        significant = p_value < self.alpha
        
        return TestResult(
            test="chi-square",
            statistic=float(chi2),
            p_value=float(p_value),
            significant=significant,
            n_observations=len(observed),
            summary=f"chi2={chi2:.2f}, p={p_value:.4f} "
                    f"({'significant' if significant else 'not significant'})",
        )


def test_pet_difference(
    pet_group1: np.ndarray,
    pet_group2: np.ndarray,
    alpha: float = 0.05,
) -> TestResult:
    """Test difference in PET between two conditions."""
    return StatisticalTester(alpha=alpha).t_test(pet_group1, pet_group2)


def test_ttc_difference(
    ttc_group1: np.ndarray,
    ttc_group2: np.ndarray,
    alpha: float = 0.05,
) -> TestResult:
    """Test difference in TTC between two conditions."""
    return StatisticalTester(alpha=alpha).t_test(ttc_group1, ttc_group2)


def chi_square_test(
    observed: np.ndarray,
    expected: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> TestResult:
    """Convenience function for chi-square test."""
    return StatisticalTester(alpha=alpha).chi_square_test(observed, expected)
