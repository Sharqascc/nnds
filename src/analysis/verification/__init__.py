"""Verification & Statistical Testing module.

Provides comprehensive statistical tests and verification utilities
for surrogate safety metrics.

Features:
- Parametric tests (t-test, ANOVA, paired tests)
- Non-parametric tests (Mann-Whitney, Kruskal-Wallis, Wilcoxon)
- Assumption checking (normality, homoscedasticity)
- Multiple testing corrections (Bonferroni, Holm, FDR)
- Effect size computation
- Post-hoc comparisons

Compliant with:
- APA statistical reporting standards
- Traffic safety research best practices
"""

from .statistical_testing import (
    StatisticalTester,
    check_assumptions,
    chi_square_test,
    multiple_comparisons,
    paired_test,
    test_drac_difference,
    test_pet_difference,
    test_ttc_difference,
)

__all__ = [
    # Main class
    "StatisticalTester",
    "check_assumptions",
    # General tests
    "chi_square_test",
    "multiple_comparisons",
    "paired_test",
    "test_drac_difference",
    # SSM-specific tests
    "test_pet_difference",
    "test_ttc_difference",
]

__version__ = "2.0.0"
