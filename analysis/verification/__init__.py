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
    test_pet_difference,
    test_ttc_difference,
    test_drac_difference,
    chi_square_test,
    paired_test,
    multiple_comparisons,
    check_assumptions,
)

__all__ = [
    # Main class
    'StatisticalTester',
    
    # SSM-specific tests
    'test_pet_difference',
    'test_ttc_difference',
    'test_drac_difference',
    
    # General tests
    'chi_square_test',
    'paired_test',
    'multiple_comparisons',
    'check_assumptions',
]

__version__ = '2.0.0'
