"""Verification & Statistical Testing module.

Provides statistical tests and verification utilities for safety metrics.
"""

from .statistical_testing import (
    StatisticalTester,
    t_test,
    anova,
    chi_square_test,
    test_pet_difference,
    test_ttc_difference,
)

__all__ = [
    'StatisticalTester',
    't_test',
    'anova',
    'chi_square_test',
    'test_pet_difference',
    'test_ttc_difference',
]

__version__ = '1.0.0'
