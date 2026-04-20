"""SSM Verification & Validation module.

This package provides surrogate safety metric verification,
uncertainty quantification, and validation utilities.
"""

from .ssm_verification import SSMVerifier, verify_pet_threshold, verify_ttc_threshold
from .uncertainty_quantifier import UncertaintyQuantifier, monte_carlo_uq, bootstrap_ci

__all__ = [
    'SSMVerifier',
    'verify_pet_threshold',
    'verify_ttc_threshold',
    'UncertaintyQuantifier',
    'monte_carlo_uq',
    'bootstrap_ci',
]

__version__ = '1.0.0'
