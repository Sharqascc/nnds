"""SSM Verification & Validation module.

This package provides surrogate safety metric verification,
uncertainty quantification, and validation utilities.

Complies with:
- FHWA traffic safety analysis standards
- Statistical verification best practices
- Research reproducibility requirements
"""

from .ssm_verification import (
    SSMVerifier,
    verify_pet_calculation,
    verify_ttc_calculation,
    verify_drac_calculation,
    run_verification_suite,
    compare_with_reference
)

# Only import if uncertainty_quantifier.py exists
try:
    from .uncertainty_quantifier import (
        UncertaintyQuantifier,
        monte_carlo_uq,
        bootstrap_ci
    )
    _HAS_UQ = True
except ImportError:
    _HAS_UQ = False
    UncertaintyQuantifier = None
    monte_carlo_uq = None
    bootstrap_ci = None


__all__ = [
    # SSM Verification
    'SSMVerifier',
    'verify_pet_calculation',
    'verify_ttc_calculation',
    'verify_drac_calculation',
    'run_verification_suite',
    'compare_with_reference',
]

# Add UQ exports if available
if _HAS_UQ:
    __all__.extend([
        'UncertaintyQuantifier',
        'monte_carlo_uq',
        'bootstrap_ci',
    ])

__version__ = '2.0.0'  # Updated for new verification suite
