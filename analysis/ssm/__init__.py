"""SSM Verification & Validation module.

This package provides surrogate safety metric verification,
uncertainty quantification, and validation utilities.

Complies with:
- FHWA traffic safety analysis standards
- Statistical verification best practices
- APA reporting guidelines
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

from .uncertainty_quantifier import (
    UncertaintyQuantifier,
    bootstrap_ci,
    monte_carlo_uq,
    compute_confidence_interval,
    compute_effect_size,
    compute_sample_size,
    sensitivity_analysis
)


__all__ = [
    # SSM Verification
    'SSMVerifier',
    'verify_pet_calculation',
    'verify_ttc_calculation',
    'verify_drac_calculation',
    'run_verification_suite',
    'compare_with_reference',
    
    # Uncertainty Quantification
    'UncertaintyQuantifier',
    'bootstrap_ci',
    'monte_carlo_uq',
    'compute_confidence_interval',
    'compute_effect_size',
    'compute_sample_size',
    'sensitivity_analysis',
]

__version__ = '2.0.0'
