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
    compare_with_reference,
    run_verification_suite,
    verify_drac_calculation,
    verify_pet_calculation,
    verify_ttc_calculation,
)
from .uncertainty_quantifier import (
    UncertaintyQuantifier,
    bootstrap_ci,
    compute_confidence_interval,
    compute_effect_size,
    compute_sample_size,
    monte_carlo_uq,
    sensitivity_analysis,
)

__all__ = [
    # SSM Verification
    "SSMVerifier",
    # Uncertainty Quantification
    "UncertaintyQuantifier",
    "bootstrap_ci",
    "compare_with_reference",
    "compute_confidence_interval",
    "compute_effect_size",
    "compute_sample_size",
    "monte_carlo_uq",
    "run_verification_suite",
    "sensitivity_analysis",
    "verify_drac_calculation",
    "verify_pet_calculation",
    "verify_ttc_calculation",
]

__version__ = "2.0.0"
