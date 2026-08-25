"""Logging & Reproducibility module.

Provides reproducibility auditing and experiment logging for the NNDS pipeline.
Ensures full compliance with ACM Artifact Evaluation and FAIR data principles.
"""

from .reproducibility_audit import (
    ReproducibilityAuditor,
    audit_environment,
    generate_audit_report,
    hash_file,
    verify_reproducibility,  # ← MISSING - needs to be added!
)

__all__ = [
    "ReproducibilityAuditor",
    "audit_environment",
    "generate_audit_report",
    "hash_file",
    "verify_reproducibility",  # ← MISSING - needs to be added!
]

__version__ = "2.1.0"  # ← Updated to match your v2.1 implementation
