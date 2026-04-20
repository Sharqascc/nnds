"""Logging & Reproducibility module.

Provides reproducibility auditing and experiment logging.
"""

from .reproducibility_audit import (
    ReproducibilityAuditor,
    generate_audit_report,
    hash_file,
    audit_environment,
)

__all__ = [
    'ReproducibilityAuditor',
    'generate_audit_report',
    'hash_file',
    'audit_environment',
]

__version__ = '1.0.0'
