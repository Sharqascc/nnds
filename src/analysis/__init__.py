"""
Traffic Safety Analysis Module

Comprehensive analysis tools for traffic safety evaluation using:
- Surrogate Safety Measures (SSM)
- Statistical analysis
- Publication-quality visualization

Sub-modules:
- visualization: Publication-quality plots and figures
- pet_conflict_checker: PET computation and conflict detection
- ssm_verification: SSM verification functions
- uncertainty_quantifier: Uncertainty quantification
"""

from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Visualization sub-package
# -------------------------------------------------------------------------
try:
    from . import visualization

    _viz_available = True
except ImportError as e:  # pragma: no cover
    _viz_available = False  # pragma: no cover
    import warnings  # pragma: no cover

    warnings.warn(f"Visualization module not available: {e}")  # pragma: no cover
    visualization = None  # pragma: no cover

# -------------------------------------------------------------------------
# PET summary analysis
# -------------------------------------------------------------------------
try:
    from .pet_summary import PETEventAnalyzer

    _pet_summary_available = True
except ImportError as e:  # pragma: no cover
    _pet_summary_available = False  # pragma: no cover
    import warnings  # pragma: no cover

    warnings.warn(f"PET summary module not available: {e}")  # pragma: no cover
    PETEventAnalyzer = None  # pragma: no cover

# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------

__all__ = []

if _viz_available:
    __all__.append("visualization")

if _pet_summary_available:
    __all__.append("PETEventAnalyzer")

# Module metadata
__version__ = "1.2.0"
__author__ = "NNDS Team"


def check_installation(use_logging: bool = True) -> dict[str, bool]:
    """Check which analysis modules are available."""
    status = {  # pragma: no cover
        "visualization": _viz_available,  # pragma: no cover
        "pet_summary": _pet_summary_available,  # pragma: no cover
    }  # pragma: no cover
    return status  # pragma: no cover
