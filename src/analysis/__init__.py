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
except ImportError as e:
    _viz_available = False
    import warnings
    warnings.warn(f"Visualization module not available: {e}")
    visualization = None

# -------------------------------------------------------------------------
# PET summary analysis
# -------------------------------------------------------------------------
try:
    from .pet_summary import PETEventAnalyzer
    _pet_summary_available = True
except ImportError as e:
    _pet_summary_available = False
    import warnings
    warnings.warn(f"PET summary module not available: {e}")
    PETEventAnalyzer = None

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
    status = {
        "visualization": _viz_available,
        "pet_summary": _pet_summary_available,
    }
    return status
