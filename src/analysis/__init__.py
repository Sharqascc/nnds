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

import importlib
import importlib.util
import logging
from typing import Any

logger = logging.getLogger(__name__)

__version__ = "1.2.0"
__author__ = "NNDS Team"

# Map public attribute names to their defining submodule.
_LAZY_IMPORTS = {
    "visualization": "src.analysis.visualization",
    "PETEventAnalyzer": "src.analysis.pet_summary",
}


# Compatibility flags for existing tests/tools (no heavy imports here)
_viz_available = importlib.util.find_spec('src.analysis.visualization') is not None
_pet_summary_available = importlib.util.find_spec('src.analysis.pet_summary') is not None

def __getattr__(name: str) -> Any:
    """Lazily import submodules/classes on first access."""
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise AttributeError(
                f"Could not lazily import '{name}' from '{module_path}': {e}"
            ) from e

        # If the attribute is a class inside the module, fetch it; otherwise the module itself
        attr = getattr(module, name) if name == "PETEventAnalyzer" else module

        globals()[name] = attr
        return attr

    raise AttributeError(f"module 'src.analysis' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))


def check_installation(use_logging: bool = True) -> dict[str, bool]:
    """Check which analysis modules are available."""
    status = {
        "visualization": False,
        "pet_summary": False,
    }

    # Visualisation subpackage
    try:
        importlib.import_module("src.analysis.visualization")
        status["visualization"] = True
    except ImportError:
        pass

    # PET summary module
    try:
        importlib.import_module("src.analysis.pet_summary")
        status["pet_summary"] = True
    except ImportError:
        pass

    if use_logging:
        logger.info("Analysis installation check: %s", status)

    return status
