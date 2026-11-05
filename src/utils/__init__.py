"""Utility helpers.

Submodules are imported directly by callers:

    from src.utils.seed import set_seed, get_seed
    from src.utils.duration_parser import parse_duration

This module intentionally re-exports nothing: `src.utils.seed` imports
torch, and eager re-export here would make `import src.utils` pay the
torch import cost even for callers that only want, e.g., duration parsing.
"""

from __future__ import annotations

__all__: list[str] = []
