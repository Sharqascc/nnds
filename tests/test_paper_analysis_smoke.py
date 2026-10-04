"""Smoke-test every paper.analysis module.

Catches syntax errors, missing imports, renamed helpers. Runs in CI on
every push, so a broken generator is caught before it silently writes
a wrong artifact.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULES = sorted(
    p.stem
    for p in (REPO / "paper" / "analysis").glob("*.py")
    if p.stem != "__init__"
)


def test_modules_discovered():
    assert MODULES, "no modules found under paper/analysis/"


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    mod = importlib.import_module(f"paper.analysis.{name}")
    assert mod is not None
