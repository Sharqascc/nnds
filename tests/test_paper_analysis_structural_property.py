"""Structural property tests for I/O-only paper.analysis modules."""

from __future__ import annotations

import importlib
import inspect

import pytest

MODULES = [
    "paper.analysis.extract_fp_list",
    "paper.analysis.lock_denominator",
    "paper.analysis.build_fp_clips",
    "paper.analysis.generate_fp_taxonomy",
]


@pytest.mark.parametrize("name", MODULES)
def test_has_callable_main(name):
    mod = importlib.import_module(name)
    assert hasattr(mod, "main")
    assert callable(mod.main)
    sig = inspect.signature(mod.main)
    required = [
        p
        for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    assert required == []


@pytest.mark.parametrize("name", MODULES)
def test_module_level_constants_are_paths_or_strs(name):
    mod = importlib.import_module(name)
    for attr in dir(mod):
        if attr.startswith("_") or not attr.isupper():
            continue
        val = getattr(mod, attr)
        assert val is not None
