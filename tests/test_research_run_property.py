"""Structural interface tests for src.analysis.research_run.

research_run is an orchestrator that shells out to subprocess. Property
testing is not applicable. These tests verify the module is importable
and its public interface is intact.
"""

from __future__ import annotations

import inspect

import pytest


def test_module_imports():
    from src.analysis import research_run

    assert hasattr(research_run, "log")
    assert hasattr(research_run, "run_cmd")
    assert hasattr(research_run, "main")


def test_defaults_are_strings():
    from src.analysis import research_run as rr

    assert isinstance(rr.DEFAULT_UVH_MODEL, str)
    assert isinstance(rr.DEFAULT_COCO_PERSON_MODEL, str)


def test_log_callable_with_levels():
    from src.analysis import research_run as rr

    rr.log("test message")
    rr.log("warn message", level="WARN")
    rr.log("err message", level="ERROR")


def test_run_cmd_signature():
    from src.analysis import research_run as rr

    sig = inspect.signature(rr.run_cmd)
    assert "cmd" in sig.parameters
    assert "cwd" in sig.parameters


def test_main_argv_default_is_none():
    from src.analysis import research_run as rr

    sig = inspect.signature(rr.main)
    assert "argv" in sig.parameters
    assert sig.parameters["argv"].default is None
