"""Enforce the two-tier dependency pinning policy.

Two policies, one per file kind:

* ``requirements*.txt`` — exact pins (``==X.Y.Z``). These are what CI installs,
  what a reviewer reproduces with, and what the coverage badge reflects. A
  silent version shift in these files is precisely the "works on my machine"
  failure mode the reviewer flagged.

* ``pyproject.toml`` ``[project]`` and ``[project.optional-dependencies]`` —
  bounded ranges (``>=X.Y.Z,<next_minor``). Library consumers get patch
  upgrades but not minor bumps that could change behavior between installs.

If you add a dependency, follow the corresponding policy. If you relax a pin,
this test fails, which is the intended friction: the version list is part of
the reproducibility contract.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

try:
    import tomllib
except ImportError:  # pragma: no cover — Python < 3.11
    pytest.skip("requires Python 3.11+ for tomllib", allow_module_level=True)

REPO = Path(__file__).resolve().parents[1]


def _parse_requirements_file(path: Path) -> list[str]:
    """Return the non-comment, non-`-r` dependency lines in a requirements file."""
    deps = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-r"):
            continue
        line = line.split("#", 1)[0].strip()
        if line:
            deps.append(line)
    return deps


def _vkey(v: str) -> tuple[int, ...]:
    """(major, minor, patch) tuple from a version string, ignoring build numbers."""
    return tuple(int(p) for p in re.findall(r"\d+", v)[:3])


@pytest.mark.parametrize(
    "fname",
    ["requirements.txt", "requirements-dev.txt", "requirements-vlm.txt"],
)
def test_requirements_files_are_exactly_pinned(fname: str) -> None:
    path = REPO / fname
    if not path.exists():
        pytest.skip(f"{fname} not present")
    for dep in _parse_requirements_file(path):
        assert "==" in dep, (
            f"{fname}: '{dep}' is not exactly pinned. Every dependency in a "
            f"requirements file must use '=='. See tests/test_dependency_pins.py "
            f"for the policy."
        )
        assert ">=" not in dep, f"{fname}: '{dep}' mixes '==' with '>='"
        assert "~=" not in dep, f"{fname}: '{dep}' mixes '==' with '~='"


def test_pyproject_runtime_deps_are_bounded() -> None:
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    deps = data["project"]["dependencies"]
    assert deps, "pyproject.toml [project].dependencies is empty"
    for dep in deps:
        assert ">=" in dep and "<" in dep, (
            f"pyproject.toml runtime dep '{dep}' is not bounded. "
            f"Expected form: 'name>=X.Y.Z,<X.Y+1'."
        )


def test_pyproject_dev_deps_are_bounded() -> None:
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    dev = data["project"]["optional-dependencies"]["dev"]
    assert dev, "pyproject.toml [project.optional-dependencies].dev is empty"
    for dep in dev:
        assert ">=" in dep and "<" in dep, (
            f"pyproject.toml dev dep '{dep}' is not bounded. Expected form: 'name>=X.Y.Z,<X.Y+1'."
        )


def test_requirements_txt_pins_satisfy_pyproject_ranges() -> None:
    """Every pin in requirements.txt must fall inside pyproject's bound for that name."""
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    runtime = data["project"]["dependencies"]

    ranges: dict[str, tuple[str, str]] = {}
    for dep in runtime:
        m = re.match(
            r"([a-zA-Z0-9_.\-]+)\s*>=\s*([\d.]+)\s*,\s*<\s*([\d.]+)",
            dep,
        )
        assert m, f"pyproject dep '{dep}' does not match expected form"
        ranges[m.group(1).lower()] = (m.group(2), m.group(3))

    for dep in _parse_requirements_file(REPO / "requirements.txt"):
        m = re.match(r"([a-zA-Z0-9_.\-]+)\s*==\s*([\d.]+)", dep)
        assert m, f"requirements.txt line '{dep}' does not match name==version"
        name, ver = m.group(1).lower(), m.group(2)
        assert name in ranges, (
            f"'{name}' is pinned in requirements.txt but missing from "
            f"pyproject.toml [project].dependencies"
        )
        lo, hi = ranges[name]
        assert _vkey(lo) <= _vkey(ver) < _vkey(hi), (
            f"{name}: requirements.txt pins {ver}, but pyproject allows "
            f"[{lo}, {hi}). Update both in the same PR."
        )
