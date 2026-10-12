"""Rigor gate: every collected test must be selected by at least one CI lane.

The CI workflow splits the suite into three parallel lanes by marker filter:

    fast     : not integration and not smoke and not full and not slow and not property
    property : property
    full     : integration or smoke or full or slow

As of this writing the three lanes are an exact partition of the 1979-collected
suite. That invariant is easy to break silently:

- A new test file named `test_heavy_smoke.py` does NOT end in `_smoke.py`, so
  tests/conftest.py will not auto-apply the `smoke` marker. The file would fall
  through every lane and never run in CI.
- A new convention (e.g. `test_*_integration.py`) that conftest does not know
  about would have the same problem.
- Editing the `-m` expression in .github/workflows/ci.yml without updating the
  corresponding pytest.ini markers would leave a marker filter matching nothing.

This test enumerates the actual collected node IDs from pytest --collect-only
under each lane's marker expression and asserts the union equals the full set.
It runs in the fast lane and executes in roughly 15 seconds.

If it fails, the fix is one of:
1. Rename the file so the conftest auto-marker recognizes it, or add an explicit
   `pytestmark = pytest.mark.smoke` (or whichever marker fits) at the top of it.
2. Add the file's convention to tests/conftest.py::pytest_collection_modifyitems.
3. Add a fourth lane to ci.yml and register it here in LANES.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Marker expression for each lane. Keep in sync with .github/workflows/ci.yml.
LANES = {
    "fast":     "not integration and not smoke and not full and not slow and not property",
    "property": "property",
    "full":     "integration or smoke or full or slow",
}

# Markers declared in pytest.ini under [pytest] markers = ...
DECLARED_MARKERS = {
    "property", "metamorphic", "differential",
    "integration", "slow", "smoke", "full",
}


def _collect(marker: str | None) -> set[str]:
    """Return the set of test node IDs pytest selects under `marker`."""
    cmd = [
        sys.executable, "-m", "pytest", "tests/",
        "--collect-only", "-q", "--no-header",
        "-p", "no:cacheprovider",
        "-p", "no:randomly",
    ]
    if marker:
        cmd += ["-m", marker]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    if r.returncode not in (0, 5):  # 5 = "no tests collected", which is fine
        raise AssertionError(
            "pytest --collect-only failed (rc={rc}) for marker={marker!r}\n"
            "stdout tail:\n{out}\nstderr tail:\n{err}".format(
                rc=r.returncode, marker=marker,
                out=r.stdout[-2000:], err=r.stderr[-2000:],
            )
        )
    return {
        ln.strip()
        for ln in r.stdout.splitlines()
        if "::" in ln and not ln.lstrip().startswith("=")
    }


def test_ci_lane_markers_are_declared_in_pytest_ini():
    """Every marker named in a lane's expression must be registered.

    Otherwise pytest emits PytestUnknownMarkWarning and the `-m` filter
    matches nothing, silently.
    """
    used = set()
    for expr in LANES.values():
        used |= set(re.findall(
            r"\b(integration|smoke|full|slow|property|metamorphic|differential)\b",
            expr,
        ))
    undeclared = used - DECLARED_MARKERS
    assert not undeclared, (
        "ci.yml lane filters reference markers not declared in pytest.ini: "
        + repr(sorted(undeclared))
    )


def test_every_test_is_selected_by_at_least_one_lane():
    """Union of lane selections must equal the full test suite.

    This is the rigor gate. It fails if a new test file falls through every
    lane (because of a filename convention none of the conftest auto-markers
    recognize), or if a lane's marker filter stops matching.
    """
    all_ids = _collect(None)
    covered: set[str] = set()
    per_lane: dict[str, int] = {}
    for lane, marker in LANES.items():
        ids = _collect(marker)
        per_lane[lane] = len(ids)
        covered |= ids

    missing = all_ids - covered
    msg_lines = [
        "{} test(s) are not selected by any CI lane.".format(len(missing)),
        "Total collected: {}; lane counts: {}".format(len(all_ids), per_lane),
        "Uncovered:",
    ]
    for t in sorted(missing)[:50]:
        msg_lines.append("  " + t)
    if len(missing) > 50:
        msg_lines.append("  ... and {} more".format(len(missing) - 50))
    msg_lines.append("")
    msg_lines.append(
        "Fix: rename the file to match an auto-marker convention, add an "
        "explicit pytestmark, update tests/conftest.py, or add a new lane."
    )
    assert not missing, "\n".join(msg_lines)


def test_lanes_do_not_overlap():
    """Lanes are an exact partition today; a regression here wastes CI time.

    Not a hard requirement — overlapping lanes are legal — but overlap means
    the same test runs in two jobs. If you intentionally add overlap, delete
    this test.
    """
    lane_sets = {lane: _collect(m) for lane, m in LANES.items()}
    seen: set[str] = set()
    overlaps: dict[str, list[str]] = {}
    for lane, ids in lane_sets.items():
        common = ids & seen
        if common:
            overlaps[lane] = sorted(common)
        seen |= ids
    assert not overlaps, (
        "CI lanes overlap on the same tests (wasteful duplication):\n"
        + "\n".join(
            "  {}: {}...".format(lane, tests[:5])
            for lane, tests in overlaps.items()
        )
        + "\nIf the overlap is intentional, remove test_lanes_do_not_overlap."
    )
