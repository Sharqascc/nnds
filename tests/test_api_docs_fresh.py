"""Verify that docs/API.md matches what generate_api_docs.py produces.

Marked ``slow`` so it runs in the full-or-slow CI shard instead of the fast
lane: the generator imports every public module (including torch-adjacent
ones) to introspect them, which takes ~5-10 seconds. That cost does not
belong in the fast lane, but the drift check does need to run on every PR.

If this test fails, regenerate and commit:

    python scripts/generate_api_docs.py
    git add docs/API.md
    git commit -m "docs: regenerate API.md"
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.slow


def _generate() -> str:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from scripts.generate_api_docs import generate

    return generate()


def test_api_md_exists() -> None:
    assert (REPO / "docs" / "API.md").exists(), (
        "docs/API.md is missing. Run `python scripts/generate_api_docs.py` and commit the result."
    )


def test_api_md_is_up_to_date() -> None:
    api_path = REPO / "docs" / "API.md"
    if not api_path.exists():
        pytest.skip("docs/API.md missing - covered by test_api_md_exists")

    committed = api_path.read_text()
    generated = _generate()
    if committed == generated:
        return

    c_lines = committed.splitlines()
    g_lines = generated.splitlines()
    diff_at = next(
        (i for i, (a, b) in enumerate(zip(c_lines, g_lines, strict=False), start=1) if a != b),
        min(len(c_lines), len(g_lines)) + 1,
    )
    c_line = c_lines[diff_at - 1] if diff_at <= len(c_lines) else "<end>"
    g_line = g_lines[diff_at - 1] if diff_at <= len(g_lines) else "<end>"
    raise AssertionError(
        f"docs/API.md is out of date (first difference at line {diff_at}).\n"
        f"  committed: {c_line!r}\n"
        f"  generated: {g_line!r}\n\n"
        f"Regenerate and commit:\n"
        f"    python scripts/generate_api_docs.py\n"
        f"    git add docs/API.md"
    )
