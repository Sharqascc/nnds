"""Verify that docs/API.md matches what generate_api_docs.py produces.

The generator parses source AST, not importing any modules. That makes it
fast (~100 ms) and independent of which optional dependencies are installed,
so the drift check runs in the fast CI lane.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


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
        return

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
