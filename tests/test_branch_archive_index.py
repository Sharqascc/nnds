"""Verify docs/BRANCH_ARCHIVE.md stays in sync with actual archive/* tags.

The doc is hand-maintained. When a branch is retired, a row is added.
This test catches:

1. A tag is created but no row is added (archive without documenting).
2. A row exists but the tag doesn't (typo, or tag deleted).
3. The sha in the doc doesn't match the actual tag sha.
4. The table format is broken (missing columns, duplicate tag name).

The tag comparison only runs when archive/* tags are available locally.
Shallow clones with no tags still get the structural checks.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parents[1] / "docs" / "BRANCH_ARCHIVE.md"

TAG_RE = re.compile(r"^archive/[a-z0-9][a-z0-9-]*$")
SHA_RE = re.compile(r"^[0-9a-f]{12}$")
# | `archive/name` | `sha12` | description |
ROW_RE = re.compile(r"^[|] *`(archive/[^`]+)` *[|] *`([0-9a-f]+)` *[|]")


def _parse_doc():
    """Return {tag: sha_prefix} from the Current archives table."""
    text = DOC.read_text()
    in_table = False
    rows = {}
    for line in text.splitlines():
        if line.strip().startswith("## Current archives"):
            in_table = True
            continue
        if in_table and line.startswith("## "):
            break
        if in_table:
            m = ROW_RE.match(line)
            if m:
                tag, sha = m.group(1), m.group(2)
                if tag in rows:
                    raise AssertionError(f"duplicate tag in doc: {tag}")
                rows[tag] = sha
    return rows


def _local_archive_tags():
    """Return {tag: full_sha} for locally-available archive tags."""
    r = subprocess.run(
        ["git", "for-each-ref", "--format=%(refname:short) %(objectname)", "refs/tags/archive/*"],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        return {}
    out = {}
    for line in r.stdout.strip().splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


def test_doc_exists():
    assert DOC.exists(), f"missing: {DOC}"


def test_doc_table_parses_and_is_well_formed():
    rows = _parse_doc()
    assert rows, "no archive rows parsed from the doc table"
    for tag, sha in rows.items():
        assert TAG_RE.match(tag), f"bad tag name in doc: {tag!r}"
        assert SHA_RE.match(sha), f"bad sha in doc for {tag!r}: {sha!r}"


def test_doc_matches_local_tags_when_available():
    """Bidirectional drift check.

    Skipped when no archive/* tags are present locally (shallow clone).
    To force it locally: `git fetch --tags`.
    """
    rows = _parse_doc()
    local = _local_archive_tags()
    if not local:
        pytest.skip("no archive/* tags locally — run `git fetch --tags`")

    missing_from_doc = set(local) - set(rows)
    missing_from_repo = set(rows) - set(local)

    assert not missing_from_doc, (
        "tags exist but are not documented in BRANCH_ARCHIVE.md: "
        + ", ".join(sorted(missing_from_doc))
    )
    assert not missing_from_repo, (
        "tags listed in BRANCH_ARCHIVE.md but not present locally: "
        + ", ".join(sorted(missing_from_repo))
    )

    for tag, doc_sha in rows.items():
        actual = local[tag]
        assert actual.startswith(doc_sha), (
            f"doc lists {tag} at {doc_sha} but tag is at {actual[:12]}"
        )
