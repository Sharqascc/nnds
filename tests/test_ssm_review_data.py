"""Verify the SSM review artifacts at data/reviews/ssm_review_114/.

The review underlies every PET/SSM quality number in docs/METRIC_STATUS.md.
Its verdict distribution is cited in that document and in REPRODUCE.md.
If the review file ever silently drifts, those citations become wrong.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

REVIEW_DIR = Path(__file__).resolve().parents[1] / "data" / "reviews" / "ssm_review_114"


def _load_to_label():
    path = REVIEW_DIR / "to_label.csv"
    if not path.exists():
        pytest.skip(f"review file not present: {path}")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_review_dir_exists():
    assert REVIEW_DIR.is_dir(), f"missing {REVIEW_DIR}"


def test_to_label_has_114_events():
    rows = _load_to_label()
    assert len(rows) == 114


def test_to_label_verdict_distribution():
    """38 Y, 76 N — the numbers cited in METRIC_STATUS.md."""
    rows = _load_to_label()
    counts: dict[str, int] = {}
    for r in rows:
        v = r["verdict"].strip()
        counts[v] = counts.get(v, 0) + 1
    assert counts == {"Y": 38, "N": 76}, counts


def test_to_label_columns():
    rows = _load_to_label()
    assert rows, "empty review file"
    assert set(rows[0].keys()) == {"idx", "track_a", "track_b", "pet", "frame", "verdict"}


def test_to_label_idx_is_contiguous():
    rows = _load_to_label()
    idxs = [int(r["idx"]) for r in rows]
    assert idxs == list(range(114))


def test_findings_doc_exists():
    findings = REVIEW_DIR / "findings.md"
    assert findings.is_file()
    text = findings.read_text()
    # canary: these numbers are cited in METRIC_STATUS.md
    assert "−0.106" in text or "-0.106" in text, "55° gate MCC not found in findings"
    assert "+0.367" in text, "2-feature model MCC not found in findings"
