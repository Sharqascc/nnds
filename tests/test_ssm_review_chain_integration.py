"""Integration: SSM review label -> mapping -> audit chain is consistent.

No heavy assets. Runs in PR CI.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
REV = REPO / "data" / "reviews" / "ssm_review_114"


def _skip_if_missing(paths):
    for p in paths:
        if not Path(p).exists():
            pytest.skip(f"artifact missing: {p}")


def test_review_dir_has_expected_files():
    expected = ["to_label.csv", "label_to_giti_mapping.csv",
                "features.csv", "all_features.csv", "ev_jumpstats.csv"]
    for name in expected:
        assert (REV / name).exists(), f"missing {name}"


def test_labels_have_114_rows():
    _skip_if_missing([REV / "to_label.csv"])
    labels = pd.read_csv(REV / "to_label.csv")
    assert len(labels) == 114
    assert set(labels.verdict.unique()).issubset({"Y", "N"})


def test_label_verdicts_are_38_y_76_n():
    _skip_if_missing([REV / "to_label.csv"])
    labels = pd.read_csv(REV / "to_label.csv")
    y = (labels.verdict == "Y").sum()
    n = (labels.verdict == "N").sum()
    assert y == 38, f"expected 38 Y, got {y}"
    assert n == 76, f"expected 76 N, got {n}"


def test_mapping_has_103_rows_after_screen():
    _skip_if_missing([REV / "label_to_giti_mapping.csv"])
    mapping = pd.read_csv(REV / "label_to_giti_mapping.csv")
    assert len(mapping) == 103
    assert mapping.label_idx.is_monotonic_increasing
    assert mapping.giti_idx.is_monotonic_increasing


def test_mapping_indices_reference_valid_screened_rows():
    _skip_if_missing([
        REV / "label_to_giti_mapping.csv",
        REPO / "outputs/giti_screened_with_gates.csv",
    ])
    mapping = pd.read_csv(REV / "label_to_giti_mapping.csv")
    giti = pd.read_csv(REPO / "outputs/giti_screened_with_gates.csv")
    assert mapping.giti_idx.max() < len(giti)
    assert mapping.giti_idx.min() >= 0


def test_ev_jumpstats_has_114_rows():
    _skip_if_missing([REV / "ev_jumpstats.csv"])
    jump = pd.read_csv(REV / "ev_jumpstats.csv")
    assert len(jump) == 114


def test_fp_list_has_73_rows():
    _skip_if_missing([REV / "fp_list.csv"])
    fps = pd.read_csv(REV / "fp_list.csv")
    assert len(fps) == 73


def test_features_csv_covers_all_114_events():
    _skip_if_missing([REV / "features.csv"])
    feats = pd.read_csv(REV / "features.csv")
    assert len(feats) == 114
