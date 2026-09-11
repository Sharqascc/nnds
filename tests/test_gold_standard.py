import json

import pytest

from src.analysis.gold_standard import (
    ALL_TIERS,
    N_ROWS,
    TIER_DETECTION,
    TIER_SAFETY,
    TIER_TRACKING,
    TIER_TRAJECTORY,
    GoldStandardRow,
    build_table,
    to_dicts,
    to_markdown,
)


def test_row_count_is_fifteen():
    assert N_ROWS == 15
    assert len(build_table()) == 15


def test_empty_inputs_all_zero():
    rows = build_table()
    assert all(r.value == 0.0 for r in rows)


def test_tier_coverage_complete():
    rows = build_table()
    assert {r.tier for r in rows} == ALL_TIERS


def test_metric_names_unique():
    rows = build_table()
    assert len({r.metric for r in rows}) == 15


def test_values_transferred_by_family():
    detection = {
        "precision": 0.9,
        "recall": 0.8,
        "f1": 0.85,
        "map50": 0.9,
        "map50_95": 0.7,
        "ap75": 0.75,
    }
    tracking = {"hota": 0.8, "idf1": 0.85, "mota": 0.9}
    trajectory = {"position_rmse_m": 0.2, "velocity_mae_mps": 0.4, "accel_mae_mps2": 0.6}
    ssm = {"pet_mae_s": 0.15, "ttc_mae_s": 0.2, "critical_conflict_recall": 0.95}
    rows = {r.metric: r.value for r in build_table(detection, tracking, trajectory, ssm)}
    assert rows["precision"] == pytest.approx(0.9)
    assert rows["mAP50:95"] == pytest.approx(0.7)
    assert rows["HOTA"] == pytest.approx(0.8)
    assert rows["position_RMSE"] == pytest.approx(0.2)
    assert rows["PET_MAE"] == pytest.approx(0.15)
    assert rows["critical_conflict_recall"] == pytest.approx(0.95)


def test_nan_and_inf_become_zero():
    rows = {r.metric: r.value for r in build_table(detection={"precision": float("nan")})}
    assert rows["precision"] == 0.0
    rows = {r.metric: r.value for r in build_table(tracking={"hota": float("inf")})}
    assert rows["HOTA"] == 0.0


def test_string_value_becomes_zero():
    rows = {r.metric: r.value for r in build_table(ssm={"pet_mae_s": "not-a-number"})}
    assert rows["PET_MAE"] == 0.0


def test_missing_key_defaults_zero():
    rows = {r.metric: r.value for r in build_table(detection={"precision": 0.5})}
    assert rows["precision"] == pytest.approx(0.5)
    assert rows["recall"] == 0.0
    assert rows["f1"] == 0.0


def test_dicts_are_json_serialisable():
    d = to_dicts(build_table())
    serialised = json.dumps(d)
    assert isinstance(serialised, str)
    assert len(d) == 15


def test_markdown_has_header_and_fifteen_rows():
    md = to_markdown(build_table())
    lines = md.strip().splitlines()
    assert lines[0].startswith("| Metric")
    assert len(lines) == 2 + 15


def test_dataclass_frozen():
    row = GoldStandardRow("x", 1.0, "ratio", TIER_DETECTION)
    with pytest.raises(Exception):
        row.metric = "y"  # type: ignore[misc]


def test_all_tiers_present_as_strings():
    for t in ALL_TIERS:
        assert isinstance(t, str)
    assert TIER_DETECTION in ALL_TIERS
    assert TIER_TRACKING in ALL_TIERS
    assert TIER_TRAJECTORY in ALL_TIERS
    assert TIER_SAFETY in ALL_TIERS
