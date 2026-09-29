import pytest

from src.analysis.gt_validation import (
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    has_errors,
    summarise,
    validate_detection_rows,
    validate_ssm_rows,
    validate_tracking_rows,
    validate_trajectory_rows,
)


def _good_detection():
    return [
        {"frame": 0, "x1": 10, "y1": 20, "x2": 50, "y2": 80, "class_name": "car"},
        {"frame": 1, "x1": 11, "y1": 21, "x2": 51, "y2": 81, "class_name": "car"},
    ]


def _good_tracking():
    return [
        {"frame": 0, "track_id": 1, "x": 10, "y": 20, "w": 40, "h": 60},
        {"frame": 1, "track_id": 1, "x": 11, "y": 21, "w": 40, "h": 60},
    ]


def _good_trajectory():
    return [
        {"frame": 0, "track_id": 1, "x": 0.0, "y": 0.0},
        {"frame": 1, "track_id": 1, "x": 1.0, "y": 0.5},
    ]


def _good_ssm():
    return [{"track_a": 1, "track_b": 2, "pet": 1.5, "ttc": 0.6}]


def test_good_detection_no_errors():
    assert not has_errors(validate_detection_rows(_good_detection()))


def test_good_tracking_no_errors():
    assert not has_errors(validate_tracking_rows(_good_tracking()))


def test_good_trajectory_no_errors():
    assert not has_errors(validate_trajectory_rows(_good_trajectory()))


def test_good_ssm_no_errors():
    assert not has_errors(validate_ssm_rows(_good_ssm()))


def test_empty_detection_is_warning_not_error():
    issues = validate_detection_rows([])
    assert issues and issues[0].severity == SEVERITY_WARNING
    assert not has_errors(issues)


def test_missing_columns_detection_error():
    issues = validate_detection_rows([{"frame": 0}])
    assert has_errors(issues)


def test_invalid_box_detection_error():
    bad = [{"frame": 0, "x1": 50, "y1": 50, "x2": 10, "y2": 10, "class_name": "car"}]
    assert has_errors(validate_detection_rows(bad))


def test_duplicate_detection_warning():
    rows = _good_detection()
    rows.append(rows[0])
    issues = validate_detection_rows(rows)
    assert any(i.kind == "duplicate" for i in issues)
    assert not has_errors(issues)


def test_unsorted_frames_warning():
    rows = list(reversed(_good_detection()))
    issues = validate_detection_rows(rows)
    assert any(i.kind == "unsorted_frames" for i in issues)


def test_negative_wh_tracking_error():
    bad = [{"frame": 0, "track_id": 1, "x": 0, "y": 0, "w": -1, "h": 10}]
    assert has_errors(validate_tracking_rows(bad))


def test_short_track_warning():
    rows = [{"frame": 0, "track_id": 1, "x": 0, "y": 0, "w": 10, "h": 10}]
    issues = validate_tracking_rows(rows)
    assert any(i.kind == "short_track" for i in issues)


def test_ssm_same_track_error():
    bad = [{"track_a": 1, "track_b": 1, "pet": 1.0}]
    assert has_errors(validate_ssm_rows(bad))


def test_ssm_non_positive_pet_error():
    bad = [{"track_a": 1, "track_b": 2, "pet": -0.5}]
    assert has_errors(validate_ssm_rows(bad))


def test_ssm_non_finite_pet_error():
    bad = [{"track_a": 1, "track_b": 2, "pet": float("nan")}]
    assert has_errors(validate_ssm_rows(bad))


def test_ssm_unknown_track_warning():
    rows = [{"track_a": 99, "track_b": 2, "pet": 1.0}]
    issues = validate_ssm_rows(rows, known_track_ids={1, 2})
    assert any(i.kind == "unknown_track" for i in issues)
    assert not has_errors(issues)


def test_summary_counts():
    issues = validate_detection_rows([])
    s = summarise(issues)
    assert s[SEVERITY_WARNING] >= 1
    assert s[SEVERITY_ERROR] == 0


def test_large_gap_warning():
    rows = [
        {"frame": 0, "x1": 10, "y1": 20, "x2": 50, "y2": 80, "class_name": "car"},
        {"frame": 100, "x1": 10, "y1": 20, "x2": 50, "y2": 80, "class_name": "car"},
    ]
    issues = validate_detection_rows(rows)
    assert any(i.kind == "large_gap" for i in issues)
