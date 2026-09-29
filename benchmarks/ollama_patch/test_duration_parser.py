import pytest

from src.utils.duration_parser import parse_duration_seconds


def test_empty_string_raises():
    with pytest.raises(ValueError):
        parse_duration_seconds("")


def test_valid_simple_seconds():
    assert parse_duration_seconds("90s") == 90.0


def test_valid_minutes_and_seconds():
    assert parse_duration_seconds("2m30s") == 150.0


def test_valid_hours():
    assert parse_duration_seconds("1h") == 3600.0
