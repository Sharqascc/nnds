
import re

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.analysis.visualization.pet_event_plots import (
    COLORS,
    DEFAULT_THRESHOLDS,
    EventPlotter,
)


def _is_valid_hex_color(s: str) -> bool:
    return bool(re.fullmatch(r"#[0-9A-Fa-f]{6}", s))


@given(st.sampled_from(list(COLORS.keys())))
def test_color_palette_valid_hex(name):
    assert _is_valid_hex_color(COLORS[name])


def test_default_thresholds_ordered():
    assert 0 < DEFAULT_THRESHOLDS["critical"] < DEFAULT_THRESHOLDS["serious"] < DEFAULT_THRESHOLDS["moderate"] < DEFAULT_THRESHOLDS["safe"]


@given(st.floats(min_value=0.0, max_value=6.0, allow_nan=False, allow_infinity=False))
def test_severity_color_known_ranges(pet_value):
    plotter = EventPlotter()
    color = plotter._get_severity_color(pet_value)
    thresholds = plotter.thresholds
    if pet_value < thresholds["critical"]:
        assert color == COLORS["red"]
    elif pet_value < thresholds["serious"]:
        assert color == COLORS["orange"]
    elif pet_value < thresholds["moderate"]:
        assert color == COLORS["yellow"]
    elif pet_value < thresholds["safe"]:
        assert color == COLORS["green"]
    else:
        assert color == COLORS["blue"]


@given(st.floats(min_value=0.0, max_value=6.0, allow_nan=False, allow_infinity=False))
def test_severity_label_known_ranges(pet_value):
    plotter = EventPlotter()
    label = plotter._get_severity_label(pet_value)
    thresholds = plotter.thresholds
    if pet_value < thresholds["critical"]:
        assert label == "Critical"
    elif pet_value < thresholds["serious"]:
        assert label == "Serious"
    elif pet_value < thresholds["moderate"]:
        assert label == "Moderate"
    elif pet_value < thresholds["safe"]:
        assert label == "Slight"
    else:
        assert label == "Safe"
