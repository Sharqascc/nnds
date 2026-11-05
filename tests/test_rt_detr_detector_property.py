"""Property-based tests for RTDetrDetector.

The RTDETR model is stubbed. Every property checks the conversion logic
between ultralytics Results objects and Detection dataclasses.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from src.pipeline.rt_detr_detector import Detection, RTDetrDetector


class _FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4)
        self.conf = torch.tensor(conf, dtype=torch.float32).reshape(-1)
        self.cls = torch.tensor(cls, dtype=torch.float32).reshape(-1)


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    def __init__(self, result):
        self._result = result

    def __call__(self, frame, conf=0.25):
        return [self._result]


def _detector_with(boxes):
    d = object.__new__(RTDetrDetector)
    d.model = _FakeModel(_FakeResult(boxes))
    return d


def _empty_frame():
    return np.zeros((100, 100, 3), dtype=np.uint8)


def test_none_boxes_returns_empty_list():
    d = _detector_with(None)
    assert d.detect(_empty_frame()) == []


@given(n=st.integers(0, 10))
def test_detection_count_matches_input(n):
    boxes = _FakeBoxes(
        xyxy=[[10.0, 10.0, 50.0, 50.0]] * n,
        conf=[0.5] * n,
        cls=[0] * n,
    )
    d = _detector_with(boxes)
    out = d.detect(_empty_frame())
    assert len(out) == n
    for det in out:
        assert isinstance(det, Detection)
        assert isinstance(det.cls, int)


@given(
    x1=st.floats(-1000.0, 1000.0, allow_nan=False, allow_infinity=False),
    y1=st.floats(-1000.0, 1000.0, allow_nan=False, allow_infinity=False),
    w=st.floats(1.0, 500.0, allow_nan=False, allow_infinity=False),
    h=st.floats(1.0, 500.0, allow_nan=False, allow_infinity=False),
)
def test_box_coordinates_round_trip(x1, y1, w, h):
    x2, y2 = x1 + w, y1 + h
    boxes = _FakeBoxes([[x1, y1, x2, y2]], [0.9], [3])
    d = _detector_with(boxes)
    out = d.detect(_empty_frame())
    assert len(out) == 1
    det = out[0]
    assert det.x1 == pytest.approx(x1, abs=1e-3)
    assert det.y1 == pytest.approx(y1, abs=1e-3)
    assert det.x2 == pytest.approx(x2, abs=1e-3)
    assert det.y2 == pytest.approx(y2, abs=1e-3)
    assert det.x1 < det.x2
    assert det.y1 < det.y2


@given(
    scores=st.lists(
        st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=8,
    )
)
def test_scores_are_preserved(scores):
    n = len(scores)
    boxes = _FakeBoxes(
        xyxy=[[1.0, 2.0, 3.0, 4.0]] * n,
        conf=scores,
        cls=[0] * n,
    )
    d = _detector_with(boxes)
    out = d.detect(_empty_frame())
    assert len(out) == n
    for expected, det in zip(scores, out, strict=True):
        assert det.score == pytest.approx(expected, abs=1e-4)


@given(
    classes=st.lists(st.integers(0, 10), min_size=0, max_size=8),
)
def test_classes_are_integers(classes):
    n = len(classes)
    boxes = _FakeBoxes(
        xyxy=[[1.0, 2.0, 3.0, 4.0]] * n,
        conf=[0.5] * n,
        cls=classes,
    )
    d = _detector_with(boxes)
    out = d.detect(_empty_frame())
    assert len(out) == n
    for expected, det in zip(classes, out, strict=True):
        assert det.cls == int(expected)
