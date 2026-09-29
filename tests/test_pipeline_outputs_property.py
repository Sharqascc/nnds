import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis.pipeline_outputs import (
    detections_from_pipeline_rows,
    ssm_events_from_pet_rows,
    tracks_from_pipeline_rows,
    trajectories_from_pipeline_rows,
)

IDENTITY = np.eye(3)

det_row_st = st.fixed_dictionaries(
    {
        "frame": st.integers(0, 100),
        "track_id": st.integers(1, 50),
        "class_id": st.integers(0, 10),
        "class_name": st.sampled_from(["car", "pedestrian", "bike"]),
        "conf": st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        "x1": st.floats(0, 500, allow_nan=False, allow_infinity=False),
        "y1": st.floats(0, 500, allow_nan=False, allow_infinity=False),
        "x2": st.floats(0, 500, allow_nan=False, allow_infinity=False),
        "y2": st.floats(0, 500, allow_nan=False, allow_infinity=False),
        "cx": st.floats(0, 500, allow_nan=False, allow_infinity=False),
        "cy": st.floats(0, 500, allow_nan=False, allow_infinity=False),
        "source": st.just("uvh26"),
    }
)

pet_row_st = st.fixed_dictionaries(
    {
        "event_id": st.integers(0, 1000),
        "site": st.sampled_from(["GITI", "MRC"]),
        "pet": st.floats(0.01, 10.0, allow_nan=False, allow_infinity=False),
        "conflict_type": st.sampled_from(["crossing", "rear_end"]),
        "grid_cell": st.just("A1"),
        "orig_track_a": st.integers(1, 20),
        "orig_track_b": st.integers(1, 20),
    }
)


@given(st.lists(det_row_st, min_size=0, max_size=20))
@settings(max_examples=50)
def test_detection_round_trip(rows):
    dets = detections_from_pipeline_rows(rows)
    assert len(dets) == len(rows)


@given(st.lists(det_row_st, min_size=0, max_size=20))
@settings(max_examples=50)
def test_tracks_round_trip(rows):
    trk = tracks_from_pipeline_rows(rows)
    assert len(trk) == len(rows)


@given(st.lists(det_row_st, min_size=0, max_size=20))
@settings(max_examples=50)
def test_trajectories_round_trip(rows):
    pts = trajectories_from_pipeline_rows(rows, IDENTITY)
    assert len(pts) == len(rows)


@given(st.lists(pet_row_st, min_size=0, max_size=20))
@settings(max_examples=50)
def test_ssm_round_trip(rows):
    evs = ssm_events_from_pet_rows(rows)
    assert len(evs) == len(rows)


@given(det_row_st, st.integers(1, 5))
@settings(max_examples=50)
def test_scale_homography_multiplies_center(row, scale):
    H = np.array([[float(scale), 0.0, 0.0], [0.0, float(scale), 0.0], [0.0, 0.0, 1.0]])
    pts = trajectories_from_pipeline_rows([row], H)
    assert pts[0].x == pytest.approx(row["cx"] * scale)
    assert pts[0].y == pytest.approx(row["cy"] * scale)
