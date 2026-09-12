import numpy as np
import pytest

from src.analysis.pipeline_outputs import (
    TrajPoint,
    detections_from_pipeline_rows,
    ssm_events_from_pet_rows,
    tracks_from_pipeline_rows,
    traj_points_to_rows,
    trajectories_from_pipeline_rows,
)

IDENTITY = np.eye(3)


def _det_row(**over):
    base = {
        "frame": 0,
        "track_id": 1,
        "class_id": 3,
        "class_name": "car",
        "conf": 0.9,
        "x1": 10,
        "y1": 20,
        "x2": 50,
        "y2": 80,
        "cx": 30,
        "cy": 50,
        "source": "uvh26",
    }
    base.update(over)
    return base


def _pet_row(**over):
    base = {
        "event_id": 0,
        "site": "GITI",
        "pet": 0.5,
        "conflict_type": "crossing",
        "grid_cell": "A1",
        "orig_track_a": 1,
        "orig_track_b": 2,
    }
    base.update(over)
    return base


def test_detections_happy_path():
    dets = detections_from_pipeline_rows([_det_row()])
    assert len(dets) == 1
    d = dets[0]
    assert d.frame == 0
    assert d.cls == "car"
    assert d.conf == pytest.approx(0.9)
    assert d.box == (10.0, 20.0, 50.0, 80.0)


def test_detections_empty():
    assert detections_from_pipeline_rows([]) == []


def test_detections_missing_column_raises():
    bad = _det_row()
    del bad["conf"]
    with pytest.raises(ValueError):
        detections_from_pipeline_rows([bad])


def test_tracks_happy_path():
    trk = tracks_from_pipeline_rows([_det_row()])
    assert len(trk) == 1
    t = trk[0]
    assert t.frame == 0
    assert t.track_id == 1
    assert t.box == (10.0, 20.0, 50.0, 80.0)


def test_tracks_empty():
    assert tracks_from_pipeline_rows([]) == []


def test_trajectories_identity_homography():
    pts = trajectories_from_pipeline_rows([_det_row()], IDENTITY)
    assert len(pts) == 1
    assert pts[0].x == pytest.approx(30.0)
    assert pts[0].y == pytest.approx(50.0)


def test_trajectories_scale_homography():
    H = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    pts = trajectories_from_pipeline_rows([_det_row()], H)
    assert pts[0].x == pytest.approx(60.0)
    assert pts[0].y == pytest.approx(100.0)


def test_trajectories_bad_H_shape_raises():
    with pytest.raises(ValueError):
        trajectories_from_pipeline_rows([_det_row()], np.eye(2))


def test_trajectories_point_at_infinity_raises():
    H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        trajectories_from_pipeline_rows([_det_row()], H)


def test_ssm_happy_path():
    evs = ssm_events_from_pet_rows([_pet_row()])
    assert len(evs) == 1
    e = evs[0]
    assert e.track_a == 1
    assert e.track_b == 2
    assert e.pet == pytest.approx(0.5)
    assert e.ttc is None


def test_ssm_empty():
    assert ssm_events_from_pet_rows([]) == []


def test_ssm_nan_pet_becomes_none():
    evs = ssm_events_from_pet_rows([_pet_row(pet=float("nan"))])
    assert evs[0].pet is None


def test_ssm_missing_column_raises():
    bad = _pet_row()
    del bad["orig_track_a"]
    with pytest.raises(ValueError):
        ssm_events_from_pet_rows([bad])


def test_traj_points_to_rows_round_trip():
    pts = [TrajPoint(frame=0, track_id=1, x=1.5, y=2.5)]
    rows = traj_points_to_rows(pts)
    assert rows == [{"frame": 0, "track_id": 1, "x": 1.5, "y": 2.5}]


def test_ssm_alternate_track_pair_schema():
    """Real pipeline PET CSV uses track_a/track_b, not orig_track_a/b."""
    row = {
        "event_id": 0,
        "site": "GITI",
        "pet": 0.7,
        "conflict_type": "other",
        "grid_cell": "A1",
        "track_a": 44000,
        "track_b": 2000,
    }
    evs = ssm_events_from_pet_rows([row])
    assert len(evs) == 1
    assert evs[0].track_a == 44000
    assert evs[0].track_b == 2000
    assert evs[0].pet == pytest.approx(0.7)


def test_ssm_no_track_pair_raises():
    with pytest.raises(ValueError):
        ssm_events_from_pet_rows([{"pet": 0.7}])


def test_ssm_both_schemas_same_result():
    row_a = {"pet": 1.2, "orig_track_a": 1, "orig_track_b": 2}
    row_b = {"pet": 1.2, "track_a": 1, "track_b": 2}
    ev_a = ssm_events_from_pet_rows([row_a])[0]
    ev_b = ssm_events_from_pet_rows([row_b])[0]
    assert ev_a == ev_b
