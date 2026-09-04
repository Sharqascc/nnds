
import pytest
from pydantic import ValidationError

from src.core.contracts import PETEventRecord


def test_pet_event_record_valid():
    row = {
        "event_id": 1,
        "pet": 0.5,
        "frame": 100,
        "track_a": 1,
        "track_b": 2,
        "conflict_type": "crossing",
        "grid_cell": "G_A_1",
        "track_a_entry_frame": 90,
        "track_a_exit_frame": 95,
        "track_a_exit_time_sec": 3.0,
        "track_b_entry_frame": 110,
        "track_b_entry_time_sec": 3.667,
        "track_b_exit_frame": 115,
        "site": "GITI",
        "time_of_day_label": "morning",
        "gate_a_entry": "G1",
        "gate_b_entry": "G2",
        "first_track_id": 1,
        "second_track_id": 2,
        "first_exit_frame": 95,
        "first_exit_time_sec": 3.0,
        "second_entry_frame": 110,
        "second_entry_time_sec": 3.667,
        "pet_frames": 15,
        "pet_s": 0.5,
        "occupancy_relation": "b_to_a",
    }
    record = PETEventRecord(**row)
    assert record.pet == 0.5


def test_pet_event_record_negative_pet():
    with pytest.raises(ValidationError):
        PETEventRecord(**{
            "event_id": 1, "pet": -0.1, "frame": 10, "track_a": 1, "track_b": 2,
            "conflict_type": "crossing", "grid_cell": "G_A_1", "site": "GITI"
        })


def test_pet_event_record_same_tracks():
    with pytest.raises(ValidationError):
        PETEventRecord(**{
            "event_id": 1, "pet": 0.5, "frame": 10, "track_a": 1, "track_b": 1,
            "conflict_type": "crossing", "grid_cell": "G_A_1", "site": "GITI"
        })


def test_pet_event_record_invalid_conflict_type():
    with pytest.raises(ValidationError):
        PETEventRecord(**{
            "event_id": 1, "pet": 0.5, "frame": 10, "track_a": 1, "track_b": 2,
            "conflict_type": "invalid", "grid_cell": "G_A_1", "site": "GITI"
        })
