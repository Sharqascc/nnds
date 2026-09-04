"""
Tests for gate counter logic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.gate_counter import TrafficVolumeCounter


def _make_dummy_video(tmp_path, name="dummy.mp4"):
    video = tmp_path / name
    video.write_bytes(b"fake video content")
    return video


def test_counter_initialization(tmp_path):
    """Test counter initializes correctly with dummy video file."""
    video = _make_dummy_video(tmp_path)
    counter = TrafficVolumeCounter(str(video))
    assert counter is not None
    assert len(counter.gates) == 0  # no gate config -> empty


def test_counter_gate_config_validation(tmp_path):
    """Test that invalid gate config does not raise, but loads no gates."""
    video = _make_dummy_video(tmp_path)
    counter = TrafficVolumeCounter(str(video), gate_config="invalid_path.yaml")
    assert counter.gates == {}


def test_counter_classes_of_interest(tmp_path):
    """Test that classes_of_interest is set correctly."""
    video = _make_dummy_video(tmp_path)
    counter = TrafficVolumeCounter(str(video), classes_of_interest=["car", "motorcycle"])
    assert counter.classes_of_interest == ["car", "motorcycle"]
