"""
Tests for gate counter logic.
"""
import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.gate_counter import TrafficVolumeCounter

def test_counter_initialization():
    """Test counter initializes correctly (requires video file)."""
    # This test is skipped if no video exists
    import os
    repo = Path(__file__).parent.parent
    video = repo/'data/sample_data/traffic_video.mp4'
    if not video.exists():
        pytest.skip("Video not available")
    
    counter = TrafficVolumeCounter(str(video))
    assert counter is not None
    assert len(counter.gates) > 0

def test_counter_gate_config_validation():
    """Test that invalid gate config raises error."""
    import os
    repo = Path(__file__).parent.parent
    video = repo/'data/sample_data/traffic_video.mp4'
    if not video.exists():
        pytest.skip("Video not available")
    
    with pytest.raises(Exception):
        TrafficVolumeCounter(str(video), gate_config="invalid_path.yaml")

def test_counter_classes_of_interest():
    """Test that classes_of_interest is set correctly."""
    import os
    repo = Path(__file__).parent.parent
    video = repo/'data/sample_data/traffic_video.mp4'
    if not video.exists():
        pytest.skip("Video not available")
    
    counter = TrafficVolumeCounter(str(video), classes_of_interest=["car", "motorcycle"])
    assert counter.classes_of_interest == ["car", "motorcycle"]
