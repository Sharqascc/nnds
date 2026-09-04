
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.pipeline.traffic_analyzer import CompleteTrafficAnalyzer


def identity_analyzer():
    analyzer = CompleteTrafficAnalyzer()
    analyzer.homography = np.eye(3, dtype=np.float32)
    return analyzer

@given(st.lists(st.tuples(st.floats(0, 100), st.floats(0, 100)), min_size=5, max_size=20),
       st.floats(min_value=0.1, max_value=10))
def test_estimate_speed_non_negative(pixel_pts, dt):
    analyzer = identity_analyzer()
    pixel_positions = np.array(pixel_pts, dtype=np.float32)
    frame_times = np.arange(len(pixel_positions)) * dt
    result = analyzer.estimate_speed(pixel_positions, frame_times, fps=30.0)
    assert result["final_speed"] >= 0
    assert result["speed_std"] >= 0

@given(st.lists(st.tuples(st.floats(0, 100), st.floats(0, 100)), min_size=5, max_size=20),
       st.floats(min_value=0.1, max_value=10))
def test_estimate_speed_finite(pixel_pts, dt):
    analyzer = identity_analyzer()
    pixel_positions = np.array(pixel_pts, dtype=np.float32)
    frame_times = np.arange(len(pixel_positions)) * dt
    result = analyzer.estimate_speed(pixel_positions, frame_times, fps=30.0)
    assert np.isfinite(result["final_speed"])
    assert np.isfinite(result["speed_std"])
