import sys
from pathlib import Path

import numpy as np
import pytest

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.traffic_analyzer import CompleteTrafficAnalyzer


def create_test_analyzer():
    """Create a CompleteTrafficAnalyzer with a simple scaling homography."""
    analyzer = CompleteTrafficAnalyzer(bev_width=100, bev_height=100)
    # Homography: pixel (x,y) -> world (x*0.1, y*0.1) meters
    H = np.array([
        [0.1, 0, 0],
        [0, 0.1, 0],
        [0, 0, 1]
    ])
    analyzer.homography = H
    analyzer.inv_homography = np.linalg.inv(H)
    return analyzer


def test_speed_constant():
    """Test that constant speed is estimated correctly (10 m/s -> 36 km/h)."""
    analyzer = create_test_analyzer()
    fps = 30.0
    dt = 1.0 / fps
    # World speed = 10 m/s -> pixel speed = 100 px/s -> 3.333 px/frame
    pixel_positions = []
    frame_times = []
    for i in range(20):
        x = i * (10.0 / fps / 0.1)  # pixel x
        y = 100.0
        pixel_positions.append([x, y])
        frame_times.append(i * dt)
    
    pixel_positions = np.array(pixel_positions, dtype=np.float32)
    frame_times = np.array(frame_times, dtype=np.float32)
    
    result = analyzer.estimate_speed(pixel_positions, frame_times, fps=fps)
    
    # Expected ~36 km/h (10 m/s * 3.6)
    assert 35.0 < result['final_speed'] < 37.0, \
        f"Expected ~36 km/h, got {result['final_speed']}"


def test_speed_insufficient_points():
    """Test fallback when fewer than 5 valid points."""
    analyzer = create_test_analyzer()
    pixel_positions = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)
    frame_times = np.array([0, 1/30, 2/30], dtype=np.float32)
    result = analyzer.estimate_speed(pixel_positions, frame_times)
    assert result['final_speed'] == 15.0
    assert result['speed_std'] == 2.0


def test_speed_invalid_length():
    """Test mismatched input lengths raise ValueError."""
    analyzer = create_test_analyzer()
    pixel_positions = np.array([[0, 0], [1, 1]], dtype=np.float32)
    frame_times = np.array([0], dtype=np.float32)
    with pytest.raises(ValueError):
        analyzer.estimate_speed(pixel_positions, frame_times)


def test_speed_no_homography():
    """Test error when homography not set."""
    analyzer = CompleteTrafficAnalyzer()
    pixel_positions = np.array([[0,0], [1,1], [2,2], [3,3], [4,4]], dtype=np.float32)
    frame_times = np.array([0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    with pytest.raises(RuntimeError):
        analyzer.estimate_speed(pixel_positions, frame_times)


def test_speed_non_finite_filter():
    """Test that non-finite pixel positions are filtered."""
    analyzer = create_test_analyzer()
    fps = 30.0
    dt = 1.0 / fps
    pixel_positions = []
    frame_times = []
    for i in range(10):
        if i == 5:
            pixel_positions.append([np.nan, np.nan])  # invalid
        else:
            pixel_positions.append([i * 10.0, 100.0])
        frame_times.append(i * dt)
    
    pixel_positions = np.array(pixel_positions, dtype=np.float32)
    frame_times = np.array(frame_times, dtype=np.float32)
    
    result = analyzer.estimate_speed(pixel_positions, frame_times, fps=fps)
    assert result['final_speed'] > 0


def test_speed_acceleration():
    """Test that a moving trajectory with varying speed gives reasonable results."""
    analyzer = create_test_analyzer()
    fps = 30.0
    dt = 1.0 / fps
    pixel_positions = []
    frame_times = []
    # Accelerating: speed goes from 5 m/s to 15 m/s
    for i in range(20):
        speed_mps = 5.0 + (i * 0.5)  # 5 + 0.5*i m/s
        x = i * (speed_mps / fps / 0.1)
        y = 100.0
        pixel_positions.append([x, y])
        frame_times.append(i * dt)
    
    pixel_positions = np.array(pixel_positions, dtype=np.float32)
    frame_times = np.array(frame_times, dtype=np.float32)
    
    result = analyzer.estimate_speed(pixel_positions, frame_times, fps=fps)
    
    # Median speed should be between 5 and 15 m/s -> 18-54 km/h
    assert 15 < result['final_speed'] < 60, \
        f"Expected speed between 18-54 km/h, got {result['final_speed']}"
