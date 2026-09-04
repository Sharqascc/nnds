"""
Tests for traffic_analyzer main entry point functions.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.traffic_analyzer import (
    CompleteTrafficAnalyzer,
    WorldPoint,
    interactive_detector,
    parse_args,
    run_demo,
)

# ============================================
# 1. parse_args Tests
# ============================================


def test_parse_args_demo():
    """Test parse_args with --demo flag."""
    sys.argv = ["traffic_analyzer.py", "--demo"]
    args = parse_args()
    assert args is not None
    assert args.demo is True


def test_parse_args_video():
    """Test parse_args with video path."""
    sys.argv = ["traffic_analyzer.py", "--video", "test.mp4"]
    args = parse_args()
    assert args is not None
    assert args.video == "test.mp4"


def test_parse_args_detector():
    """Test parse_args with detector selection."""
    sys.argv = ["traffic_analyzer.py", "--detector", "yolo-cpu"]
    args = parse_args()
    assert args is not None
    assert args.detector == "yolo-cpu"


def test_parse_args_default():
    """Test parse_args with defaults."""
    sys.argv = ["traffic_analyzer.py"]
    args = parse_args()
    assert args is not None
    assert args.detector == "uvh-coco-fused"
    assert args.pet_threshold == 2.0
    assert args.max_gap == 5


# ============================================
# 2. CompleteTrafficAnalyzer Tests
# ============================================


def test_analyzer_initialization():
    """Test CompleteTrafficAnalyzer initializes."""
    analyzer = CompleteTrafficAnalyzer(bev_width=100, bev_height=100)
    assert analyzer is not None
    assert analyzer.bev_width == 100
    assert analyzer.bev_height == 100


def test_analyzer_calibration_required():
    """Test that calibration must be called before mapping."""
    analyzer = CompleteTrafficAnalyzer()
    with pytest.raises(RuntimeError):
        analyzer.pixel_to_world([100, 100])


def test_analyzer_pixel_to_world():
    """Test pixel_to_world after calibration."""
    analyzer = CompleteTrafficAnalyzer()
    analyzer.homography = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    world = analyzer.pixel_to_world([100, 100])
    assert world is not None
    assert abs(world[0] - 100) < 0.01
    assert abs(world[1] - 100) < 0.01


def test_world_point_dataclass():
    """Test WorldPoint dataclass."""
    point = WorldPoint(t=1.5, x=10.0, y=20.0)
    assert point.t == 1.5
    assert point.x == 10.0
    assert point.y == 20.0


def test_analyzer_estimate_speed_requires_calibration():
    """Test speed estimation requires calibration."""
    analyzer = CompleteTrafficAnalyzer()
    with pytest.raises(RuntimeError):
        analyzer.estimate_speed(np.array([[0, 0], [1, 1]]), np.array([0, 1 / 30]))


def test_analyzer_estimate_speed_basic():
    """Test speed estimation with homography set."""
    analyzer = CompleteTrafficAnalyzer()
    analyzer.homography = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1.0]])
    pixel_positions = np.array([[i * 1.67, 100.0] for i in range(20)], dtype=np.float32)
    frame_times = np.arange(20) / 30.0
    result = analyzer.estimate_speed(pixel_positions, frame_times, fps=30.0)
    assert result is not None
    assert "final_speed" in result
    assert result["final_speed"] > 0


def test_analyzer_generate_report():
    """Test report generation."""
    analyzer = CompleteTrafficAnalyzer()
    speed_results = {"final_speed": 15.0, "speed_std": 2.0}
    report = analyzer.generate_report(speed_results)
    assert report is not None
    assert isinstance(report, dict)


def test_analyzer_save_calibration():
    """Test saving calibration."""
    analyzer = CompleteTrafficAnalyzer()
    analyzer.homography = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    tmp_dir = Path(tempfile.mkdtemp())
    save_path = tmp_dir / "calibration.json"
    analyzer.save_calibration(str(save_path))
    assert save_path.exists()


def test_analyzer_validate_bev_requires_calibration():
    """Test validate_bev requires calibration."""
    analyzer = CompleteTrafficAnalyzer()
    with pytest.raises(RuntimeError):
        analyzer.validate_bev()


# ============================================
# 3. run_demo Test
# ============================================


def test_run_demo():
    """Test run_demo executes."""
    result = run_demo()
    assert result is not None
    assert len(result) == 3


# ============================================
# 4. interactive_detector Test
# ============================================


def test_interactive_detector():
    """Test interactive_detector function exists."""
    assert callable(interactive_detector)
