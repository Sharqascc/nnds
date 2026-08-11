from pathlib import Path
import importlib.util


def load_module():
    path = Path("/content/nnds/traffic_analyzer_demo.py")
    spec = importlib.util.spec_from_file_location("traffic_analyzer_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_module_imports():
    mod = load_module()
    assert hasattr(mod, "CompleteTrafficAnalyzer")
    assert hasattr(mod, "WorldPoint")


def test_calibrate_with_minimal_points():
    mod = load_module()
    analyzer = mod.CompleteTrafficAnalyzer()
    pixel_points = [(0, 0), (10, 0), (10, 10), (0, 10)]
    world_points = [(0, 0, 0), (5, 0, 0), (5, 5, 0), (0, 5, 0)]
    H, mask = analyzer.calibrate(pixel_points, world_points)
    assert H.shape == (3, 3)
    assert mask is not None
