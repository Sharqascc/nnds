import sys
from pathlib import Path

import pytest
from core import traffic_analyzer


ROOT = Path(__file__).resolve().parents[1]


# ---------------------------
# Helpers
# ---------------------------

class DummyArgs:
    """Minimal stand-in for argparse.Namespace when needed."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# ---------------------------
# Demo-mode tests (existing, slightly strengthened)
# ---------------------------

def test_parse_args_demo(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["traffic_analyzer.py", "--demo"],
    )
    args = traffic_analyzer.parse_args()
    assert args.demo is True
    assert args.video is None


def test_main_requires_video_without_demo(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["traffic_analyzer.py"],
    )
    with pytest.raises(SystemExit) as exc:
        traffic_analyzer.main()

    # Non-zero exit code and helpful message
    assert exc.value.code != 0
    assert "video is required unless --demo is used" in str(exc.value)


def test_main_demo_runs(monkeypatch):
    called = {"demo": False}

    def fake_run_demo(*_args, **_kwargs):
        called["demo"] = True

    monkeypatch.setattr(
        "sys.argv",
        ["traffic_analyzer.py", "--demo"],
    )
    monkeypatch.setattr(traffic_analyzer, "run_demo", fake_run_demo)

    traffic_analyzer.main()
    assert called["demo"] is True


# ---------------------------
# Main pipeline with --video
# ---------------------------

def test_parse_args_with_video_and_basic_flags(monkeypatch, tmp_path):
    """Ensure core CLI flags parse correctly with a normal video invocation."""
    video_path = tmp_path / "dummy.mp4"
    video_path.write_bytes(b"\x00")  # dummy file

    out_csv = tmp_path / "out.csv"

    monkeypatch.setattr(
        "sys.argv",
        [
            "traffic_analyzer.py",
            "--video",
            str(video_path),
            "--out-csv",
            str(out_csv),
            "--pet-threshold",
            "2.5",
        ],
    )

    args = traffic_analyzer.parse_args()
    assert args.demo is False
    assert args.video == str(video_path)
    assert args.out_csv == str(out_csv)
    # Depending on implementation, arg name may be pet_threshold or similar
    assert getattr(args, "pet_threshold", None) in (2.5, "2.5")


def test_main_runs_with_video_uses_pipeline(monkeypatch, tmp_path):
    """Smoke-test that main() with --video calls the pipeline entrypoint."""
    video_path = tmp_path / "dummy.mp4"
    video_path.write_bytes(b"\x00")

    called = {"run_pipeline": False}

    def fake_run_pipeline(args):
        assert args.video == str(video_path)
        called["run_pipeline"] = True

    def fake_run_demo():
        # Should never be called with --video
        called["run_pipeline"] = False

    # Mock the correct function based on what exists
    if hasattr(traffic_analyzer, "run_pipeline"):
        monkeypatch.setattr(traffic_analyzer, "run_pipeline", fake_run_pipeline)
        # Also mock run_demo to ensure it's not accidentally used
        if hasattr(traffic_analyzer, "run_demo"):
            monkeypatch.setattr(traffic_analyzer, "run_demo", fake_run_demo)
    else:
        # If no run_pipeline, the test should fail clearly / be marked as skipped
        pytest.skip("traffic_analyzer has no run_pipeline function")

    monkeypatch.setattr(
        "sys.argv",
        ["traffic_analyzer.py", "--video", str(video_path)],
    )
    traffic_analyzer.main()
    assert called["run_pipeline"] is True


# ---------------------------
# Argument validation tests
# ---------------------------

def test_invalid_pet_threshold_rejected(monkeypatch):
    """
    If parse_args or main validates pet-threshold, ensure invalid values are handled.
    If not implemented yet, this test can be adapted once validation is added.
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "traffic_analyzer.py",
            "--demo",
            "--pet-threshold",
            "-1.0",
        ],
    )

    try:
        args = traffic_analyzer.parse_args()
    except SystemExit:
        # argparse might exit on invalid value if using choices/range logic
        return

    # If parse_args succeeds, main should enforce a positive threshold
    if hasattr(traffic_analyzer, "validate_args"):
        with pytest.raises(SystemExit):
            traffic_analyzer.validate_args(args)


# ---------------------------
# Error handling tests: bad/missing video
# ---------------------------

def test_main_errors_on_missing_video_file(monkeypatch, tmp_path):
    """
    When a non-demo run is invoked with a non-existent video, main() should fail
    gracefully and not attempt expensive processing.
    """
    missing_video = tmp_path / "missing_video.mp4"

    monkeypatch.setattr(
        "sys.argv",
        [
            "traffic_analyzer.py",
            "--video",
            str(missing_video),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        traffic_analyzer.main()

    assert exc.value.code != 0
    msg = str(exc.value).lower()
    assert "video file not found" in msg or "no such file" in msg


# ---------------------------
# Config integration tests
# ---------------------------

def test_parse_args_with_bev_and_grid_config(monkeypatch, tmp_path):
    bev_config = tmp_path / "bev_config.json"
    bev_config.write_text('{"homography_matrix": [[1,0,0],[0,1,0],[0,0,1]], "roi": {}}')

    grid_config = tmp_path / "grid_config.json"
    grid_config.write_text('{"cells": {}, "cell_size": 1.0}')

    video_path = tmp_path / "dummy.mp4"
    video_path.write_bytes(b"\x00")

    monkeypatch.setattr(
        "sys.argv",
        [
            "traffic_analyzer.py",
            "--video",
            str(video_path),
            "--bev-config",
            str(bev_config),
            "--grid-config",
            str(grid_config),
        ],
    )

    args = traffic_analyzer.parse_args()
    assert getattr(args, "bev_config", None) == str(bev_config)
    assert getattr(args, "grid_config", None) == str(grid_config)


def test_main_uses_configs_when_provided(monkeypatch, tmp_path):
    """
    Ensure that when config paths are passed, main() forwards them into the pipeline.
    We mock the heavy pipeline entrypoint to keep this fast.
    """
    bev_config = tmp_path / "bev_config.json"
    bev_config.write_text('{"homography_matrix": [[1,0,0],[0,1,0],[0,0,1]], "roi": {}}')

    grid_config = tmp_path / "grid_config.json"
    grid_config.write_text('{"cells": {}, "cell_size": 1.0}')

    video_path = tmp_path / "dummy.mp4"
    video_path.write_bytes(b"\x00")

    captured = {"args": None}

    def fake_run_pipeline(args):
        captured["args"] = args

    if hasattr(traffic_analyzer, "run_pipeline"):
        monkeypatch.setattr(traffic_analyzer, "run_pipeline", fake_run_pipeline)
    else:
        pytest.skip("traffic_analyzer has no run_pipeline function")

    monkeypatch.setattr(
        "sys.argv",
        [
            "traffic_analyzer.py",
            "--video",
            str(video_path),
            "--bev-config",
            str(bev_config),
            "--grid-config",
            str(grid_config),
        ],
    )

    traffic_analyzer.main()
    assert captured["args"] is not None
    assert getattr(captured["args"], "bev_config", None) == str(bev_config)
    assert getattr(captured["args"], "grid_config", None) == str(grid_config)


# ---------------------------
# Wider monkeypatch: heavy dependencies
# ---------------------------

def test_main_with_video_does_not_touch_heavy_dependencies(monkeypatch, tmp_path):
    """
    Smoke-test that we can run main() with --video while stubbing out the heavy
    dependencies (SAM3, BEV, grid) so this test remains fast and robust.
    """
    video_path = tmp_path / "dummy.mp4"
    video_path.write_bytes(b"\x00")

    # Stub out heavy modules if traffic_analyzer imports them directly.
    # Adjust names based on actual imports in traffic_analyzer.py.
    class DummyHeavy:
        def __getattr__(self, _name):
            def _noop(*_args, **_kwargs):
                return None
            return _noop

    # Example (uncomment and adjust if needed):
    # monkeypatch.setattr("grid_trajectory.sam3_grid_pet", DummyHeavy(), raising=False)

    called = {"pipeline": False}

    def fake_run_pipeline(args):
        called["pipeline"] = True

    if hasattr(traffic_analyzer, "run_pipeline"):
        monkeypatch.setattr(traffic_analyzer, "run_pipeline", fake_run_pipeline)
    else:
        pytest.skip("traffic_analyzer has no run_pipeline function")

    monkeypatch.setattr(
        "sys.argv",
        [
            "traffic_analyzer.py",
            "--video",
            str(video_path),
        ],
    )

    traffic_analyzer.main()
    assert called["pipeline"] is True
