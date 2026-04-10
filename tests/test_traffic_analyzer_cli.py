import pytest
import traffic_analyzer


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
    assert "video is required unless --demo is used" in str(exc.value)


def test_main_demo_runs(monkeypatch):
    called = {"demo": False}

    def fake_run_demo():
        called["demo"] = True

    monkeypatch.setattr(
        "sys.argv",
        ["traffic_analyzer.py", "--demo"],
    )
    monkeypatch.setattr(traffic_analyzer, "run_demo", fake_run_demo)

    traffic_analyzer.main()
    assert called["demo"] is True
