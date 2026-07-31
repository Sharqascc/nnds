from pathlib import Path
import importlib.util
import pytest


def load_module():
    path = Path("/content/nnds/analysis/research_run.py")
    spec = importlib.util.spec_from_file_location("research_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_module_exports():
    mod = load_module()
    assert hasattr(mod, "run_cmd")
    assert hasattr(mod, "main")
    assert callable(mod.run_cmd)
    assert callable(mod.main)


def test_main_requires_video():
    mod = load_module()
    with pytest.raises(SystemExit):
        mod.main([])


def test_run_cmd_executes(monkeypatch):
    mod = load_module()
    calls = []

    class DummyResult:
        returncode = 0

    def fake_run(cmd, cwd=None):
        calls.append((cmd, cwd))
        return DummyResult()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    mod.run_cmd(["echo", "hello"], cwd=Path("/content/nnds"))
    assert calls and calls[0][0] == ["echo", "hello"]
