
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import given
from hypothesis import strategies as st

# Make scripts importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import agentic_fix


# ---------- Basic constants ----------
def test_models_list_nonempty():
    assert isinstance(agentic_fix.MODELS, list)
    assert len(agentic_fix.MODELS) > 0
    assert all(isinstance(m, str) for m in agentic_fix.MODELS)


def test_max_file_limits_positive():
    assert agentic_fix.MAX_FILE_LINES > 0
    assert agentic_fix.MAX_FILE_CHARS > 0


# ---------- run_cmd ----------
def test_run_cmd_returns_completed_process():
    res = agentic_fix.run_cmd(["git", "--version"])
    assert hasattr(res, "returncode")
    assert res.returncode == 0


# ---------- call_with_fallback ----------
def test_call_with_fallback_success_first_model(monkeypatch):
    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self):
            self.chat = FakeChat()

    client = FakeClient()
    monkeypatch.setattr(agentic_fix.time, "sleep", lambda s: None)
    result = agentic_fix.call_with_fallback(
        client,
        messages=[{"role": "user", "content": "hi"}],
        models=["good-model"],
        max_retries_per_model=1,
        base_wait=0,
    )
    assert result.choices[0].message.content == "ok"


def test_call_with_fallback_rate_limit_then_success(monkeypatch):
    class FakeCompletions:
        def create(self, **kwargs):
            model = kwargs.get("model")
            if model == "bad-model":
                raise Exception("rate_limit")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self):
            self.chat = FakeChat()

    client = FakeClient()
    monkeypatch.setattr(agentic_fix.time, "sleep", lambda s: None)
    result = agentic_fix.call_with_fallback(
        client,
        messages=[{"role": "user", "content": "hi"}],
        models=["bad-model", "good-model"],
        max_retries_per_model=1,
        base_wait=0,
    )
    assert result.choices[0].message.content == "ok"


def test_call_with_fallback_all_models_fail_raises(monkeypatch):
    class FakeCompletions:
        def create(self, **kwargs):
            raise Exception("rate_limit")

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self):
            self.chat = FakeChat()

    client = FakeClient()
    monkeypatch.setattr(agentic_fix.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        agentic_fix.call_with_fallback(
            client,
            messages=[{"role": "user", "content": "hi"}],
            models=["bad-model", "bad-model-2"],
            max_retries_per_model=1,
            base_wait=0,
        )

def test_run_cmd_timeout_returns_completed_process(monkeypatch):
    # Monkeypatch subprocess.run to simulate timeout
    import subprocess as sp
    def fake_run(cmd, **kwargs):
        raise sp.TimeoutExpired(cmd, kwargs.get('timeout', 120))
    monkeypatch.setattr(agentic_fix.subprocess, 'run', fake_run)
    res = agentic_fix.run_cmd(["fake"], timeout=1)
    assert res.returncode == 124



# ---------- Property-based test ----------
@given(st.lists(st.text(min_size=1, max_size=20), min_size=2, max_size=5))
def test_call_with_fallback_always_returns_success_when_one_model_works(model_names):
    """If the first model rate-limits, fallback to the second model succeeds."""
    class FakeCompletions:
        def __init__(self):
            self.calls = 0
        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise Exception("rate_limit")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self):
            self.chat = FakeChat()

    client = FakeClient()
    result = agentic_fix.call_with_fallback(
        client,
        messages=[{"role": "user", "content": "hi"}],
        models=model_names,
        max_retries_per_model=1,
        base_wait=0,
    )
    assert result.choices[0].message.content == "ok"

def test_extract_patch_strips_markdown():
    raw = """```diff
--- a/foo.py
+++ b/foo.py
@@ -1 +1 @@
-old
+new
```"""
    result = agentic_fix.extract_patch(raw)
    assert result.startswith('--- a/foo.py')
    assert '+new' in result
    assert '```' not in result

