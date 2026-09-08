
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Make scripts importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import agentic_fix


def test_models_list_nonempty():
    assert isinstance(agentic_fix.MODELS, list)
    assert len(agentic_fix.MODELS) > 0
    assert all(isinstance(m, str) for m in agentic_fix.MODELS)


def test_run_cmd_returns_completed_process():
    res = agentic_fix.run_cmd(["git", "--version"])
    assert hasattr(res, "returncode")
    assert res.returncode == 0


def test_call_with_fallback_rate_limit_then_success(monkeypatch):
    """When the first model hits a rate limit, the function falls back to the next model."""
    # Simulate a Groq-compatible client where completions.create is fake
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

    fake_client = FakeClient()
    monkeypatch.setattr(agentic_fix.time, "sleep", lambda s: None)

    result = agentic_fix.call_with_fallback(
        fake_client,
        messages=[{"role": "user", "content": "hi"}],
        models=["bad-model", "good-model"],
        max_retries_per_model=2,
        base_wait=0,
    )
    assert result.choices[0].message.content == "ok"
