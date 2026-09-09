import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import given
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import agentic_fix_ollama


# ---------- Basic constants ----------
def test_models_list_nonempty():
    assert isinstance(agentic_fix_ollama.OLLAMA_MODELS, list)
    assert len(agentic_fix_ollama.OLLAMA_MODELS) > 0


def test_max_file_limits_positive():
    assert agentic_fix_ollama.MAX_FILE_LINES > 0
    assert agentic_fix_ollama.MAX_FILE_CHARS > 0


# ---------- call_ollama with mocked requests ----------
def test_call_ollama_success(monkeypatch):
    def fake_post(url, json, timeout):
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"message": {"content": "hello"}},
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(agentic_fix_ollama.requests, "post", fake_post)
    result = agentic_fix_ollama.call_ollama([{"role": "user", "content": "hi"}])
    assert result == "hello"


def test_call_ollama_fallback_on_connection_error(monkeypatch):
    call_count = {"n": 0}

    def fake_post(url, json, timeout):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise agentic_fix_ollama.requests.exceptions.ConnectionError("conn refused")
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"message": {"content": "ok"}},
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(agentic_fix_ollama.requests, "post", fake_post)
    monkeypatch.setattr(agentic_fix_ollama.time, "sleep", lambda s: None)
    result = agentic_fix_ollama.call_ollama([{"role": "user", "content": "hi"}])
    assert result == "ok"
    assert call_count["n"] == 2


def test_call_ollama_all_models_fail_raises(monkeypatch):
    def fake_post(url, json, timeout):
        raise agentic_fix_ollama.requests.exceptions.ConnectionError("no server")

    monkeypatch.setattr(agentic_fix_ollama.requests, "post", fake_post)
    monkeypatch.setattr(agentic_fix_ollama.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        agentic_fix_ollama.call_ollama([{"role": "user", "content": "hi"}])


# ---------- extract_patch ----------
def test_extract_patch_with_markers():
    raw = "some text\n<<<PATCH_START>>>\n--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new\n<<<PATCH_END>>>\nmore"
    patch = agentic_fix_ollama.extract_patch(raw)
    assert patch.startswith("--- a/foo.py")
    assert "+new" in patch


def test_extract_patch_no_changes():
    raw = "NO_CHANGES"
    patch = agentic_fix_ollama.extract_patch(raw)
    assert patch == ""


# ---------- Property test for extract_patch ----------
@given(st.text(min_size=0, max_size=500))
def test_extract_patch_never_throws(text):
    # It should always return a string without raising
    result = agentic_fix_ollama.extract_patch(text)
    assert isinstance(result, str)
