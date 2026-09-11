"""Ollama-based agentic fix helper."""

from __future__ import annotations

import os
import re
import time
from typing import Any

import requests

OLLAMA_MODELS = [
    os.environ.get("OLLAMA_MODEL_1", "qwen2.5-coder:7b"),
    os.environ.get("OLLAMA_MODEL_2", "llama3.1:8b"),
    os.environ.get("OLLAMA_MODEL_3", "deepseek-coder:6.7b"),
]
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

MAX_FILE_LINES = 2000
MAX_FILE_CHARS = 200_000

PATCH_START = "<<<PATCH_START>>>"
PATCH_END = "<<<PATCH_END>>>"


def call_ollama(messages: list[dict[str, Any]]) -> str:
    """Call Ollama, trying each model in turn. Raise RuntimeError if all fail."""
    last_exc: Exception | None = None
    for model in OLLAMA_MODELS:
        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={"model": model, "messages": messages, "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except requests.exceptions.ConnectionError as exc:
            last_exc = exc
            time.sleep(1.0)
            continue
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            continue
    raise RuntimeError(f"All Ollama models failed: {last_exc}")


def extract_patch(raw: str) -> str:
    """Extract a unified diff between PATCH_START/PATCH_END markers.

    Always returns a string; never raises.
    """
    if not isinstance(raw, str):
        return ""
    if "NO_CHANGES" in raw:
        return ""
    m = re.search(
        re.escape(PATCH_START) + r"(.*?)" + re.escape(PATCH_END),
        raw,
        flags=re.DOTALL,
    )
    return m.group(1).strip() if m else ""
