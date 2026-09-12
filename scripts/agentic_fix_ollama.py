"""Utilities for the local Ollama agentic fixer."""

import re
import subprocess
import time
from typing import Any

import requests

OLLAMA_MODELS = ["model1", "model2", "model3"]
MAX_FILE_LINES = 400
MAX_FILE_CHARS = 16000
OLLAMA_URL = "http://localhost:11434/api/chat"
REQUEST_TIMEOUT = 120


def run_cmd(
    cmd: list[str],
    timeout: int = 120,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            **kwargs,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            cmd,
            returncode=124,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "timeout",
        )


def extract_patch(raw: str) -> str:
    if not raw:
        return ""

    text = str(raw).strip()

    if text.upper() == "NO_CHANGES":
        return ""

    if "<<<PATCH_START>>>" in text:
        text = text.split("<<<PATCH_START>>>", 1)[1]
        if "<<<PATCH_END>>>" in text:
            text = text.split("<<<PATCH_END>>>", 1)[0]

    match = re.search(
        r"```(?:diff|patch)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        text = match.group(1)

    start = text.find("diff --git ")
    if start >= 0:
        text = text[start:]
    else:
        start = text.find("--- ")
        if start >= 0:
            text = text[start:]

    return text.strip()


def call_ollama(messages: list[dict[str, str]]) -> str:
    last_error: Exception | None = None

    for model in OLLAMA_MODELS:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }

        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama model {model} returned HTTP "
                    f"{response.status_code}"
                )

            content = response.json().get("message", {}).get("content", "")
            if content is not None:
                return str(content)

            raise RuntimeError(f"Ollama model {model} returned no content")

        except requests.exceptions.RequestException as exc:
            last_error = exc
            time.sleep(1)

        except Exception as exc:
            last_error = exc
            time.sleep(1)

    raise RuntimeError("All Ollama models failed") from last_error
