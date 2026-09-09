#!/usr/bin/env python3
"""Agentic code review and auto-fix using local Ollama models.

This script is an alternative to scripts/agentic_fix.py. Instead of cloud
providers, it uses a local Ollama server (e.g., in Google Colab or a developer
machine) to review Python files and apply unified diffs.

Requirements:
- Ollama installed and running on localhost:11434
- At least one coding model pulled, e.g.:
    ollama pull qwen2.5-coder:7b
- For Colab: mount Google Drive, set OLLAMA_MODELS to Drive folder, start server.

Usage:
    python scripts/agentic_fix_ollama.py
"""

import ast
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]

# Local Ollama models to try in order (fallback on connection or model errors)
OLLAMA_MODELS = [
    "qwen2.5-coder:7b",
    "deepseek-coder:6.7b",
    "codellama:7b",
]

# Increased thresholds to avoid skipping large files unless truly extreme.
MAX_FILE_CHARS = 200000
MAX_FILE_LINES = 5000

OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"


def call_ollama(messages, models=OLLAMA_MODELS, max_retries_per_model=1, base_wait=5):
    """Try multiple Ollama models in sequence, returning the first successful response."""
    last_error = None
    for model in models:
        for attempt in range(max_retries_per_model):
            try:
                resp = requests.post(
                    OLLAMA_ENDPOINT,
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["message"]["content"]
            except Exception as e:
                last_error = e
                if isinstance(e, requests.exceptions.ConnectionError):
                    # Server may not be ready; wait and retry
                    time.sleep(base_wait * (2**attempt))
                else:
                    print(f"Model {model} failed with {e}. Trying next model...")
                    break
        else:
            continue
    raise RuntimeError(f"All Ollama models failed. Last error: {last_error}")


def run_cmd(cmd, cwd=REPO_ROOT, timeout=120):
    """Run a shell command and return CompletedProcess."""
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout}s: {cmd}")
        return subprocess.CompletedProcess(cmd, 124, stdout="", stderr="Timeout")


def extract_patch(raw):
    """Extract a unified diff from Ollama output, stripping fences and explanations."""
    start_marker = "<<<PATCH_START>>>"
    end_marker = "<<<PATCH_END>>>"
    if start_marker in raw and end_marker in raw:
        start = raw.index(start_marker) + len(start_marker)
        end = raw.index(end_marker)
        patch = raw[start:end].strip("\n")
        if patch.startswith("```") and patch.endswith("```"):
            patch = patch[3:-3].strip("\n")
        return patch
    # Fallback: try to find diff lines
    lines = raw.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if line.startswith(("--- ", "*** ", "+++ ", "@@ ")):
            start_idx = i
            break
    if start_idx is None:
        return ""
    end_idx = len(lines)
    for i in range(start_idx, len(lines)):
        if lines[i].startswith("```") or lines[i].strip() == "NO_CHANGES":
            end_idx = i
            break
    return "\n".join(lines[start_idx:end_idx]).strip() + "\n"


def review_and_fix():
    print("Running ruff --fix ...")
    run_cmd(["ruff", "check", "src", "tests", "--fix"])

    py_files = sorted((REPO_ROOT / "src").glob("**/*.py"))
    print(f"Found {len(py_files)} Python files under src/")

    fixes_applied = []

    for f in py_files:
        rel_path = f.relative_to(REPO_ROOT)
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
            if len(content) > MAX_FILE_CHARS or len(content.splitlines()) > MAX_FILE_LINES:
                print(f"  Skipping {rel_path} (too large)")
                continue

            print(f"\n=== Reviewing {rel_path} ===")
            combined_prompt = f"""You are a rigorous senior Python code reviewer.
Your task is to find and fix at least one real issue in the code below.

Common issues to look for:
- Bugs (e.g., division by zero, mutable default arguments, incorrect logic)
- Style violations (PEP 8, unsorted imports, inconsistent naming)
- Missing type hints where appropriate
- Resource leaks (files not closed, connections not released)
- Error handling gaps (missing exceptions, overly broad catches)
- Performance issues (redundant loops, inefficient data structures)

Analyze the code line by line. If you find ANY issue, produce a unified diff patch that fixes it.
Only output NO_CHANGES if you are 100% certain the code is perfect in every way.
Never output NO_CHANGES for code with obvious bugs or style problems.

Wrap the patch between these markers:
<<<PATCH_START>>>
... unified diff ...
<<<PATCH_END>>>

Do not include any explanation outside the markers.

File: {rel_path}

File contents:
{content}
"""
            try:
                raw_output = call_ollama(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert Python developer. Provide unified diffs.",
                        },
                        {"role": "user", "content": combined_prompt},
                    ]
                )
                patch = extract_patch(raw_output)
                if patch.strip() == "NO_CHANGES":
                    print("  No changes suggested.")
                elif patch:
                    patch_path = REPO_ROOT / "temp_ollama_patch.diff"
                    patch_path.write_text(patch, encoding="utf-8")
                    check = run_cmd(["git", "apply", "--check", str(patch_path)])
                    if check.returncode == 0:
                        apply = run_cmd(["git", "apply", str(patch_path)])
                        if apply.returncode == 0:
                            print(f"  Applied patch for {rel_path}")
                            fixes_applied.append(str(rel_path))
                        else:
                            print(f"  Failed to apply patch: {apply.stderr}")
                    else:
                        print(f"  Patch check failed: {check.stderr}")
                    patch_path.unlink(missing_ok=True)
                else:
                    print("  No valid patch extracted")
            except Exception as e:
                print(f"  LLM call failed for {rel_path}: {e}")
        except Exception as e:
            print(f"  Unexpected error for {rel_path}: {e}")

    if fixes_applied:
        run_cmd(["git", "add", "-A"])
        commit_msg = "Auto-fix (Ollama): apply local LLM patches\n\nFiles:\n" + "\n".join(
            fixes_applied
        )
        run_cmd(["git", "commit", "-m", commit_msg])
        print("Committed changes. Please push manually or configure push.")

    return fixes_applied


def push_changes_to_new_branch(fixes_applied, base_branch="cleanup/system-reorganization"):
    """Create a new branch, commit changes, push, and optionally open a PR."""
    import datetime
    import subprocess as sp

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    branch_name = f"auto-fix-ollama-{timestamp}"

    run_cmd(["git", "checkout", "-b", branch_name])
    run_cmd(["git", "add", "-A"])
    commit_msg = "Auto-fix (Ollama): apply local LLM patches\n\nFiles:\n" + "\n".join(fixes_applied)
    commit_res = run_cmd(["git", "commit", "-m", commit_msg])
    if commit_res.returncode != 0:
        print("Commit failed or no changes; returning")
        return None

    push_res = run_cmd(["git", "push", "--no-verify", "origin", branch_name])
    if push_res.returncode != 0:
        print("Push failed:", push_res.stderr)
        return None

    print(f"Pushed to branch {branch_name}")

    # Try to create a PR if gh CLI is available
    gh_check = run_cmd(["gh", "--version"])
    if gh_check.returncode == 0:
        pr_res = run_cmd(
            [
                "gh",
                "pr",
                "create",
                "--base",
                base_branch,
                "--head",
                branch_name,
                "--title",
                "Ollama Auto-Fix",
                "--body",
                "Automated fixes generated by local Ollama agent.",
            ]
        )
        if pr_res.returncode == 0:
            print("Pull request created.")
        else:
            print("PR creation failed:", pr_res.stderr)
    return branch_name


if __name__ == "__main__":
    fixes = review_and_fix()
    if fixes and os.environ.get("OLLAMA_AUTO_FIX_BRANCH") == "true":
        push_changes_to_new_branch(fixes)
