#!/usr/bin/env python3
"""
Autonomous AI code fixer for NNDS.

Uses Groq LLM to propose a unified diff, applies it, runs checks,
and pushes if successful. Retries up to MAX_ATTEMPTS per run.
"""

import os
import subprocess
import sys
import json
import re
from pathlib import Path
from groq import Groq

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)

MAX_ATTEMPTS = 5

def run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, **kwargs)

def get_llm_diff():
    """Ask Groq for a single code improvement as a unified diff."""
    api_key = os.environ.get("GROUQ_API_KEY")
    if not api_key:
        print("No GROUQ_API_KEY set; exiting.")
        sys.exit(1)
    client = Groq(api_key=api_key)

    prompt = """
You are an AI coding assistant. Analyze the repository and suggest one small, mechanical code improvement.
Return ONLY a unified diff in plain text, no explanations. The diff must apply cleanly with `git apply`.
Ensure the diff is against the current working tree.
"""
    resp = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1500,
    )
    content = resp.choices[0].message.content
    return content

def apply_diff(diff_text):
    """Apply unified diff to the working tree."""
    patch_file = REPO / ".agentic.patch"
    patch_file.write_text(diff_text)
    result = run(["git", "apply", str(patch_file)])
    patch_file.unlink(missing_ok=True)
    return result.returncode == 0

def run_checks():
    """Run fast quality gates."""
    checks = [
        ["pytest", "tests/", "-m", "property", "-q", "-o", "addopts="],
        ["ruff", "check", "src", "tests", "scripts"],
        ["mypy", "--config-file", "mypy.ini", "src/analysis", "src/diffusion", "src/pipeline", "src/vlm", "src/bev", "src/utils"],
    ]
    for cmd in checks:
        r = run(cmd)
        if r.returncode != 0:
            print(f"Check failed: {' '.join(cmd)}")
            return False
    return True

def commit_and_push():
    """Commit and push the current changes."""
    run(["git", "config", "user.name", "sharqascc-agent"])
    run(["git", "config", "user.email", "agent@users.noreply.github.com"])
    run(["git", "add", "-A"])
    status = run(["git", "status", "--short"])
    if not status.stdout.strip():
        print("No changes to commit.")
        return True
    commit = run(["git", "commit", "-m", "Agentic auto-fix"])
    if commit.returncode != 0:
        print("Commit failed.")
        return False
    push = run(["git", "push", "origin", "HEAD"])
    return push.returncode == 0

def main():
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\nAttempt {attempt}/{MAX_ATTEMPTS}")
        diff_text = get_llm_diff()
        if not diff_text or "diff --git" not in diff_text:
            print("LLM did not return a valid diff.")
            continue
        if apply_diff(diff_text):
            if run_checks():
                print("Checks passed. Committing.")
                if commit_and_push():
                    print("Successfully committed and pushed.")
                    return 0
            else:
                print("Checks failed. Reverting.")
                run(["git", "checkout", "--", "."])
        else:
            print("Could not apply diff.")
    print("Max attempts reached.")
    return 1

if __name__ == "__main__":
    sys.exit(main())
