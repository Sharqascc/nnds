#!/usr/bin/env python3
"""Agentic code review and auto-fix using Groq with fallback models.

This script:
1. Runs `ruff --fix` to clean style issues.
2. Reviews all Python files under src/ using Groq.
3. Asks the model to generate unified diff patches to fix issues.
4. Applies valid patches and commits/pushes changes.

Requires GROQ_API_KEY environment variable and git identity configured.
"""

import ast
import os
import subprocess
import sys
import time
from pathlib import Path

from groq import Groq


def call_with_fallback(
    client, messages, models, max_retries_per_model=2, base_wait=10, max_tokens=2000
):
    """Try multiple models in order, falling back on rate limit/transient errors."""
    last_error = None
    for model in models:
        for attempt in range(max_retries_per_model):
            try:
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                last_error = e
                if "rate_limit" in str(e) or "429" in str(e):
                    wait = base_wait * (2**attempt)
                    print(
                        f"Rate limit on {model}, attempt {attempt + 1}/{max_retries_per_model}. Waiting {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    print(f"Model {model} failed with {e}. Trying next model...")
                    break
        else:
            continue
    raise RuntimeError(f"All models failed. Last error: {last_error}")


# Available models ordered by preference (fallback on rate limit)
MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "qwen/qwen3.6-27b",
    "allam-2-7b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "gemma-7b-it",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "deepseek-r1-distill-llama-70b",
    "deepseek-r1-distill-qwen-32b",
    "qwen-2.5-32b",
    "qwen-2.5-coder-32b",
    "qwen-qwq-32b",
    "qwen3-30b-a3b",
    "kimi-k2-instruct",
    "glm-4.5-air",
]
REPO_ROOT = Path(__file__).resolve().parents[1]

# Increased thresholds to avoid skipping large files unless truly extreme.
MAX_FILE_CHARS = 200000  # ~50k tokens, within Groq context
MAX_FILE_LINES = 5000
REPORT_PATH = REPO_ROOT / "groq_review_report.md"


def run_cmd(cmd, cwd=REPO_ROOT, timeout=120):
    """Run a shell command and return CompletedProcess."""
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout}s: {cmd}")
        return subprocess.CompletedProcess(cmd, 124, stdout="", stderr="Timeout")


def extract_patch(raw):
    """Extract a valid unified diff from model output, stripping fences and explanations."""
    start_marker = "<<<PATCH_START>>>"
    end_marker = "<<<PATCH_END>>>"
    if start_marker in raw and end_marker in raw:
        start = raw.index(start_marker) + len(start_marker)
        end = raw.index(end_marker)
        patch = raw[start:end].strip("\n")
        # Remove code fences if present
        if patch.startswith("```") and patch.endswith("```"):
            patch = patch[3:-3].strip("\n")
        return patch
    # Fallback: try to find diff-style lines
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


def ensure_git_identity():
    """Set git identity if not already configured (needed for CI)."""
    if run_cmd(["git", "config", "user.email"]).stdout.strip() == "":
        run_cmd(["git", "config", "user.email", "agentic-bot@users.noreply.github.com"])
    if run_cmd(["git", "config", "user.name"]).stdout.strip() == "":
        run_cmd(["git", "config", "user.name", "Agentic Bot"])


def review_and_fix():
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    # Step 1: style fixes
    print("Running ruff --fix ...")
    run_cmd(["ruff", "check", "src", "tests", "--fix"])

    # Step 2: review and patch each file
    py_files = sorted((REPO_ROOT / "src").glob("**/*.py"))
    print(f"Found {len(py_files)} Python files under src/")

    report_parts = []
    fixes_applied = []

    for f in py_files:
        rel_path = f.relative_to(REPO_ROOT)
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
            if len(content) > MAX_FILE_CHARS or len(content.splitlines()) > MAX_FILE_LINES:
                print(
                    f"  ⏭️ Skipping {rel_path} (too large: {len(content)} chars, {len(content.splitlines())} lines)"
                )
                continue

            print(f"\n=== Reviewing {rel_path} ===")

            # Single prompt: ask for review and unified diff in one response
            combined_prompt = f"""You are a senior Python code reviewer and developer.
Analyze the following file {rel_path} for bugs, missing contracts, style issues, and potential improvements.
Then produce a unified diff patch that fixes the issues. If no changes are needed, output exactly NO_CHANGES.
Wrap the patch between:
<<<PATCH_START>>>
... unified diff ...
<<<PATCH_END>>>
Do not include any explanation outside the markers.

File contents:
{content}
"""
            try:
                response = call_with_fallback(
                    client,
                    models=MODELS,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert Python developer. Provide actionable feedback and unified diffs.",
                        },
                        {"role": "user", "content": combined_prompt},
                    ],
                    max_tokens=4000,
                )
                raw_output = response.choices[0].message.content
                # Save review text (outside patch) for report if possible
                # For simplicity, we only store the patch; report could be separate but we skip
                patch = extract_patch(raw_output)
                if patch.strip() == "NO_CHANGES":
                    print("  No changes suggested.")
                    report_parts.append(f"## {rel_path}\nNo changes suggested.")
                elif patch:
                    # Validate patch syntax by applying it with git apply
                    patch_path = REPO_ROOT / "temp_patch.diff"
                    patch_path.write_text(patch, encoding="utf-8")
                    # Check if patch applies cleanly
                    check = run_cmd(["git", "apply", "--check", str(patch_path)])
                    if check.returncode != 0:
                        print(f"  ❌ Patch does not apply cleanly for {rel_path}: {check.stderr}")
                        patch_path.unlink(missing_ok=True)
                        continue
                    # Apply patch
                    apply = run_cmd(["git", "apply", str(patch_path)])
                    if apply.returncode == 0:
                        print(f"  ✅ Applied patch for {rel_path}")
                        fixes_applied.append(str(rel_path))
                        report_parts.append(f"## {rel_path}\nPatch applied.")
                    else:
                        print(f"  ❌ Failed to apply patch for {rel_path}: {apply.stderr}")
                    patch_path.unlink(missing_ok=True)
                else:
                    print("  ⚠️ No valid patch extracted; skipping.")
            except Exception as e:
                print(f"  ❌ LLM call failed for {rel_path}: {e}")
                continue
        except Exception as e:
            print(f"  ❌ Unexpected error for {rel_path}: {e}")
            continue

    # Save review report
    REPORT_PATH.write_text("\n\n".join(report_parts), encoding="utf-8")

    # Commit and push if fixes were applied
    if fixes_applied:
        if not run_quick_tests():
            print("❌ Quick tests failed. Reverting patches and aborting.")
            run_cmd(["git", "restore", "."])
            sys.exit(1)
        ensure_git_identity()
        run_cmd(["git", "add", "-A"])
        commit_msg = "Auto-fix: apply LLM-generated patches\n\nFiles:\n" + "\n".join(fixes_applied)
        commit = run_cmd(["git", "commit", "-m", commit_msg])
        if commit.returncode == 0:
            push = run_cmd(["git", "push", "--no-verify", "origin", "HEAD"])
            if push.returncode == 0:
                print("✅ Pushed auto-fixes.")
            else:
                print("Push failed:", push.stderr)
        else:
            print("No changes to commit or commit failed.")
    else:
        print("No fixes applied.")


def run_quick_tests():
    """Run a fast subset of tests and return True if they pass."""
    print("\nRunning quick tests...")
    res = run_cmd(
        [
            "pytest",
            "tests/test_pet_summary_property.py",
            "tests/test_traffic_analyzer_property.py",
            "tests/test_core_validation_property.py",
            "-q",
            "-o",
            "addopts=",
        ]
    )
    print(res.stdout)
    print(res.stderr)
    return res.returncode == 0


if __name__ == "__main__":
    try:
        if not os.environ.get("GROQ_API_KEY"):
            print("GROQ_API_KEY not set. Exiting.")
            sys.exit(1)
        review_and_fix()
    except Exception as e:
        print(f"❌ Agentic fixer failed: {e}")
        sys.exit(1)
