#!/usr/bin/env python3
"""Agentic code review and auto-fix using Groq.

This script:
1. Runs `ruff --fix` to clean style issues.
2. Reviews all Python files under src/ using Groq.
3. Asks the model to generate unified diff patches to fix issues.
4. Applies valid patches and commits/pushes changes.

Requires GROQ_API_KEY environment variable and git identity configured.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

from groq import Groq


def call_with_fallback(client, messages, models, max_retries_per_model=2, base_wait=10):
    """Try multiple models in order, falling back on rate limit/transient errors."""
    last_error = None
    for model in models:
        for attempt in range(max_retries_per_model):
            try:
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=800,
                )
            except Exception as e:
                last_error = e
                if "rate_limit" in str(e) or "429" in str(e):
                    wait = base_wait * (2 ** attempt)
                    print(f"Rate limit on {model}, attempt {attempt+1}/{max_retries_per_model}. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    # Non-rate-limit error: break retry loop for this model and try next
                    print(f"Model {model} failed with {e}. Trying next model...")
                    break
        else:
            continue
        # If all retries on a model exhausted (rate limit), continue to next model
        continue
    raise RuntimeError(f"All models failed. Last error: {last_error}")


# Available models ordered by preference (fallback on rate limit)
MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "qwen/qwen3.6-27b",
    "allam-2-7b",
]
REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "groq_review_report.md"


def run_cmd(cmd, cwd=REPO_ROOT):
    """Run a shell command and return CompletedProcess."""
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


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
    py_files = sorted((REPO_ROOT / "src").glob("**/*.py"))[:20]
    print(f"Found {len(py_files)} Python files under src/")

    report_parts = []
    fixes_applied = []

    for f in py_files:
        rel_path = f.relative_to(REPO_ROOT)
        content = f.read_text(encoding="utf-8", errors="ignore")
        print(f"\n=== Reviewing {rel_path} ===")

        # Review
        review_prompt = f"""Review the following Python code from {rel_path}.
Identify bugs, missing contracts, style issues, and potential improvements.
Be concise.

Code:
{content}
"""
        review_resp = call_with_fallback(
            client,
            models=MODELS,
            messages=[
                {"role": "system", "content": "You are a senior Python code reviewer. Provide actionable feedback."},
                {"role": "user", "content": review_prompt},
            ],
            model=MODEL,
        )
        review_text = review_resp.choices[0].message.content
        report_parts.append(f"## {rel_path}\n{review_text}")

        # Request patch
        patch_prompt = f"""Given the code and the review, produce a unified diff patch to fix the issues.
Output only the patch. If no changes are needed, output exactly NO_CHANGES.
Do not include any explanation or markdown fences.

Code:
{content}

Review:
{review_text}
"""
        patch_resp = call_with_fallback(
            client,
            models=MODELS,
            messages=[
                {"role": "system", "content": "You are an expert Python developer. Provide a valid unified diff patch."},
                {"role": "user", "content": patch_prompt},
            ],
            model=MODEL,
        )
        patch_text = patch_resp.choices[0].message.content.strip()

        if patch_text != "NO_CHANGES":
            patch_file = REPO_ROOT / "temp_patch.diff"
            patch_file.write_text(patch_text)
            # Verify patch applies cleanly
            check = run_cmd(["git", "apply", "--check", str(patch_file)])
            if check.returncode == 0:
                apply = run_cmd(["git", "apply", str(patch_file)])
                if apply.returncode == 0:
                    print(f"  ✅ Applied patch for {rel_path}")
                    fixes_applied.append(str(rel_path))
                else:
                    print(f"  ❌ Failed to apply patch for {rel_path}: {apply.stderr}")
            else:
                print(f"  ❌ Patch check failed for {rel_path}: {check.stderr}")
            patch_file.unlink(missing_ok=True)
        else:
            print("  No changes suggested.")

    # Save review report
    REPORT_PATH.write_text("\n\n".join(report_parts), encoding="utf-8")

    # Commit and push if fixes were applied
    if fixes_applied:
        if not run_quick_tests():
            print("❌ Quick tests failed. Reverting patches and aborting.")
            run_cmd(["git", "restore", "."])
            return
        ensure_git_identity()
        run_cmd(["git", "add", "-A"])
        commit_msg = "Auto-fix: apply LLM-generated patches\n\nFiles:\n" + "\n".join(fixes_applied)
        commit = run_cmd(["git", "commit", "-m", commit_msg])
        if commit.returncode == 0:
            push = run_cmd(["git", "push", "origin", "HEAD"])
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
    res = run_cmd([
        "pytest", "tests/test_pet_summary_property.py",
        "tests/test_traffic_analyzer_property.py",
        "tests/test_core_validation_property.py",
        "-q", "-o", "addopts=",
    ])
    print(res.stdout)
    print(res.stderr)
    return res.returncode == 0

if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY not set. Exiting.")
        sys.exit(1)
    review_and_fix()
