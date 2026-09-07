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

MODEL_FALLBACK = [
    "qwen/qwen3.8-27b",
    "groq/compound-mini",
    "allam-2-7b",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "groq_review_report.md"


def run_cmd(cmd, cwd=REPO_ROOT):
    """Run a shell command and return CompletedProcess."""
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def ensure_git_identity():
    """Set git identity if not already configured (needed for CI)."""
    if not run_cmd(["git", "config", "user.email"]).stdout.strip():
        run_cmd(["git", "config", "user.email", "agentic-bot@users.noreply.github.com"])
    if not run_cmd(["git", "config", "user.name"]).stdout.strip():
        run_cmd(["git", "config", "user.name", "Agentic Bot"])


def get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not set. Exiting.")
        sys.exit(1)
    return Groq(api_key=api_key)


def call_llm(client, messages, max_tokens=600, temperature=0.2):
    """Call Groq with model fallback."""
    last_error = None
    for model in MODEL_FALLBACK:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_error = e
            print(f"    Model {model} failed: {e}. Trying next...")
            time.sleep(2)
    raise RuntimeError(f"All models failed: {last_error}")


def review_file(client, file_path):
    """Review a single Python file and return review text."""
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    chunk_size = 3000
    chunks = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]
    review_parts = []
    for i, chunk in enumerate(chunks, 1):
        system_msg = (
            "You are a senior Python code reviewer. Identify bugs, missing contracts, "
            "style issues, and potential improvements. Be concise."
        )
        user_msg = (
            f"Review the following Python code (part {i}/{len(chunks)}). "
            "Focus on correctness and maintainability:\n\n" + chunk
        )
        review = call_llm(
            client,
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=500,
        )
        review_parts.append(f"### Part {i}/{len(chunks)}\n{review}")
    return "\n\n".join(review_parts)


def get_patch(client, file_path, review_text):
    """Ask model to produce a unified diff patch to fix issues."""
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    # Limit content to avoid input size errors
    MAX_CONTENT = 4000
    truncated = content[:MAX_CONTENT]
    if len(content) > MAX_CONTENT:
        truncated += "\n... [truncated]"
    system_msg = (
        "You are an expert Python developer. Provide a valid unified diff patch "
        "to fix the issues. Output only the patch. If no changes are needed, "
        "output exactly NO_CHANGES."
    )
    user_msg = f"Code:\n{truncated}\n\nReview:\n{review_text}"
    patch_text = call_llm(
        client,
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=1000,
        temperature=0.1,
    )
    return patch_text.strip()


def apply_patch(patch_text):
    """Apply patch if valid. Returns True if applied, False otherwise."""
    if patch_text == "NO_CHANGES":
        return False
    patch_file = REPO_ROOT / "temp_patch.diff"
    patch_file.write_text(patch_text)
    check = run_cmd(["git", "apply", "--check", str(patch_file)])
    if check.returncode != 0:
        patch_file.unlink(missing_ok=True)
        return False
    apply = run_cmd(["git", "apply", str(patch_file)])
    patch_file.unlink(missing_ok=True)
    return apply.returncode == 0


def main():
    client = get_client()
    ensure_git_identity()

    # Step 1: style fixes
    print("Running ruff --fix ...")
    run_cmd(["ruff", "check", "src", "tests", "--fix"])

    # Step 2: review and patch each file under src/
    py_files = sorted((REPO_ROOT / "src").glob("**/*.py"))
    print(f"Found {len(py_files)} Python files under src/")

    report_parts = []
    fixes_applied = []

    for f in py_files:
        rel_path = f.relative_to(REPO_ROOT)
        print(f"\n=== Reviewing {rel_path} ===")
        review = review_file(client, f)
        report_parts.append(f"## {rel_path}\n{review}")

        # Skip patch generation for very large files to avoid input size errors
        if f.stat().st_size > 4000:
            print("  File too large for patch generation; skipping patch.")
            continue
        patch_text = get_patch(client, f, review)
        if patch_text == "NO_CHANGES":
            print("  No changes suggested.")
            continue
        print("  Patch proposed. Checking validity...")
        if apply_patch(patch_text):
            print("  ✅ Applied patch.")
            fixes_applied.append(str(rel_path))
        else:
            print("  ❌ Patch invalid or did not apply cleanly; skipping.")

    # Save review report
    REPORT_PATH.write_text("\n\n".join(report_parts), encoding="utf-8")

    # Step 3: commit and push if fixes applied
    if fixes_applied:
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
            print("Commit failed or nothing to commit:", commit.stderr)
    else:
        print("No fixes applied.")


if __name__ == "__main__":
    main()
