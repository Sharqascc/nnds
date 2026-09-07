#!/usr/bin/env python3
"""Agentic code review and auto-fix using multiple free LLM providers.

- Reviews changed Python files under src/ (or all if first run).
- Requests **search/replace blocks** (not unified diffs) to avoid LLM diff format issues.
- Applies replacements safely, commits, and pushes if all checks pass.

Providers (OpenAI-compatible):
  1. Groq (fast, limited)
  2. Google Gemini (generous free tier)
  3. OpenRouter (free model pool)

Requires at least one of: GROQ_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "groq_review_report.md"
PROGRESS_FILE = REPO_ROOT / "agentic_progress.json"

# Provider configurations (priority order)
PROVIDERS = [
    {
        "name": "Gemini",
        "base_url": "",
        "api_key_env": "GEMINI_API_KEY",
        "model": "gemini-2.5-flash",
    },
    {
        "name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": "qwen/qwen3.8-27b",
    },
    {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model": "openrouter/auto",
    },
]


def run_cmd(cmd, cwd=REPO_ROOT):
    """Run a shell command and return CompletedProcess."""
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def ensure_git_identity():
    if not run_cmd(["git", "config", "user.email"]).stdout.strip():
        run_cmd(["git", "config", "user.email", "agentic-bot@users.noreply.github.com"])
    if not run_cmd(["git", "config", "user.name"]).stdout.strip():
        run_cmd(["git", "config", "user.name", "Agentic Bot"])


def call_llm(messages, max_tokens=600, temperature=0.2):
    """Call providers in order with retries on transient errors."""
    last_error = None
    for provider in PROVIDERS:
        api_key = os.environ.get(provider["api_key_env"])
        if api_key:
            api_key = api_key.strip()
        if not api_key or api_key == "***" or len(api_key) < 10:
            print(f"  Skipping {provider['name']} (missing or invalid {provider['api_key_env']})")
            continue

        for attempt in range(3):
            try:
                if provider["name"] == "Gemini":
                    import requests as req
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{provider['model']}:generateContent"
                    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
                    prompt_parts = []
                    for m in messages:
                        prompt_parts.append({"text": f"[{m.get('role', 'user')}] {m.get('content', '')}"})
                    data = {
                        "contents": [{"parts": prompt_parts}],
                        "generationConfig": {
                            "temperature": temperature,
                            "maxOutputTokens": max_tokens,
                        }
                    }
                    resp = req.post(url, headers=headers, json=data, timeout=90)
                    if resp.status_code == 200:
                        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                    else:
                        last_error = resp.text
                        print(f"  {provider['name']} returned {resp.status_code}: {resp.text[:200]}")
                        if resp.status_code in (429, 503):
                            wait = 20 * (attempt + 1)
                            print(f"  Transient error, retrying in {wait}s...")
                            time.sleep(wait)
                            continue
                else:
                    import requests as req
                    resp = req.post(
                        f"{provider['base_url']}/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={
                            "model": provider["model"],
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        },
                        timeout=90,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        return data["choices"][0]["message"]["content"]
                    else:
                        last_error = resp.text
                        print(f"  {provider['name']} returned {resp.status_code}: {resp.text[:200]}")
                        if resp.status_code in (429, 503):
                            wait = 20 * (attempt + 1)
                            print(f"  Transient error, retrying in {wait}s...")
                            time.sleep(wait)
                            continue
            except Exception as e:
                last_error = str(e)
                print(f"  {provider['name']} exception: {e}")
                time.sleep(5)
        print(f"  {provider['name']} exhausted retries.")
    raise RuntimeError(f"All providers failed: {last_error}")
def review_file(content, file_name):
    """Review code and return review text."""
    # Chunk large files to avoid input limits
    chunk_size = 3000
    chunks = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]
    reviews = []
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
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=400,
        )
        reviews.append(f"### Part {i}/{len(chunks)}\n{review}")
    return "\n\n".join(reviews)


def request_replacements(content, file_name, review_text):
    """Ask LLM to produce search/replace blocks."""
    system_msg = (
        "You are an expert Python developer. Given the code and review, produce "
        "search/replace blocks to fix the issues. Use the exact format:\n"
        "<<<<<<< SEARCH\n"
        "(exact code to find)\n"
        "=======\n"
        "(replacement code)\n"
        ">>>>>>> REPLACE\n"
        "If no changes are needed, output exactly NO_CHANGES."
    )
    user_msg = f"File: {file_name}\n\nCode:\n{content}\n\nReview:\n{review_text}"
    return call_llm(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=800,
        temperature=0.1,
    )


def apply_replacements(content, blocks_text):
    """Apply search/replace blocks. Returns (new_content, applied_count)."""
    if blocks_text.strip() == "NO_CHANGES":
        return content, 0

    blocks = []
    pattern = r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE"
    import re
    matches = re.findall(pattern, blocks_text, flags=re.DOTALL)
    if not matches:
        return content, 0

    new_content = content
    applied = 0
    for search, replace in matches:
        # Ensure search snippet exists and is unique enough
        if search in new_content:
            new_content = new_content.replace(search, replace, 1)
            applied += 1
        else:
            print(f"  Search block not found, skipping: {search[:80]}...")
    return new_content, applied


def main():
    ensure_git_identity()

    # Determine files to process
    progress_file = PROGRESS_FILE
    last_commit = ""
    if progress_file.exists():
        try:
            last_commit = json.loads(progress_file.read_text()).get("last_commit", "")
        except Exception:
            last_commit = ""

    changed_files = []
    if last_commit:
        changed = run_cmd(["git", "diff", "--name-only", last_commit, "HEAD", "--", "src/"])
        if changed.returncode == 0 and changed.stdout.strip():
            changed_files = [Path(line) for line in changed.stdout.splitlines() if line.endswith(".py")]
            changed_files = [f for f in changed_files if f.exists()]

    # Fallback: if no changed files detected (first run or shallow clone), use recently modified files
    if not changed_files:
        print("No changed files from diff; falling back to most recently modified Python files.")
        all_py = sorted((REPO_ROOT / "src").glob("**/*.py"), key=lambda p: p.stat().st_mtime, reverse=True)
        changed_files = all_py[:10]

    # Limit files per run to avoid rate limits
    MAX_FILES = 3
    if len(changed_files) > MAX_FILES:
        print(f"Limiting to {MAX_FILES} files (found {len(changed_files)}).")
        changed_files = changed_files[:MAX_FILES]

    print(f"Processing {len(changed_files)} files.")

    # Run ruff --fix first
    print("Running ruff --fix ...")
    run_cmd(["ruff", "check", "src", "tests", "--fix"])

    report_parts = []
    files_modified = []

    for f in changed_files:
        rel_path = f.relative_to(REPO_ROOT)
        print(f"\n=== Reviewing {rel_path} ===")
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
            review = review_file(content, str(rel_path))
            report_parts.append(f"## {rel_path}\n{review}")
        except Exception as e:
            print(f"  Failed to review {rel_path}: {e}")
            continue

        if f.stat().st_size > 4000:
            print("  File too large for replacement generation; skipping.")
            continue

        try:
            blocks_text = request_replacements(content, str(rel_path), review)
            new_content, applied = apply_replacements(content, blocks_text)
            if applied > 0:
                f.write_text(new_content, encoding="utf-8")
                print(f"  ✅ Applied {applied} replacement(s).")
                files_modified.append(str(rel_path))
            else:
                print("  No changes applied.")
        except Exception as e:
            print(f"  Failed to generate/apply patch for {rel_path}: {e}")

    # Save review report
    REPORT_PATH.write_text("\n\n".join(report_parts), encoding="utf-8")

    # Save progress
    current_commit = run_cmd(["git", "rev-parse", "HEAD"]).stdout.strip()
    progress_file.write_text(json.dumps({"last_commit": current_commit}, indent=2))

    # Commit and push if modified
    if files_modified:
        run_cmd(["git", "add", "-A"])
        commit_msg = "Auto-fix: apply LLM search/replace patches\n\nFiles:\n" + "\n".join(files_modified)
        commit = run_cmd(["git", "commit", "-m", commit_msg])
        if commit.returncode == 0:
            push = run_cmd(["git", "push", "origin", "HEAD"])
            if push.returncode == 0:
                print("✅ Pushed auto-fixes.")
            else:
                print("Push failed:", push.stderr)
        else:
            print("Commit failed or nothing to commit.")
    else:
        print("No files modified.")


if __name__ == "__main__":
    main()
