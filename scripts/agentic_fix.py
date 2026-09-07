#!/usr/bin/env python3
"""Agentic code review and auto-fix using Groq.

This script:
1. Reviews all Python files under src/ using Groq.
2. Saves a combined report to groq_review_report.md.
3. Applies safe automatic fixes (ruff --fix) before review to clean style issues.

Requires GROQ_API_KEY environment variable.
"""
import os
import subprocess
from pathlib import Path

from groq import Groq

MODEL = "qwen/qwen3.8-27b"  # Update if model changes

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
REPORT_PATH = REPO_ROOT / "groq_review_report.md"


def review_text(code_chunk, file_name, part, total_parts):
    prompt = f"""Review the following Python code from {file_name} (part {part}/{total_parts}).
Identify bugs, missing contracts, style issues, and potential improvements.
Be concise.

Code:
{code_chunk}
"""
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a senior Python code reviewer. Provide actionable feedback."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=500,
    )
    return response.choices[0].message.content


def review_file(path):
    content = path.read_text(encoding="utf-8", errors="ignore")
    chunk_size = 5000
    chunks = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]
    reviews = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  Reviewing {path.relative_to(REPO_ROOT)} chunk {i}/{len(chunks)}...")
        review = review_text(chunk, str(path.relative_to(REPO_ROOT)), i, len(chunks))
        reviews.append(f"## {path.relative_to(REPO_ROOT)} (part {i}/{len(chunks)})\n{review}")
    return "\n\n".join(reviews)


def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY not set. Exiting.")
        return

    # Apply safe style fixes first
    print("Running ruff --fix ...")
    subprocess.run(["ruff", "check", "src", "tests", "--fix"], cwd=REPO_ROOT, check=False)

    # Review all Python files under src/
    py_files = sorted(SRC_DIR.glob("**/*.py"))
    print(f"Found {len(py_files)} Python files under src/")

    report_parts = []
    for f in py_files:
        print(f"\n=== Reviewing {f.relative_to(REPO_ROOT)} ===")
        report_parts.append(review_file(f))

    full_report = "\n\n".join(report_parts)
    REPORT_PATH.write_text(full_report, encoding="utf-8")
    print(f"✅ Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
