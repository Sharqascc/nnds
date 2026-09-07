#!/usr/bin/env python3
"""
Autonomous agentic fixer for NNDS.

Parses groq_review_agentic.md to build a task queue, applies
mechanical fixes with built-in AST transformations, and uses
LLM fallback for unknown issues. Validates with property tests,
ruff, and mypy. Commits only if all checks pass. Tracks progress
in agentic_progress.json to avoid repeating completed tasks.
"""

import os
import subprocess
import sys
import json
import re
import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)

PROGRESS_FILE = REPO / "agentic_progress.json"
REPORT_FILE = REPO / "groq_review_agentic.md"

MAX_ATTEMPTS_PER_TASK = 5

def run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, **kwargs)

def get_llm_diff(issue_description):
    """Ask Groq for a code improvement as a unified diff."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("No GROQ_API_KEY set; exiting.")
        sys.exit(1)
    from groq import Groq
    client = Groq(api_key=api_key)
    prompt = f"""
You are an AI coding assistant. The repository has this issue:
{issue_description}

Suggest one small, mechanical code improvement to fix it.
Return ONLY a unified diff that applies cleanly with `git apply`.
"""
    resp = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1500,
    )
    return resp.choices[0].message.content

def apply_diff(diff_text):
    patch_file = REPO / ".agentic.patch"
    patch_file.write_text(diff_text)
    result = run(["git", "apply", str(patch_file)])
    patch_file.unlink(missing_ok=True)
    return result.returncode == 0

def run_full_checks():
    """Run property tests, ruff, and mypy on key directories."""
    checks = [
        ["pytest", "tests/", "-q", "--timeout=120",
         "--ignore=tests/test_snapshot_bev_mapper.py",
         "--ignore=tests/test_snapshot_pet_summary.py",
         "-m", "not integration and not slow",
         "-o", "addopts="],
        ["ruff", "check", "src", "tests", "scripts"],
        ["mypy", "--config-file", "mypy.ini", "src/analysis", "src/diffusion", "src/pipeline", "src/vlm", "src/bev", "src/utils"],
    ]
    for cmd in checks:
        r = run(cmd)
        if r.returncode != 0:
            print(f"❌ Check failed: {' '.join(cmd)}")
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
        print("❌ Commit failed.")
        return False
    push = run(["git", "push", "origin", "HEAD"])
    if push.returncode != 0:
        print("❌ Push failed.")
        return False
    return True

def load_progress():
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed": []}

def save_progress(progress):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))

def parse_tasks():
    """Extract numbered tasks with full descriptions from the review report."""
    text = REPORT_FILE.read_text()
    pattern = re.compile(r'^###\s+(\d+)\.\s+(.+)$', re.MULTILINE)
    matches = list(pattern.finditer(text))
    tasks = []
    for idx, match in enumerate(matches):
        task_id = int(match.group(1))
        title = match.group(2).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        description = text[start:end].strip()
        tasks.append((task_id, title, description))
    return tasks

# ============ Built-in fixers ============

def fix_numpy_aliases():
    """Replace deprecated NumPy scalar aliases."""
    replacements = {
        r'\bnp\.int\b': 'int',
        r'\bnp\.float\b': 'float',
        r'\bnp\.bool\b': 'bool',
        r'\bnp\.object\b': 'object',
        r'\bnp\.str\b': 'str',
    }
    all_py_files = list((REPO / 'src').rglob('*.py')) + list((REPO / 'scripts').rglob('*.py'))
    for f in all_py_files:
        text = f.read_text()
        new = text
        for pat, repl in replacements.items():
            new = re.sub(pat, repl, new)
        if new != text:
            f.write_text(new)

def fix_exception_chaining():
    """Add 'from None' to raise statements lacking a cause."""
    all_py_files = list((REPO / 'src').rglob('*.py')) + list((REPO / 'scripts').rglob('*.py'))
    for f in all_py_files:
        text = f.read_text()
        new_text = re.sub(r'(\s+raise\s+[^\n:]+)(?=\n)', r'\1  from None', text)
        if new_text != text:
            f.write_text(new_text)

def add_ruff_select():
    """Add F401 to ruff select."""
    p = REPO / 'pyproject.toml'
    text = p.read_text()
    if 'select = ["F401"]' not in text:
        if '[tool.ruff]' in text:
            text = text.replace('[tool.ruff]\n', '[tool.ruff]\nselect = ["F401"]\n', 1)
        else:
            text += '\n[tool.ruff]\nselect = ["F401"]\n'
        p.write_text(text)

# Map task_id to built-in function if possible
BUILTIN_MAP = {
    3: fix_numpy_aliases,
    9: fix_exception_chaining,
    15: add_ruff_select,
}

def main():
    progress = load_progress()
    completed = set(progress["completed"])
    tasks = parse_tasks()
    print(f"Loaded {len(tasks)} tasks from report.")

    for task_id, title, description in tasks:
        if task_id in completed:
            print(f"\n✅ Task {task_id} already completed: {title}")
            continue
        print(f"\n🔧 Processing task {task_id}: {title}")

        # Save current state for revert
        run(["git", "stash", "push", "--include-untracked", "-m", f"pre-task-{task_id}"])

        success = False
        if task_id in BUILTIN_MAP:
            print("  Using built-in transformation.")
            try:
                BUILTIN_MAP[task_id]()
                success = True
            except Exception as e:
                print(f"  Built-in failed: {e}")
        else:
            print("  Using LLM fallback.")
            for attempt in range(MAX_ATTEMPTS_PER_TASK):
                print(f"    Attempt {attempt+1}")
                try:
                    diff_text = get_llm_diff(description)
                    if diff_text and "diff --git" in diff_text and apply_diff(diff_text):
                        success = True
                        break
                    else:
                        print("    LLM diff invalid or failed to apply.")
                        run(["git", "checkout", "--", "."])
                except Exception as e:
                    print(f"    LLM error: {e}")
                    run(["git", "checkout", "--", "."])

        if success and run_full_checks():
            print("✅ Checks passed. Committing.")
            if commit_and_push():
                progress["completed"].append(task_id)
                save_progress(progress)
                print("  Successfully committed and pushed.")
            else:
                print("  Commit/push failed. Reverting.")
                run(["git", "checkout", "--", "."])
                run(["git", "stash", "drop"])
        else:
            print("  Checks failed or no change. Reverting.")
            run(["git", "checkout", "--", "."])
            run(["git", "stash", "drop"])

if __name__ == "__main__":
    main()
