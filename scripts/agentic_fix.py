#!/usr/bin/env python3
"""
Autonomous agentic fixer for NNDS.

Runs multiple mechanical fixes with built-in transformations and
LLM fallback for unknown issues. Validates with property tests,
ruff, and mypy. Commits only if all checks pass.
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

MAX_ATTEMPTS_PER_TASK = 5

def run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, **kwargs)

def get_llm_diff():
    """Ask Groq for a code improvement as a unified diff."""
    api_key = os.environ.get("GROUQ_API_KEY")
    if not api_key:
        print("No GROUQ_API_KEY set; exiting.")
        sys.exit(1)
    from groq import Groq
    client = Groq(api_key=api_key)
    prompt = """
You are an AI coding assistant. Suggest one small, mechanical code improvement.
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
        ["pytest", "tests/", "-m", "property", "-q", "-o", "addopts="],
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

# ============ Built-in fixers ============

def fix_numpy_aliases(files):
    """Replace deprecated NumPy scalar aliases with built-in types."""
    replacements = {
        r'\bnp\.int\b': 'int',
        r'\bnp\.float\b': 'float',
        r'\bnp\.bool\b': 'bool',
        r'\bnp\.object\b': 'object',
        r'\bnp\.str\b': 'str',
    }
    for f in files:
        text = f.read_text()
        new = text
        for pat, repl in replacements.items():
            new = re.sub(pat, repl, new)
        if new != text:
            f.write_text(new)
            print(f"Fixed NumPy aliases in {f.relative_to(REPO)}")

def fix_exception_chaining(files):
    """Add 'from None' to raise statements that lack a cause."""
    for f in files:
        text = f.read_text()
        new_text = re.sub(r'(\s+raise\s+[^\n:]+)(?=\n)', r'\1  from None', text)
        if new_text != text:
            f.write_text(new_text)
            print(f"Added exception chaining in {f.relative_to(REPO)}")

def add_ruff_select():
    """Add F401 to ruff select in pyproject.toml."""
    p = REPO / 'pyproject.toml'
    text = p.read_text()
    if 'select = ["F401"]' not in text:
        if '[tool.ruff]' in text:
            text = text.replace('[tool.ruff]\n', '[tool.ruff]\nselect = ["F401"]\n', 1)
        else:
            text += '\n[tool.ruff]\nselect = ["F401"]\n'
        p.write_text(text)
        print("Added F401 to ruff select")

def main():
    # List of built-in fixes to apply
    all_py_files = list((REPO / 'src').rglob('*.py')) + list((REPO / 'scripts').rglob('*.py'))
    tasks = [
        ("numpy_aliases", lambda: fix_numpy_aliases(all_py_files)),
        ("exception_chaining", lambda: fix_exception_chaining(all_py_files)),
        ("ruff_select", add_ruff_select),
    ]

    for task_name, task_fn in tasks:
        print(f"\n🔧 Processing task: {task_name}")
        run(["git", "stash", "push", "--include-untracked", "-m", f"pre-{task_name}"])
        task_fn()
        if run_full_checks():
            print("✅ Checks passed. Committing.")
            if commit_and_push():
                print("Successfully committed and pushed.")
            else:
                print("❌ Commit/push failed. Reverting.")
                run(["git", "checkout", "--", "."])
                run(["git", "stash", "drop"])
        else:
            print("❌ Checks failed. Reverting.")
            run(["git", "checkout", "--", "."])
            run(["git", "stash", "drop"])
        print(f"Done with task {task_name}")

if __name__ == "__main__":
    main()
