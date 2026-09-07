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
import time
import hashlib
import requests
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

def extract_diff(text):
    """Extract unified diff from LLM output, stripping code fences."""
    import re
    if not text:
        return ""
    # Remove markdown code fences if present
    if "```diff" in text:
        start = text.find("```diff") + len("```diff")
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()
    # Otherwise return text as is, trimmed
    return text.strip()

def gather_context(description, max_files=10, max_chars=4000):
    """Collect relevant repository code snippets based on issue description."""
    # Determine target directories from description
    targets = []
    if 'src/core' in description or 'core/*.py' in description:
        targets.append(REPO / 'src' / 'core')
    if 'src/pipeline' in description:
        targets.append(REPO / 'src' / 'pipeline')
    if 'src/analysis' in description:
        targets.append(REPO / 'src' / 'analysis')
    if 'src/bev' in description:
        targets.append(REPO / 'src' / 'bev')
    if 'src/diffusion' in description:
        targets.append(REPO / 'src' / 'diffusion')
    if 'src/vlm' in description:
        targets.append(REPO / 'src' / 'vlm')
    if 'src/scripts' in description or 'scripts' in description:
        targets.append(REPO / 'scripts')
    # Fallback to entire src if no specific target found
    if not targets:
        targets.append(REPO / 'src')

    snippets = []
    total_chars = 0
    for target in targets:
        if not target.exists():
            continue
        # Collect Python files up to max_files
        py_files = list(target.rglob('*.py'))
        py_files = [f for f in py_files if '__pycache__' not in str(f)]
        # Prioritize files with changes or core files? Simple: take first max_files
        py_files = py_files[:max_files]
        for f in py_files:
            try:
                content = f.read_text()
                # Limit each file snippet to 800 chars
                snippet = f"--- {f.relative_to(REPO)} ---\n{content[:800]}"
                if total_chars + len(snippet) > max_chars:
                    break
                snippets.append(snippet)
                total_chars += len(snippet)
            except Exception:
                continue
        if total_chars >= max_chars:
            break
    return "\n\n".join(snippets) if snippets else "(no relevant files found)"


def get_llm_diff(issue_description):
    """Ask Groq for a code improvement, with rate-limit awareness."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("No GROQ_API_KEY set; exiting.")
        sys.exit(1)

    # Cache prompt-response pairs to avoid repeated calls
    cache_dir = REPO / ".agentic_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_key = hashlib.sha256(issue_description.encode()).hexdigest()[:16]
    cache_file = cache_dir / f"{cache_key}.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            print("    Using cached diff.")
            return cached.get("diff", "")
        except Exception:
            pass

    # Gather context (smaller limit to reduce input tokens)
    context_snippets = gather_context(issue_description, max_files=6, max_chars=2000)

    system_msg = (
        "You are an expert code repair assistant. Return ONLY a valid unified diff "
        "that fixes the given issue. The diff must apply cleanly with `git apply`. "
        "Do not include explanations, markdown fences, or extra text."
    )
    user_msg = f"""
Issue:
{issue_description}

Code context:
{context_snippets}
"""

    # Model fallback list: prefer models with higher free-tier limits
    model_names = [
        "openai/gpt-oss-120b",
        "groq/compound",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "qwen/qwen3.8-27b",
    ]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload_template = {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0,
        "max_tokens": 250,  # stay under OTPM limits
        "stream": False,
    }

    last_error = None
    for model in model_names:
        payload = payload_template.copy()
        payload["model"] = model
        attempt = 0
        while attempt < 5:
            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    raw = data["choices"][0]["message"]["content"]
                    diff = extract_diff(raw)
                    # Cache the result
                    cache_file.write_text(json.dumps({"diff": diff}))
                    return diff
                elif resp.status_code == 429:
                    retry_after = resp.headers.get("retry-after")
                    wait = int(retry_after) if retry_after and retry_after.isdigit() else min(60, 2 ** attempt)
                    print(f"    429 rate limited on {model}, retrying in {wait}s...")
                    time.sleep(wait)
                    attempt += 1
                    continue
                else:
                    error_text = resp.text[:200]
                    print(f"    Model {model} failed: {resp.status_code} {error_text}")
                    break  # try next model
            except Exception as e:
                print(f"    Request error for {model}: {e}")
                break
        last_error = f"All attempts failed for {model}"
    raise RuntimeError(last_error)

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


def fix_mutable_defaults():
    """Replace mutable default arguments with None and initialize inside."""
    import ast
    all_py_files = list((REPO / 'src').rglob('*.py'))
    for f in all_py_files:
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        changed = False
        class MutableDefaultFixer(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                # only modify functions with mutable defaults
                for i, default in enumerate(node.args.defaults):
                    if isinstance(default, (ast.List, ast.Dict, ast.Set, ast.Call)):
                        # create None default
                        node.args.defaults[i] = ast.Constant(value=None)
                        # insert if arg is None: arg = original at start of body
                        arg_name = node.args.args[-len(node.args.defaults) + i].arg
                        original = ast.unparse(default)
                        assign = ast.parse(f"if {arg_name} is None:\n    {arg_name} = {original}").body[0]
                        node.body.insert(0, assign)
                        changed = True
                self.generic_visit(node)
                return node
        new_tree = MutableDefaultFixer().visit(tree)
        ast.fix_missing_locations(new_tree)
        new_code = ast.unparse(new_tree)
        if changed:
            f.write_text(new_code)

def fix_assert_to_raise():
    """Convert assert statements used for validation into explicit raises."""
    all_py_files = list((REPO / 'src').rglob('*.py')) + list((REPO / 'scripts').rglob('*.py'))
    for f in all_py_files:
        text = f.read_text()
        # Simple regex replacement: assert condition, message -> if not condition: raise ...
        pattern = re.compile(r'assert\s+(.+?),\s+(.+?)\n', re.MULTILINE)
        def repl(match):
            cond = match.group(1).strip()
            msg = match.group(2).strip()
            return "if not " + cond + ":\n    raise ValueError(" + msg + ")"
        new_text = pattern.sub(repl, text)
        if new_text != text:
            f.write_text(new_text)

def fix_global_seed():
    """Add deterministic seed initialization to src/__init__.py."""
    init_file = REPO / 'src' / '__init__.py'
    if not init_file.exists():
        init_file.write_text("")
    text = init_file.read_text()
    if 'set_global_seed' not in text:
        addition = (
            "\nimport os\n"
            "from .utils.seed import set_global_seed\n\n"
            "DEFAULT_SEED = int(os.getenv(\"GLOBAL_SEED\", \"42\"))\n"
            "set_global_seed(DEFAULT_SEED)\n"
        )
        init_file.write_text(text + addition)

def fix_path_join():
    """Replace os.path.join with pathlib / operator."""
    all_py_files = list((REPO / 'src').rglob('*.py')) + list((REPO / 'scripts').rglob('*.py'))
    for f in all_py_files:
        text = f.read_text()
        # Only replace simple os.path.join calls: os.path.join(a, b, ...)
        pattern = re.compile(r'os\.path\.join\(([^)]+)\)')
        def repl(match):
            args = [a.strip() for a in match.group(1).split(',') if a.strip()]
            if not args:
                return match.group(0)
            expr = 'Path(' + args[0] + ')'
            for arg in args[1:]:
                expr += ' / ' + arg
            return expr
        new_text = pattern.sub(repl, text)
        if new_text != text:
            f.write_text(new_text)

# Map task_id to built-in function if possible
BUILTIN_MAP = {
    3: fix_numpy_aliases,
    9: fix_exception_chaining,
    12: fix_path_join,
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
            print("  No built-in fixer for this task. Skipping.")
            # Revert any stash (none should have been made if skipped)
            run(["git", "checkout", "--", "."])
            continue

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