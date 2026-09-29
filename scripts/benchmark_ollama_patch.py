#!/usr/bin/env python3
"""Benchmark a local Ollama model on a small, safe code-fix task.

The task is deliberately isolated from the scientific pipeline:
- src/utils/duration_parser.py contains a known bug (empty string should raise).
- tests/test_duration_parser.py contains a failing test.

The script:
1. Reads both files.
2. Asks the model to produce a unified diff fixing the bug.
3. Applies the diff if `git apply --check` passes.
4. Runs the specific test file.
5. Reports response time, patch validity, test pass/fail.

Usage:
    python scripts/benchmark_ollama_patch.py [model_name]
Default model: qwen2.5-coder:1.5b
"""

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import agentic_fix_ollama


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5-coder:1.5b"
    print(f"Using model: {model}")

    target_file = REPO_ROOT / "src" / "utils" / "duration_parser.py"
    test_file = REPO_ROOT / "benchmarks" / "ollama_patch" / "test_duration_parser.py"

    target_content = target_file.read_text()
    test_content = test_file.read_text()

    prompt = f"""You are a senior Python developer.
The file `src/utils/duration_parser.py` has a bug: an empty string should raise ValueError,
but currently returns 0.0. A test in `tests/test_duration_parser.py` fails because of this.

Here is the current implementation:

{target_content}

Here is the failing test:

{test_content}

Produce a unified diff that fixes the bug. Do not change any other behavior.
Wrap the patch between <<<PATCH_START>>> and <<<PATCH_END>>> markers.
"""

    messages = [
        {"role": "system", "content": "You are an expert Python developer. Provide unified diffs."},
        {"role": "user", "content": prompt},
    ]

    print("\nCalling Ollama (this may take a while)...")
    start_time = time.time()
    try:
        raw = agentic_fix_ollama.call_ollama(
            messages=[messages[0], messages[1]],
            models=[model],
            max_retries_per_model=1,
            base_wait=2,
        )
    except Exception as e:
        print(f"LLM call failed: {e}")
        return 1
    elapsed = time.time() - start_time
    print(f"Response time: {elapsed:.1f}s")
    print("\n=== RAW OUTPUT ===")
    print(raw)

    patch = agentic_fix_ollama.extract_patch(raw)
    if not patch.strip():
        print("No patch extracted (model likely returned NO_CHANGES or invalid output)")
        return 2

    patch_path = REPO_ROOT / "temp_benchmark_patch.diff"
    patch_path.write_text(patch)
    print(f"Patch written to {patch_path.name}")

    # Check patch applies
    check = subprocess.run(
        ["git", "apply", "--check", str(patch_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        print(f"Patch check failed: {check.stderr}")
        print(f"Patch file kept at {patch_path}")
        return 3

    # Apply patch
    apply = subprocess.run(
        ["git", "apply", str(patch_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if apply.returncode != 0:
        print(f"Patch apply failed: {apply.stderr}")
        print(f"Patch file kept at {patch_path}")
        return 4

    patch_path.unlink(missing_ok=True)
    print("Patch applied successfully.")

    # Run the specific test
    test_result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_duration_parser.py", "-q", "-o", "addopts="],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    print("Test return code:", test_result.returncode)
    print(test_result.stdout)
    if test_result.returncode != 0:
        print(test_result.stderr)

    # Revert the applied patch to keep repo clean for next run
    target_file.write_text(target_content)
    print("Reverted target file to original state.")

    if test_result.returncode == 0:
        print("\n✅ Patch fixed the failing test.")
        return 0
    else:
        print("\n❌ Patch did not fix the failing test.")
        return 5


if __name__ == "__main__":
    sys.exit(main())
