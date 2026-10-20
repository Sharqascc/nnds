"""Guard against accidentally committing model weights."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

WEIGHT_EXTS = (
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".engine",
    ".tflite",
    ".safetensors",
    ".h5",
    ".mlmodel",
    ".gguf",
    ".msgpack",
    ".bin",
    ".pkl",
)

MAX_BYTES = 1024 * 1024

ALLOWLIST: frozenset[str] = frozenset(
    {
        # (No exceptions today. Add entries here only with a comment.)
    }
)


def _git_ls_tree() -> list[tuple[str, int]]:
    """Return [(path, blob_bytes)] for every file tracked by HEAD."""
    r = subprocess.run(
        ["git", "ls-tree", "-r", "-l", "HEAD"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        pytest.skip("not a git checkout; cannot run commit-hygiene test")
    entries: list[tuple[str, int]] = []
    for line in r.stdout.splitlines():
        meta, _, path = line.partition("\t")
        fields = meta.split()
        if len(fields) < 4:
            continue
        try:
            size = int(fields[3])
        except ValueError:
            continue
        entries.append((path, size))
    return entries


@pytest.mark.skipif(not (REPO / ".git").exists(), reason="requires a git checkout")
def test_no_model_weights_committed() -> None:
    offenders: list[tuple[str, int]] = []
    for path, size in _git_ls_tree():
        if path in ALLOWLIST:
            continue
        if not path.lower().endswith(WEIGHT_EXTS):
            continue
        if size > MAX_BYTES:
            offenders.append((path, size))
    if offenders:
        lines = [
            f"  {p}  ({s / 1024 / 1024:.1f} MB)" for p, s in sorted(offenders, key=lambda x: -x[1])
        ]
        raise AssertionError(
            "Model weights committed to git:\n"
            + "\n".join(lines)
            + "\n\nThe repo does not store model weights. Remove the file "
            "with `git rm --cached <path>`, add it to .gitignore, or route it "
            "through Git LFS and add it to ALLOWLIST in this test. "
            "See docs/MODEL_WEIGHTS.md."
        )


@pytest.mark.skipif(not (REPO / ".git").exists(), reason="requires a git checkout")
def test_gitignore_covers_common_weight_extensions() -> None:
    gitignore = (REPO / ".gitignore").read_text()
    required = ("*.pt", "*.pth", "*.onnx", "*.engine", "*.safetensors", "*.ckpt")
    missing = [ext for ext in required if ext not in gitignore]
    assert not missing, f".gitignore is missing weight extensions: {missing}"
