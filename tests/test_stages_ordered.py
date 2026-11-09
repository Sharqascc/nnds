"""Static check: stage numbering in the three grid-PET backends.

Phase 1 of docs/ENTRYPOINT_TODO.md. Fast lane: pure AST, no module imports,
no fixtures. Asserts the *current* state is internally consistent — stage
numbers appear in source order, strictly increasing, and begin(N) agrees
with max(stage(N)). Does not (yet) assert the count is 6 — that check
arrives with the uvh relabel, once all three backends use CANONICAL_STAGES.

Backends that do not yet emit stages skip rather than fail, so this test
can land *before* the relabel without blocking the branch.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

BACKENDS = (
    "src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py",
    # yolo_cpu_grid_pet.py and sam3_grid_pet.py are deferred per
    # docs/ENTRYPOINT_TODO.md Phase 3. Their stage labeling is out of
    # scope for this PR. When their relabel work begins, add each path
    # back here — the parametrized tests below already cover them.
)


def _parse(path: str) -> ast.AST:
    return ast.parse((REPO / path).read_text())


def _stage_numbers(tree: ast.AST) -> list[tuple[int, int]]:
    """(lineno, N) for every stages.stage(N, ...) call, in source order."""
    out: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "stage"):
            continue
        if not (isinstance(f.value, ast.Name) and f.value.id == "stages"):
            continue
        if not node.args:
            continue
        n = node.args[0]
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            out.append((node.lineno, n.value))
    out.sort()
    return out


def _begin_total(tree: ast.AST) -> int | None:
    """First positional int argument to stages.begin(N, ...), if any."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "begin"):
            continue
        if not (isinstance(f.value, ast.Name) and f.value.id == "stages"):
            continue
        if node.args and isinstance(node.args[0], ast.Constant):
            v = node.args[0].value
            if isinstance(v, int):
                return v
    return None


@pytest.mark.parametrize("path", BACKENDS)
def test_stage_numbers_strictly_increasing(path: str):
    """stage(N) calls appear in source order, strictly increasing, no gaps."""
    nums = [n for _, n in _stage_numbers(_parse(path))]
    if not nums:
        pytest.skip(f"{path} does not yet emit stages — see TODO Phase 1/3")
    assert nums == sorted(nums), f"{path}: stage numbers out of order: {nums}"
    assert len(nums) == len(set(nums)), f"{path}: duplicate stage numbers: {nums}"
    assert nums == list(range(1, len(nums) + 1)), (
        f"{path}: stage numbers must be 1..{len(nums)} with no gaps, got {nums}"
    )


@pytest.mark.parametrize("path", BACKENDS)
def test_begin_total_matches_highest_stage(path: str):
    """begin(N) must agree with max(stage(N)) — else [n/total] is wrong."""
    tree = _parse(path)
    nums = [n for _, n in _stage_numbers(tree)]
    if not nums:
        pytest.skip(f"{path} does not yet emit stages — see TODO Phase 1/3")
    total = _begin_total(tree)
    assert total is not None, f"{path}: begin(...) call missing"
    assert total == max(nums), (
        f"{path}: begin({total}) but highest stage is {max(nums)}; "
        f"[n/total] denominator would be wrong"
    )


def _stage_calls(tree: ast.AST) -> list[tuple[int, int, str]]:
    """(lineno, N, name) for every stages.stage(N, name, ...) call."""
    out: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "stage"):
            continue
        if not (isinstance(f.value, ast.Name) and f.value.id == "stages"):
            continue
        if len(node.args) < 2:
            continue
        n, name = node.args[0], node.args[1]
        if (
            isinstance(n, ast.Constant)
            and isinstance(n.value, int)
            and isinstance(name, ast.Constant)
            and isinstance(name.value, str)
        ):
            out.append((node.lineno, n.value, name.value))
    out.sort()
    return out


@pytest.mark.parametrize("path", BACKENDS)
def test_stage_names_match_canonical(path: str):
    """Every stage(N, name, ...) call uses CANONICAL_STAGES[N-1] as its name."""
    from src.analysis._stages import CANONICAL_STAGES

    calls = _stage_calls(_parse(path))
    if not calls:
        pytest.skip(f"{path} does not yet emit stages — see TODO Phase 1/3")
    for lineno, n, name in calls:
        expected = CANONICAL_STAGES[n - 1]
        assert name == expected, (
            f"{path}:{lineno} stage({n}, {name!r}, ...) does not match "
            f"CANONICAL_STAGES[{n - 1}] = {expected!r}"
        )
