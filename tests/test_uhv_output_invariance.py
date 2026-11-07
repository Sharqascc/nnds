"""Output-invariance check for the Phase 1 stage relabel.

docs/ENTRYPOINT_TODO.md locks this acceptance criterion:

    "PET events, CSVs, and metrics must be identical before and after
     the [stage relabel]."

This test proves it by AST-diff: strips every stages.stage()/begin()/
progress()/done() call from the base and head versions of the uvh
backend, then asserts the remaining AST is identical. If it is, the
only things that changed are progress-trace calls — which print to
stdout and cannot affect CSV output.

Stronger than running the pipeline twice: the AST-diff proves the
invariant for every input, not just one fixture, and is immune to
CPU-vs-GPU numerical non-determinism in the detectors.

Opt-in because it needs the base commit reachable in git history.
CI shallow clones (--depth=1) won't have it. Run before opening the
PR:

    NNDS_INVARIANCE_BASE=591e5c0c5f008342451da661a439738438a8601d \
        pytest tests/test_uhv_output_invariance.py -v
"""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

BASE_SHA = os.environ.get("NNDS_INVARIANCE_BASE")

UVH_PATH = "src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py"
STAGES_PATH = "src/analysis/_stages.py"

STAGE_FUNCS = {"stage", "begin", "progress", "done"}
STAGE_FUNC_BODIES = {"begin", "stage", "progress", "done"}


def _is_stage_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in STAGE_FUNCS
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "stages"
    )


class _StripStageCalls(ast.NodeTransformer):
    """Remove stage-related expression statements; neutralize nested calls."""

    def visit_Expr(self, node: ast.Expr):
        if _is_stage_call(node.value):
            return None  # remove the whole statement
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if _is_stage_call(node):
            return ast.Constant(value=None)
        return node


def _strip_stage_calls(tree: ast.AST) -> ast.AST:
    tree = _StripStageCalls().visit(tree)
    ast.fix_missing_locations(tree)
    return tree


def _normalized_source(body: str) -> str:
    return ast.unparse(_strip_stage_calls(ast.parse(body)))


def _git_show(ref: str, path: str) -> str:
    r = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    if r.returncode != 0:
        pytest.skip(f"git show {ref}:{path} failed (shallow clone?): {r.stderr.strip()[:200]}")
    return r.stdout


def _assert_equal(a: str, b: str, label: str) -> None:
    if a == b:
        return
    a_lines, b_lines = a.splitlines(), b.splitlines()
    for i, (la, lb) in enumerate(zip(a_lines, b_lines, strict=False)):
        if la != lb:
            raise AssertionError(
                f"{label}: normalized AST differs at line {i + 1}:\n  base: {la!r}\n  head: {lb!r}"
            )
    raise AssertionError(
        f"{label}: normalized AST differs in length "
        f"(base {len(a_lines)} lines vs head {len(b_lines)})"
    )


@pytest.mark.skipif(
    not BASE_SHA,
    reason="set NNDS_INVARIANCE_BASE=<sha> to enable (requires base commit in history)",
)
def test_uvh_backend_differs_only_in_stage_calls():
    """uvh source changed only in stages.stage/begin/progress/done() calls."""
    base = _git_show(BASE_SHA, UVH_PATH)
    head = (REPO / UVH_PATH).read_text()
    _assert_equal(
        _normalized_source(base),
        _normalized_source(head),
        UVH_PATH,
    )


@pytest.mark.skipif(
    not BASE_SHA,
    reason="set NNDS_INVARIANCE_BASE=<sha> to enable (requires base commit in history)",
)
def test_stages_helper_function_bodies_unchanged():
    """begin/stage/progress/done implementations identical to base."""
    base_tree = ast.parse(_git_show(BASE_SHA, STAGES_PATH))
    head_tree = ast.parse((REPO / STAGES_PATH).read_text())

    def _bodies(tree: ast.AST) -> dict[str, str]:
        return {
            node.name: ast.unparse(node)
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in STAGE_FUNC_BODIES
        }

    base_bodies = _bodies(base_tree)
    head_bodies = _bodies(head_tree)

    assert set(base_bodies) == set(head_bodies) == STAGE_FUNC_BODIES, (
        f"stages.py function set changed: base={sorted(base_bodies)} head={sorted(head_bodies)}"
    )
    for name in STAGE_FUNC_BODIES:
        _assert_equal(base_bodies[name], head_bodies[name], f"stages.{name}")
