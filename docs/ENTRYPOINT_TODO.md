# Entry-Point TODO

**This file is the single reference for the entry-point observability work.** Every subsequent change is one checkbox below. Nothing outside this scope is touched.

**Created:** 2026-10-31
**Branch:** `feature/pipeline-to-metric-schemas`
**Scope:** `src/analysis/_stages.py`, three grid-PET backends, entry-point tests, entry-point pre-push gate.

---

## Acceptance criteria (locked)

Every change below must satisfy:

1. **No change to pipeline outputs.** PET events, CSVs, and metrics must be identical before and after.
2. **No core logic changes.** Kalman filter, spatial math, gate geometry untouched.
3. **Every push passes the local pre-push gate** (Phase 2) before reaching GitHub.

The reviewer feedback that locked this in:

> "The most important acceptance criterion is that the new progress reporting and testing infrastructure must not change the scientific outputs of the existing pipeline."

---

## Phase 1 — Stage relabeling

**Chosen layout (6 stages)** — determined by inspecting `uvh_coco_fused_grid_pet.py` (grid assignment happens at lines using `track_cells[...]`/`get_cell_from_pixels`; gate crossing happens at lines calling `_get_entry_gate`/reading `gate_a_entry`/`gate_b_entry`; see investigation log).

6 stages: gate crossing is fused with PET computation (measured honestly)

| # | Label |
|---|---|
| 1 | Setup |
| 2 | Detect + Track (fused per frame) |
| 3 | Index & Split Tracks |
| 4 | Grid + Gate Assignment |
| 5 | PET Event Computation |
| 6 | Write Outputs |

### Work items (PR 1 — one branch, one PR)

- [ ] **Canonical stage list in `src/analysis/_stages.py`.**
  Replace the free-form `begin(total, header)` signature with a canonical ordered tuple of stage names. Add a per-backend `stage_set` so each backend declares which stages it actually performs. This prevents the string drift that made `yolo-cpu` and `sam3` silent in the first place.

- [ ] **Relabel `uvh_coco_fused_grid_pet.py`.** 6 stage calls, sequentially numbered. Existing `stages.stage(N, ...)` calls updated to the new API. No logic around them changes.

- [ ] **Relabel `yolo_cpu_grid_pet.py`.** Only the stages it actually performs. Inspect its structure first, add only the truthful stage boundaries. Probably 3–4 of the 6.

- [ ] **Relabel `sam3_grid_pet.py`.** Same treatment as `yolo_cpu_grid_pet.py`.

- [ ] **`tests/test_stages_ordered.py`.** Three classes of test:
  - **Static check** (fast): parse each backend's source with `ast`, assert `stage(n, ...)` calls appear in strictly increasing order with no duplicates and no gaps.
  - **Fake-detector smoke** (medium): run `uvh-coco-fused` with a mock detector on a 5-frame fixture, capture stdout, assert `[1/6]` … `[6/6]` appear in order.
  - **Output invariance** (the critical one): run `uvh-coco-fused` twice on the same fixture — once from the merge base commit (before relabeling) and once from the PR head — assert the emitted PET CSV files are byte-identical (excluding any timestamp columns).

- [ ] **Grep for old `/5` references.** Search entire repo (`src/`, `scripts/`, `tests/`, `docs/`) for `[1/5]` through `[5/5]` strings and update to the new `/6` form. Prevent docs/tests from asserting against stale numbers.

- [ ] **Local pre-push gate (manual, before opening PR):** `compileall` + `ruff check` + `ruff format --check` + `bash -n` + `yaml.safe_load` + the three entry-point test files (`test_stages_ordered.py`, `test_cli_contract.py`, `test_entrypoint_wrapper_fixes.py`). **Refuse to push if any fails.**

- [ ] **Open PR, wait for CI green, merge.**

---

## Phase 2 — Pre-push gate consolidation (separate PR)

Reviewer 2's rationale, honored:

> "A `.githooks/pre-push` file does nothing until each clone runs `git config core.hooksPath .githooks` ... It also doesn't survive `--no-verify`. ... Keep them apart."

This phase is its own PR. It is **not** bundled with Phase 1.

### Work items (PR 2)

- [ ] **Write `scripts/pre_push_checks.sh`** — one source of truth. Fails fast on first error. Runs:
  1. `python -m compileall -q src tests scripts conftest.py`
  2. `ruff check` and `ruff format --check` on the same paths CI uses (respect repo config; do not expand scope)
  3. `bash -n` on every `scripts/*.sh` and `.githooks/*`
  4. `yaml.safe_load` on every `.github/workflows/*.yml`
  5. Targeted `pytest` on entry-point tests only — not the full suite
- [ ] **Wire into `.githooks/pre-push`** — invoke `scripts/pre_push_checks.sh`, exit 1 on failure.
- [ ] **Wire into `.github/workflows/ci.yml`** — invoke the same script in the static lane. **One script, two callers, no duplication.**
- [ ] **Document hook activation** in `CONTRIBUTING.md` or `docs/WORKFLOW.md`: `git config core.hooksPath .githooks`.
- [ ] **Verify the gate actually blocks.** Stage a commit that intentionally breaks ruff (e.g. an unused import). Confirm `git push` is refused with a message pointing at the script.
- [ ] **Open PR, wait for CI green, merge.**

---

## Phase 3 — Deferred (not part of this work)

Listed for reference only. **Do not work on these without explicit go-ahead.**

- [ ] **Decouple detection from tracking** (reviewer 2's proposal). Requires a golden-master test: run fused vs. decoupled on the same fixture, compare outputs exactly. Large, multi-day. Only if the pipeline owner asks.
- [ ] **`--visualize` flag** (Option A's "separate feature later"). Renders grid + gates on output video. Separately testable.
- [ ] **Resolve PR #49 threads and merge the release** (blocked on the 89-thread cleanup + `KNOWN_ISSUES.md`, independent of this TODO).

---

## Non-goals (locked)

- No changes to `paper/analysis/`, `src/diffusion/`, `src/vlm/`, `baselines/`, or `tests/` other than entry-point tests.
- No new third-party dependencies.
- No new CLI flags in Phase 1.
- No changes to the tracker, PET math, gate geometry, or metrics.
- No visualization in the pipeline (it is Option A: separate feature later).

---

## Progress log

_Update this section as items are completed. One line per merged PR._

| Date | Item | PR | Notes |
|---|---|---|---|
| 2026-10-31 | TODO created | — | Stage count = 6 (verified against source) |
