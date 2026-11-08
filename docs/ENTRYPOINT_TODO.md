# Entry-Point TODO

**This file is the single reference for the entry-point observability work.** Every subsequent change is one checkbox below. Nothing outside this scope is touched.

**Created:** 2026-10-31
**Branch:** `feature/pipeline-to-metric-schemas`
**Scope:** `src/analysis/_stages.py`, uvh grid-PET backend, entry-point tests, entry-point pre-push gate. (yolo/sam3 backends deferred — see Phase 3.)
**MEP (main entry point):** `src/pipeline/traffic_analyzer.py` (console script `nnds`). `scripts/run_pipeline.py` (`nnds-run`) is a preflight/dispatch wrapper around it, not the MEP.

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
| 4 | Grid Assignment |
| 5 | Gate + PET Event Computation |
| 6 | Write Outputs |

### Work items (PR 1 — one branch, one PR)

- [x] **Canonical stage list in `src/analysis/_stages.py`.** `CANONICAL_STAGES` tuple added in commit `307c1523`. (Two originally-scoped sub-items deferred to Phase 3: the `begin(total, header)` signature rewrite — kept backward-compatible because tests pass `total` positionally — and the per-backend `stage_set` — no longer needed while uvh is the only in-scope backend.)

- [x] **Relabel `uvh_coco_fused_grid_pet.py`.** 6 stage calls, sequentially numbered. Existing `stages.stage(N, ...)` calls updated to the new API. No logic around them changes. Done in commits `efd48449` + `d10ef12c` (6/6 stages match CANONICAL_STAGES).

- [ ] **Relabel `yolo_cpu_grid_pet.py`.** Deferred — see Phase 3.

- [ ] **Relabel `sam3_grid_pet.py`.** Deferred — see Phase 3.

- [x] **`tests/test_stages_ordered.py`.** Static check done (commits `ab86d3d3` + `d10ef12c`).
- [ ] **Fake-detector smoke** (medium): run `uvh-coco-fused` with a mock detector on a 5-frame fixture, capture stdout, assert `[1/6]` … `[6/6]` appear in order. Deferred — the AST-diff output-invariance check below gives strictly stronger assurance.
- [x] **Output invariance** (the critical one): AST-diff between base and head proves the uvh backend differs only in stage-call statements, and `_stages.py` function bodies are unchanged. See `tests/test_uhv_output_invariance.py` (commit `3b796e19`). Opt-in via `NNDS_INVARIANCE_BASE=<sha>` so CI stays fast. Strictly stronger than a single-fixture pipeline run: proves the invariant for *all* inputs, immune to detector numerical non-determinism.

- [x] **Grep for old `/5` references.** Search entire repo (`src/`, `scripts/`, `tests/`, `docs/`) for `[1/5]` through `[5/5]` strings and update to the new `/6` form. Prevent docs/tests from asserting against stale numbers. Done in commit `c3b6cc18` (the only remaining `[1/5]`/`[5/5]` strings are inside this TODO itself, in the description of this very task).

- [ ] **Local pre-push gate (manual, before opening PR):** `compileall` + `ruff check` + `ruff format --check` + `bash -n` + `yaml.safe_load` + the three entry-point test files (`test_stages_ordered.py`, `test_cli_contract.py`, `test_entrypoint_wrapper_fixes.py`) **plus** the stage-related test files that assert on the real backend (`test_stages.py`, `test_stages_progress.py`). **Refuse to push if any fails.**

  > **2026-11-07 lesson:** the initial gate omitted `test_stages.py`, whose `test_pipeline_file_calls_five_stages` hardcoded `range(1, 6)`. The uvh relabel legitimately added `stage(6, ...)`; CI caught the resulting failure (SonarCloud + two CI runs) but the local gate did not. Fixed in commit `4c43a253` by making the test count-agnostic via `CANONICAL_STAGES`. The lesson is not just "add the file to the gate" — it is "gate the tests that hardcode the thing you are changing."

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

- [ ] **Relabel `yolo_cpu_grid_pet.py`.** ~3–4 truthful stages. Inspect its structure first.
- [ ] **Relabel `sam3_grid_pet.py`.** Same treatment as `yolo_cpu_grid_pet.py`.

- [ ] **Decouple detection from tracking** (reviewer 2's proposal). Requires a golden-master test: run fused vs. decoupled on the same fixture, compare outputs exactly. Large, multi-day. Only if the pipeline owner asks.
- [ ] **`--visualize` flag** (Option A's "separate feature later"). Renders grid + gates on output video. Separately testable.
- [ ] **Resolve PR #49 threads and merge the release** (blocked on the 89-thread cleanup + `KNOWN_ISSUES.md`, independent of this TODO).

---

## Non-goals (locked)

- No changes to `paper/analysis/`, `src/diffusion/`, `src/vlm/`, `baselines/`, or `tests/` other than entry-point tests.
- No new third-party dependencies.
- No new CLI flags in Phase 1.
- No changes to the tracker, PET math, gate geometry, or metrics.
- No visualization inside the numeric stage markers. The `--out-video` render is a
  post-pipeline step that runs after the CSV is written; it does not touch
  `stages.stage(...)` calls or the backend's numbering.

---

## Progress log

_Update this section as items are completed. One line per merged PR._

| Date | Item | PR | Notes |
|---|---|---|---|
| 2026-10-31 | TODO created | — | Stage count = 6 (verified against source) |
| 2026-11-04 | `--out-video` flag + render module | (this PR) | Adds `--out-video PATH`; renders grid/gates/tracks/PET over the run's video. Post-pipeline step; no backend stage labels changed. |
| 2026-11-07 | Phase 1: 6-stage relabel + CANONICAL_STAGES + MEP designation | feature/entrypoint-stages | uvh now emits 6/6 stages matching `CANONICAL_STAGES`; output invariance proven by AST-diff (`3b796e19`), zero non-stage AST change vs base. MEP formalized as `src/pipeline/traffic_analyzer.py` with guard test (`585a77e0`). Pre-push gate passes. yolo/sam3 relabel deferred to Phase 3. |
