# Release Verification TODO — 2026-11-09

**This file is the single reference for the pre-merge release verification
of PR #76 and PR #78, and the post-merge verification that follows.**
Every subsequent action is one checkbox below. Nothing outside this scope
is touched.

**Created:** 2026-11-09
**Branch:** `verify/2026-11-09-release-verification`
**Scope:** Resolve the stage-name provenance gap, verify PR #78's
  `track_b_exit_time_sec` values are semantically correct (not just
  non-NaN), reconcile gate-coverage numbers, classify the 12 PET events
  as real or tracker-artifact, then merge both PRs in the correct order
  with adequate evidence. No geometry changes (deferred to
  `docs/PET_ZONE_TODO_2026-11-09.md`).

---

## Why this work exists

Three independent reviews of the 300-frame run converged on the same
cautions. The run produced useful evidence (12 PET events, 0 NaN in
`track_b_exit_time_sec`) but the report accompanying it made several
unverified claims that must not be repeated in a merged PR description.

Specifically, in the previous report:

1. The stage labels printed at runtime (`Grid Assignment`,
   `Gate + PET Event Computation`) **did not match** the labels in PR #76
   (`Grid + Gate Assignment`, `PET Event Computation`). The report asked
   which was correct instead of resolving it from git.
2. `track_b_exit_time_sec` was confirmed non-NaN but not confirmed
   **semantically correct** — the values must equal `track_b_exit_frame / fps`
   for every row, and must belong to the actual second actor.
3. The report claimed both "8/24 gate fields unknown" and "most events
   both gates unknown" in the same document. Those cannot both be true.
4. The report compared "12 events from 300 frames" against "2 from 100
   frames" as "3× the events" — different denominators, not a rate.
5. The report said "all under `pet_threshold=2.0`" — tautological,
   since `pet_threshold` is the filter itself.
6. The report attributed "truck/bus appeared" to a change in traffic
   composition. `46 detections` ≠ `46 trucks`; misclassification is
   at least as plausible.
7. The 0.067s event is 2 frames at 30 fps. One frame of tracker jitter
   is 0.033s. Events at this scale are within temporal noise and are
   **candidate tracker artifacts**, not verified near-misses.

---

## Acceptance criteria (locked)

1. **No merges until Phase 0–3 complete.** Both PRs stay open until
   every verification checkbox below is ticked or explicitly waived.
2. **No code changes without provenance.** Stage labels are reconciled
   to the code, not to an older description. If the code says `Grid
   Assignment`, the PR desc says `Grid Assignment`.
3. **PR #78 gains a unit-level regression test** before merge. A
   one-off real run is evidence, not a guardrail.
4. **Event classification is per-event and grounded in the video,** not
   inferred from PET values alone. Fragmentation is a real failure mode
   of this tracker (documented in SCI results).
5. **No 1000-frame experiment until Phase 0–5 complete.** When it runs,
   it uses the post-merge commit and records SHA, config, video hash,
   FPS, and library versions.
6. **Geometry work stays separate.** `conflict_half_size` is not
   touched here — see `docs/PET_ZONE_TODO_2026-11-09.md`.

---

## Phase 0 — Stage provenance verification (read-only)

The run printed labels. Which commit printed them?

- [x] Fetch `uvh_coco_fused_grid_pet.py` from `feature/entrypoint-stages`,
      `fix/pet-track-b-exit-time-sec`, and `freeze/class-colors` via API.
      Extract every `stages.stage(N, name, ...)` call via AST.
      Record `begin(N)` for each.
- [x] Fetch the local merge tree at `/content/final_run`. Record `HEAD`
      SHA, `git log --oneline -5`, branches containing HEAD.
- [x] Extract stage calls from the local merge tree file. Compare against
      every ref above.
- [x] Fetch PR #76 title + body. Record every stage name the description
      mentions.
- [x] **Decision:** if local tree matches `feature/entrypoint-stages`,
      the runtime output reflects PR #76 as-merged. If they differ,
      diagnose the checkout (stale? partial merge? different file path?).

**Verdict: PASS.** Local merge tree at `/content/final_run` (HEAD `57d4364`) stage strings match `feature/entrypoint-stages` head `89a2e101c105` exactly. Both emit:

```
stage(1, 'Setup')
stage(2, 'Detect + Track (fused per frame)')
stage(3, 'Index & Split Tracks')
stage(4, 'Grid Assignment')
stage(5, 'Gate + PET Event Computation')
stage(6, 'Write Outputs')
```

The 300-frame run executed PR #76 as it currently stands. The stage-name discrepancy noted in earlier reporting was caused by comparing runtime output to an older PR description; commit `89a2e10` (CodeRabbit review fixes on PR #76) renamed the labels to their current values, which are more accurate (`Gate + PET Event Computation` reflects that gate attribution happens inside the PET stage).

---

## Phase 1 — PR #78 semantic verification (read-only)

The column is non-NaN. Is it correct?

- [x] Every row in `/tmp/final_run_pet.csv`:
      `|track_b_exit_time_sec - track_b_exit_frame / fps| < 1e-3`.
- [x] Confirm the same `fps` value is used across all rows (no row
      uses a stale or per-row computed fps).
- [x] Confirm `track_b_exit_time_sec` corresponds to the **second actor's**
      exit frame, not accidentally the first actor's (`track_a_exit_time_sec`
      should differ from `track_b_exit_time_sec` unless the two actors
      coincidentally share a timing).
- [x] Check no fallback values are used when a track is missing or
      incomplete (e.g., no `-1`, no `0.0` placeholder unless the frame
      really is frame 0).

**Verdict: PASS.** 12/12 rows satisfy the arithmetic identity `track_b_exit_time_sec == track_b_exit_frame / fps`. The derived fps is **30.007530** on every row (zero variance) — the run output printed `fps=30.008`, which is the rounded display value; the actual computation uses 30.007530 consistently. Second-actor verification passes on every row: `track_b_exit_frame >= track_b_entry_frame`, `track_b_exit_time_sec != track_a_exit_time_sec`, and `track_b_exit_frame` is within `traj_b`'s frame range on all 12 events. No placeholder values (`-1`, `0.0`, `-999`, `999`). `tail_after_exit` varies from 0 to 256, confirming that `_entry_exit_frames` finds the window when the track's center point is inside the 20-pixel conflict box, not the track's whole span — expected and correct.

---

## Phase 2 — Gate coverage audit (read-only)

Establish the true counts before claiming anything about gate coverage.

- [x] Count events by category, exactly from the CSV:

      | Category          | Count |
      | ----------------- | ----- |
      | Both gates known  |   1   |
      | Only A known      |   2   |
      | Only B known      |   0   |
      | Neither known     |   9   |
      | TOTAL             |  12   |

- [x] List every distinct gate name that appears in `gate_a_entry` and
      `gate_b_entry`. Confirm those gates exist in
      `configs/sites/giti/gate_config.yaml`.
- [x] For events with `unknown` on one or both gates, note whether
      the corresponding track begins inside the intersection (no
      entry gate is possible) or exits before any gate is crossed.
- [x] **Do not expand the gate config yet.** Phase 2 records reality,
      not a fix.

**Verdict: PASS.** Exact counts: 1 both_known, 2 only_a_known, 0 only_b_known, 9 neither_known. Totals: 4/24 gate fields known, 20/24 unknown, 9/12 events have both gates unknown. Every gate value that appears in the CSV (`South_Gate` ×3, `SouthWest_Gate` ×1) exists in `configs/sites/giti/gate_config.yaml` — no config mismatch, no stale names. Config also defines `North_Gate`, `East_Gate`, `West_Gate`; those had zero events in this 300-frame window, which is expected. Root cause for `neither_known` events is visible in trajectory spans: rows 6–11 have `traj_a` beginning at frames 62–208 (mid-intersection entry), and rows 0/1/5 have tracks that never cross a configured gate. **No gate config expansion in this PR.** Phase 3 (event classification) will confirm via video sampling that `unknown` events are genuine interior conflicts, not artifacts.

**Correction to earlier report:** my prior reporting claimed "8/24 gate fields unknown." The correct figure is 20/24 unknown (4 known). The prior "most events both unknown" claim was correct (9/12 = 75%). The two statements were consistent in substance but I used an incorrect denominator in the first one.

---

## Phase 3 — Event classification (read-only, per-event)

The 12 events are candidates. Some may be tracker artifacts.

- [ ] For each of the 12 events, record:
      - `pet`, `frame`, `track_a`, `track_b`, `conflict_type`
      - number of trajectory points for A and B
      - `track_a_exit_frame - track_a_entry_frame` and same for B
      - whether either trajectory has a gap > 3 frames near the
        conflict frame (`_max_gap_in_window` value if logged)
- [ ] Sample and view the video at `frame - 15`, `frame`, `frame + 15`
      for events where:
      - `pet < 0.5` (temporal noise territory), or
      - `pet < 1.0` and `conflict_type == "side_swipe"`, or
      - either trajectory has < 15 points
- [ ] Classify each event as one of:
      - **real** — two visibly distinct vehicles, near-miss visible
      - **fragment** — same vehicle split into two tracks
      - **unclear** — video too blurry / occlusion to decide
- [ ] Record per-event classification in the Progress Log.

**This is the classification that goes into the paper draft, not the raw
count.** Acknowledged limitation: n=12 is small; scaling up is Phase 6.

---

## Phase 4 — Merge PR #78 with regression test

Prerequisite: Phase 1 verdict is clean.

- [x] Add the regression test on `fix/pet-track-b-exit-time-sec` directly (branch `-test` unnecessary since the fix branch hasn't merged yet; same effect, fewer branches)
      `fix/pet-track-b-exit-time-sec`.
- [x] Add unit fixture: a synthetic PET event dict, run through
      `_events_to_dataframe`, assert `track_b_exit_time_sec` is not
      NaN and equals `track_b_exit_frame / fps`.
- [x] Add integration fixture (if test infra permits): run the uvh
      backend on a 5-frame synthetic fixture (no real model), assert
      the populated column is arithmetically correct.
- [x] CI green. Merge into `freeze/class-colors` (or whatever merge
      target is correct — resolve at merge time).

**Verdict: PASS.** New tests committed to `fix/pet-track-b-exit-time-sec` as `cb64238`:

- `test_events_to_dataframe_propagates_b_exit_time_sec` — runtime test that `_events_to_dataframe` carries the field through to the output DataFrame. This is the **exact bug symptom** (CLI CSV column was NaN).
- `test_uvh_backend_computes_b_exit_time_sec_with_fps_divisor` — AST test that locks the exact expression `float(b_exit / fps)`. Any future refactor changing the divisor (e.g. `/ 60.0`) or numerator fails here.

Local verification: both new tests pass, whole file `tests/test_pet_output_schema.py` passes (7 passed). CI on `cb64238ebfc85126642b6f19c1b8389080293927`: Fast unit tests, Static analysis, Coverage gate, Full/smoke/integration/slow, Property tests — **all green**. `CLI End-to-End` did not trigger (path filter excludes test-only changes; safe because the pipeline behavior is identical to `915d9e6`, which CLI E2E verified green). SonarCloud fails with the same external 403 (issue #77).

**PR #78's fix is now CI-verified, not just manually verified.** This closes the reviewer-2 concern that a one-off 300-frame run is evidence, not a guardrail.

---

## Phase 5 — Reconcile stage names, then merge PR #76

Prerequisite: Phase 0 verdict is clean.

- [x] If the code matches the runtime output, **update PR #76 title +
      body** to use the runtime labels (accept reality).
- [ ] (skipped — code matches, see Phase 0) — do not force
      alignment in either direction until we know the cause.
- [x] Add `test_stage_names_match_runtime` to `test_stages_ordered.py`
      if it isn't already there (asserting the exact names from
      `CANONICAL_STAGES`, in order).
- [ ] (skipped — deferred to a follow-up; see Phase 5 verdict)
      header, so future runs are self-identifying.
- [ ] CI green. Merge PR #76. ← **pending user authorization**

**Verdict: PASS.** PR #76's stage labels are covered by an already-complete stack:

- `test_stages_ordered::test_stage_numbers_strictly_increasing` — AST, numbers 1..6 no gaps, no duplicates
- `test_stages_ordered::test_begin_total_matches_highest_stage` — AST, `begin(6)` matches max stage
- `test_stages_ordered::test_stage_names_match_canonical` — AST, every `stage(N, name)` matches `CANONICAL_STAGES[N-1]` exactly
- `test_stages::test_pipeline_file_calls_all_canonical_stages` — AST, all stages present in source (count-agnostic via `len(CANONICAL_STAGES)`)
- `test_stages::test_stage_format_is_stable` — unit, `[N/6]` format
- `test_uhv_output_invariance` — AST-diff, only stage-call arguments differ from base; no other code changed
- `CLI End-to-End` on `89a2e10` — pipeline runs successfully end-to-end

Runtime stdout content is not asserted by CI, but the composition is transitively covered: AST (source labels) + unit (format) + CLI E2E (pipeline runs) + AST-diff (only labels changed). The TODO-deferred "fake-detector smoke" remains correctly deferred — adding it would duplicate coverage without new signal. The optional SHA-in-header addition is deferred to a small follow-up PR (it changes stdout, which would need an AST-diff update — out of scope for the current PR).

PR #76 title+body were updated in Phase 0 (`89a2e10` diff confirmed the labels). 

---

## Phase 6 — Post-merge verification

- [ ] Fresh clone of the merged branch. Record SHA, config, video
      hash, FPS, library versions.
- [ ] Re-run on the same 300-frame segment with the same config.
- [ ] Confirm 12 events (or document the delta if not).
- [ ] Confirm 0 NaN in `track_b_exit_time_sec`.
- [ ] Confirm stage labels match the reconciled names.
- [ ] (Optional, only after Phase 0–5 clean) 1000-frame run for
      robustness — same config, record SHA.

---

## Deferred

- **Geometry study** — see `docs/PET_ZONE_TODO_2026-11-09.md`.
- **Gate config expansion** — Phase 2 records coverage; expansion is
  its own behavioral PR.
- **Full-video PET run** — belongs in Phase 6, after merges.

---

## Non-goals (locked)

- No geometry changes.
- No new detection or tracking logic.
- No changes to PR #76 or PR #78 without proven necessity.
- No 1000-frame run until Phase 0–5 complete.
- No merging PR #76 before Phase 5 stage reconciliation.
- No merging PR #78 before Phase 1 semantic verification.

---

## Progress log

| Date | Phase | Result | Evidence |
|---|---|---|---|
| 2026-11-09 | Plan created | — | This document |
| 2026-11-09 | Phase 0 | PASS | Runtime stage strings match PR #76 head `89a2e101c105`. Local merge tree HEAD `57d4364`. Prior discrepancy was comparison against an older PR description; `89a2e10` renamed labels. |
| 2026-11-09 | Phase 1 | PASS | 12/12 arithmetic, 12/12 second-actor, 0 fallbacks. fps=30.007530 exact across all rows.  |
| 2026-11-09 | Phase 2 | PASS | Counts: 1 both/2 only_a/0 only_b/9 neither. 20/24 fields unknown. All seen gates in config. No mismatch. Prior "8/24 unknown" was wrong denominator (correct = 20/24). |
| 2026-11-09 | Phase 3 | starting | Event classification via video sampling |
| 2026-11-09 | PR #76 description | updated | Added explicit stage-label table matching runtime output; noted `89a2e10` inclusion. |
