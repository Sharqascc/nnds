# PET Conflict-Zone Geometry TODO — 2026-11-09

**This file is the single reference for the PET conflict-zone geometry
audit and any resulting default change.** Every subsequent change is
one checkbox below. Nothing outside this scope is touched.

**Created:** 2026-11-09
**Branch:** `feature/pet-zone-geometry`
**Scope:** `conflict_half_size` geometry only — its unit, its
  configurability, its measurement against labels, and any resulting
  change of default, unit, or class-awareness. No change to the PET
  formula, tracker, gate geometry, or the outputs of PRs #76 / #78.

---

## Why this work exists

During the Phase-1 real-run verification (PR #76), a PET event was
observed whose proximity was diluted by the fixed-pixel conflict zone
(`conflict_half_size = 20.0`). Three independent reviews converged on
the cautions below, all encoded into the acceptance criteria.

1. **Sensitivity ≠ correctness.** Counting events at four sizes tells
   you how sensitive the output is. Only labels tell you which size
   is correct.
2. **Smaller is not monotone.** Shrinking the zone generally increases
   PET for the same pair (first actor exits earlier, second enters
   later) and can silently drop real conflicts when the zone becomes
   smaller than the vehicle footprint. Verify event-by-event.
3. **Fixed pixels ≠ physical distance.** Perspective makes 20 px mean
   different metres near and far from the camera. If a homography
   exists, metres are the principled unit.
4. **Mixed traffic ⇒ class-aware.** Motorcycle, auto-rickshaw, car,
   and bus/truck footprints differ by 2×–10×. One size cannot be
   accurate for all.
5. **Principled bound** for point-vs-square occupancy is roughly the
   sum of the two vehicles' half-extents plus tracker noise — not
   "half the widest vehicle". Treat any rule as a starting range.
6. **Report sensitivity regardless.** A single tuned value is weak
   evidence; a table of event count and F1 by zone size pre-empts
   the reviewer question "how arbitrary is 20?".

---

## Acceptance criteria (locked)

1. **PR #76 and PR #78 remain untouched.** Their invariants (output
   identity, bug-fix isolation) are preserved.
2. **Diagnostic precedes code.** No branch, no PR, no edit to the
   production geometry until Phase 0 establishes what `20.0`
   represents and whether `docs/pet_gate_v1.md` justifies it.
3. **Configurability is output-neutral.** Exposing `conflict_half_size`
   as config with the default left at `20.0` must produce identical
   outputs. A test pins the default; a second pins that the configured
   value is used.
4. **No default change without held-out evidence.** A new default
   (or a move to metres, or class-aware sizing) lands only if a
   precision/recall/F1 sweep against human annotations shows a
   clear held-out improvement, with the sensitivity table committed
   as evidence.
5. **All experiments reproducible.** Record git SHA, config, input
   video hash, FPS, and library versions for every run.

---

## Phase 0 — Read-only diagnostic

Answers one question: **does the codebase justify `20.0`, and in what
unit?**

- [ ] Grep for every reference to `conflict_half_size`, `half_size`,
      `_entry_exit_frames`, and `20.0` near "conflict" or "zone".
- [ ] Read `docs/pet_gate_v1.md`. If it justifies `20.0`, record the
      justification. If it does not, `20.0` is an untested default.
- [ ] Determine the coordinate system: pixel plane, BEV plane, or
      metres? Inspect `_entry_exit_frames` and `_point_in_square`.
- [ ] Check for a homography / BEV mapper that permits a metre-based
      zone (`src/bev/bev_mapper.py`, `grid_config`, `bev_config`).
- [ ] Locate the 80-clip human annotation workbook (if present).
      Record its path and schema.
- [ ] List every test that pins `conflict_half_size` or asserts on
      PET event counts.

**Outcome recorded in the Progress Log below.**

---

## Phase 1 — Configurability (PR A, output-neutral)

- [ ] Expose `conflict_half_size` as config (file and/or CLI flag,
      following the repo's existing patterns).
- [ ] Default remains `20.0`. No behavioural change.
- [ ] Test: default is `20.0`.
- [ ] Test: configured value flows to `_entry_exit_frames`.
- [ ] Test: outputs with default config are identical to today's.
- [ ] Open PR A. Wait for CI. Merge.

---

## Phase 2 — Controlled sensitivity sweep (experiment, not code)

Prerequisites: Phase 1 merged; Phase 0 identified labels.

- [ ] Sweep `conflict_half_size` over `{5, 10, 15, 20, 30, 40}` on the
      labelled clips.
- [ ] Match events across sizes by **actor-ID pair + conflict-zone
      identity + temporal overlap**. Never by PET value.
- [ ] Produce this comparison table:

      | Measurement                  | 5  | 10 | 15 | 20 | 30 | 40 |
      | ---------------------------- | -- | -- | -- | -- | -- | -- |
      | PET events detected          |    |    |    |    |    |    |
      | Events matched to baseline   |    |    |    |    |    |    |
      | Events lost vs baseline      |    |    |    |    |    |    |
      | New events vs baseline       |    |    |    |    |    |    |
      | Median PET (s)               |    |    |    |    |    |    |
      | Mean PET (s)                 |    |    |    |    |    |    |
      | PET ≤ 0.7 s                  |    |    |    |    |    |    |
      | PET ≤ 1.0 s                  |    |    |    |    |    |    |
      | Tracks with no zone crossing |    |    |    |    |    |    |
      | Invalid / incomplete events  |    |    |    |    |    |    |
      | Precision vs labels          |    |    |    |    |    |    |
      | Recall vs labels             |    |    |    |    |    |    |
      | F1 vs labels                 |    |    |    |    |    |    |

- [ ] Split into dev / held-out clips. Select on dev; report on
      held-out.
- [ ] Break down results by vehicle class where the labels permit.
- [ ] Record SHA, config, video hash, FPS, library versions per run.

---

## Phase 3 — Independent validation and decision

- [ ] If one size beats `20.0` on held-out, proceed to Phase 4 with
      that size.
- [ ] If no single size beats `20.0`, keep the default and commit the
      sweep as paper evidence.
- [ ] If class-aware sizing beats a single size, proceed to Phase 4
      with class-aware sizing.
- [ ] If a metre-based zone (via homography) beats pixels, proceed to
      Phase 4 with metres.

---

## Phase 4 — Behavioural change (PR B, explicit "outputs will change")

Lands only with held-out improvement evidence.

- [ ] Apply the chosen geometry (size, unit, class-awareness).
- [ ] Update `docs/pet_gate_v1.md` with the sweep results and the
      rationale for the new value.
- [ ] Add the sensitivity table to the paper draft.
- [ ] Open PR B with an explicit "outputs will change" note. Cite
      the held-out F1 delta and the sweep table.
- [ ] Wait for CI. Wait for reviewer. Merge only after
      behavioural-change approval.

---

## Non-goals (locked)

- No change to the PET formula (`(b_entry - a_exit) / fps`).
- No change to the tracker, gate geometry, or `pet_threshold`
  semantics.
- No change to `_events_to_dataframe`'s schema beyond what PR #78
  already fixed.
- No bundling of geometry changes with PRs #76 or #78.
- No "smaller = better" heuristic adopted without labels.

---

## Progress log

_One line per completed item or merged PR._

| Date | Item | PR / commit | Notes |
|---|---|---|---|
| 2026-11-09 | TODO created | — | Scope = geometry only. Diagnostic is Phase 0. |
