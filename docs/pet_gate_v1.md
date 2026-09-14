# PET gate — first measured improvement

## Change

Added `_track_missing_ratio()` and a `max_missing_ratio` knob
(default `0.10`) to `run_uvh_coco_fused_grid_pet`. The gate skips
track pairs where either participant has more than 10% of its own
lifespan missing detections.

Commit: `feat(pet): add tracking-quality gate to reduce false PET events`
Branch: `fix/mypy-python-version`

## Measurement (same 300-frame GITI eval, same 53-event review)

| Metric      | Baseline | Gated   |
|-------------|----------|---------|
| events      | 53       | 26      |
| Y (real)    | 14       | 13      |
| N (false)   | 39       | 13      |
| precision   | 0.264    | **0.500** |
| recall      | 1.000    | 0.929   |

- 26 of the 53 baseline events survived the gate.
- 13 of 14 real events survived; `(24000, 32000)` was the one dropped.
- 13 of 39 false events survived — those are the next target.

## Open issues

**13 surviving false events, mostly fragmented tracks.**

    (7000, 15000)     (26000, 32000)
    (15000, 17000)    (26000, 85000)
    (15000, 95000)    (26000, 95000)
    (17000, 65000)    (33000, 38000)
    (18000, 65000)    (40000, 42000)
    (20000, 82000)    (96001, 112000)
    (25000, 46000)

Track `26000` appears 3x, `15000` appears 3x. `96001` is a split
composite of orig track 96.

The fragmentation pattern suggests the tracker is splitting single
physical vehicles into multiple ID segments; each segment then
participates in separate PET events against the same neighbors.

Next predeclared change: reject PET pairs where either track has
a nonzero segment index (i.e. it is a split fragment, not the
original track). Verify first that no Y event uses a nonzero segment.


## Why we did not loosen to 0.15

The one Y event dropped by the gate, `(24000, 32000)`, sits at
max_missing_ratio 0.1267. Loosening the threshold from 0.10 to
0.15 recovers it, but it also recovers four N events that the
human rejected:

  (24000, 26000)  0.1267  N
  (24000, 32000)  0.1267  Y   <- the one we want
  (24000, 41000)  0.1267  N
  (25000, 124000) 0.1356  N
  (48000, 82000)  0.1267  N

Three of the four new N events share track 24000 -- a track with
six dropout gaps that pairs against three different neighbours,
only one of which is a real PET. The gate correctly rejects all
three; loosening it would admit the whole cluster.

    config        Y   N   precision  recall
    current (0.10) 13  13  0.500      0.929
    loosened (0.15) 14  17  0.452      1.000

Net: +0.071 recall, -0.048 precision. Rejected.

## The 13 surviving N events are not a tracking problem

Missing-ratio range of the 13 survivors: 0.000 to 0.083. Every one
is comfortably under the threshold. The cleanest is
`(33000, 38000)`, missing_ratio 0.000 on both sides, human
verdict N.

No tracking-quality gate can separate these from the real events
in the same range. The disagreement is semantic: the pipeline and
the reviewer are using different definitions of "conflict" (e.g.
same-direction following, benign passing, geometric conflicts
that don't represent near-misses). Resolving it requires either
(a) strips + a reason column on the N events, or (b) a geometric
filter (heading, lateral offset, min-distance in pixels at the
conflict point). Not a missing-ratio threshold.

## Segment gate rejected

Considered adding `if seg_a > 0 or seg_b > 0: continue` to reject
split-track fragments. Checked against the human labels:

  Y events with seg>0: 4 / 14
  N survivors with seg>0: 1 / 13

The gate would kill 4 real events to reject 1 false one.
Discarded.

## Second gate: sequential-angle rejection (v2)

Added `_heading_near_point()` and `_angle_diff_deg()` and a
`max_sequential_angle_deg` knob (default `55.0`). Pairs whose travel
directions at the conflict point differ by more than the threshold,
or whose headings cannot be measured, are rejected.

### Why an angle gate

On the 13 clean-track N survivors of the missing-ratio gate,
`missing_ratio` was 0.000-0.083 on both sides -- no tracking-quality
threshold separates them from real events. But six of them have
heading difference 78-166 deg at the conflict point: two vehicles
travelling in substantially different directions cannot be a
sequential near-miss, because timing alone cannot make them collide.

### Measurement (same 300-frame eval, same 53-event review)

    gate                    events  Y   N   precision  recall
    baseline (no gate)         53  14  39   0.264     1.000
    missing_ratio 0.10         26  13  13   0.500     0.929
    + angle 30 deg             19  12   7   0.632     0.857
    + angle 55 deg             20  13   7   0.650     0.929

### Why 55 deg and not 30 deg

30 deg was an initial guess from an offline simulation that used
cross-track closest approach to approximate the conflict point.
The pipeline's `_pair_conflict_point` uses segment intersection and
returns a different point; one real event `(134000, 171000)`
measured 54 deg at the pipeline's conflict point while the
simulation put it at 14 deg. The threshold was recalibrated to
55 deg after seeing the divergence.

Real events max at 54 deg on this eval. Rejected events start at
78 deg. Any threshold in (24, 78) produces identical decisions.
55 deg was chosen inside that band with margin on both sides.

### Preserved

- 6 clearly-wrong N events with 78-166 deg heading: rejected.
- `(24000, 32000)` -- real Y, missing_ratio 0.127 -- still
  rejected by the missing-ratio gate; the angle threshold was
  not widened to recover it.

### Not resolved

7 N events survive both gates. All have heading difference
<= 24 deg at the conflict point, indistinguishable from real
events by tracking quality or by heading. Next step is a
reviewer reason column rather than another numeric threshold.

