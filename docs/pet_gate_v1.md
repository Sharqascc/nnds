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
