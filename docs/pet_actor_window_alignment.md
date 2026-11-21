# PET actor/window alignment correction

## Export contract

For events emitted by the UVH-COCO fused grid PET runner:

- `track_a` is the first-passing actor and `track_b` is the second.
- Original track IDs, segment IDs, trajectories, gate labels, and
  entry/exit frame and time fields must follow that same ordering.
- Exported `first_actor="a"` and `second_actor="b"` refer to the exported
  actors, not the original candidate enumeration.
- The structured PET helper retains its own input-relative actor labels;
  its helper-level semantics are unchanged.

Previously, B-first events reordered actor IDs and trajectories but retained
the original candidate ordering for the exported timing fields. This could
associate an actor with the other actor's occupancy window.

## Scope

This correction aligns exported timing fields and actor labels.
It does not change PET thresholds, positive-PET filtering, overlap exclusion,
gate behavior, or conflict classification.

Older CSV files are not corrected automatically. Regenerate or explicitly
audit them before using actor-specific timing relationships.

## Regression evidence

Validation performed in Colab on 2026-11-21:

- Focused patched suite: 44 passed.
- New four-case regression against pre-fix source at commit dd44779:
  2 failed and 2 passed. Both failures occurred because the exported
  first actor's entry frame was 3 instead of 1.
- Broader patched backend suite: 208 passed, 1 warning.
- The warning concerned scripts.validate_outputs already being present
  in sys.modules before execution through runpy; it remains unresolved.

The regression covers both passage orders and both candidate insertion
orders, checking returned events and written CSV rows. It checks IDs,
segments, timing fields, PET consistency, actor labels, and actor-specific
trajectory markers.

## Limitations

These are mocked backend regression checks, not full-video or scientific
acceptance validation. The full repository test suite was not run.

The Colab installation reported dependency conflicts involving google-colab
(IPython and pandas) and numba (NumPy). Passing these checks does not establish
a conflict-free or fully reproducible Colab environment.
