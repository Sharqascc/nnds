# Calibration Units & Provenance

## Coordinate Units
The world coordinates in `giti_calibration_points.json` and `bev_config.json`
are in **US survey feet**, not metres. The rectangle formed by the six
correspondences has dimensions **20 ft × 16 ft** (≈ 6.10 m × 4.88 m).

When converting to local metric coordinates for BEV visualization, we apply
the conversion `1 ft = 0.3048 m`.

## Provenance of Control Points
The six calibration points were derived from an **idealized rectangular target
geometry**, not independently surveyed with a total station or GPS. They were
placed at the corners and mid‑edges of the assumed rectangle to produce a
consistent planar coordinate system.

## Implication
The near‑zero reprojection residual (≈ 0.24 µm after conversion) reflects
**internal numerical consistency** with the planar calibration assumption.
It is **not** evidence of independent field‑level metric accuracy.

Absolute real‑world localisation accuracy remains **unvalidated** in this
study. Any physical metric derived from BEV coordinates (distance, speed,
conflict location) should be interpreted as conditional on the assumed planar
calibration.
