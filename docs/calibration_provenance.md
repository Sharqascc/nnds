# Calibration Units & Provenance

## Coordinate Units
- **Horizontal world coordinates (Easting, Northing): metres**
- **Elevation: feet** (as indicated by `elevation_ft`)

The rectangle formed by the six correspondences has dimensions **20 m × 16 m**.

## Provenance of Control Points
The six calibration points were derived from an **idealized rectangular target
geometry**, not independently surveyed with a total station or GPS. They were
placed at the corners and mid‑edges of the assumed rectangle to produce a
consistent planar coordinate system.

## Implication
The near‑zero reprojection residual reflects **internal numerical consistency**
with the planar calibration assumption. It is **not** evidence of independent
field‑level metric accuracy.

Absolute real‑world localisation accuracy remains **unvalidated** in this
study. Any physical metric derived from BEV coordinates (distance, speed,
conflict location) should be interpreted as conditional on the assumed planar
calibration.
