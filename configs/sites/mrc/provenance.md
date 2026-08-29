# MRC Calibration Provenance

## Calibration Source
**IMPORTANT:** The MRC world coordinates are **synthetic** (an idealized 20 m × 16 m rectangle with origin at P4).  
They are **NOT independent field measurements** (no total station, RTK GPS, or tape measure).  
This means the reported reprojection errors and leave-one-out results quantify **internal geometric consistency**, not absolute real-world metric accuracy.

## Method
- Homography estimated via Hartley-normalized DLT from 6 pixel↔world correspondences.
- Pixel coordinates were refined using an automatic optimization routine (coordinate descent) to minimize reprojection error.

## Validation Results
- **Full-fit mean error:** 0.0038 m (max 0.0094 m)
- **Leave-one-out mean error:** 0.015 m (max 0.033 m)
- **Condition number (Hartley normalized):** 1.05 × 10⁴ (workable, but higher than ideal)
- **Perturbation sensitivity (±1 px, 200 trials):** Median BEV displacement ≈ 0.05 m (see `perturbation_sensitivity.json`)

## Limitations
- The calibration is **internally consistent** but **not field-validated**.
- Absolute metric accuracy is **not established**.
- Lens distortion correction is **not applied** to the video (checkerboard calibration from 3000×4000 stills is incompatible with the 1600×720 video due to sensor crop mismatch).
- Outside the calibration region (beyond the 24×20 m ROI), homography extrapolation is unreliable.

## Coordinate Reference and Validation Scope
The MRC configuration defines a local planar reference frame spanning an assumed 20 m × 16 m region. The reported full-fit and leave-one-out residuals assess internal consistency of the image-to-local-coordinate homography with this reference geometry. The grid-distance checks verify configuration and rendering consistency because the same local-coordinate definition is used to specify the reference distances.

These diagnostics do not constitute independent field validation of absolute roadway scale or vehicle-localization accuracy. Independent surveyed checkpoints or measured pavement distances not used during homography estimation are required for such validation.

## Publication Figure
- `docs/mrc_bev_grid_publication.png` – Clean BEV figure with 1m grid, axes labeled, saved at 300 DPI.
- Origin verified: P4 projects near (0,0) at approximately (-0.014 m, -0.217 m), consistent with homography residuals.
