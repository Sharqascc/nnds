# Calibration Limitations — GCP Coverage and Homography Accuracy

This document records a known limitation in the pixel-to-world calibration used
in this project, tied to the commits and analysis that produced it.

## Summary

The pixel-to-world homography was fit using six ground control points (GCPs)
surveyed at the GITI intersection site.

- **Self-residual reprojection error** (fit and tested on the same 6 points): **RMSE = 0.038 m**
- **Leave-one-out cross-validation (LOOCV) error** (fit on 5 points, tested on the held-out 6th, repeated across all 6): **RMSE = 0.311 m**
- The LOOCV estimate is **8.2x higher** than the self-residual estimate.

With only six correspondences fitting an eight-degree-of-freedom homography,
the self-residual metric is expected to look artificially low regardless of
true generalization accuracy -- there's minimal spare constraint left to reveal
real error. LOOCV RMSE (0.311 m) is adopted as the calibration's true
generalization error and is propagated as the default homography uncertainty
in downstream Post-Encroachment Time (PET) conflict analysis.

## GCP spatial coverage

The six surveyed GCPs span only approximately 1.7 m x 1.4 m in world
coordinates -- a footprint implausibly small for a real signalized
intersection. This likely:

- constrains the homography's conditioning, and
- limits how representative the LOOCV estimate is of error across the full
  mapped region (e.g., near intersection edges, far from the surveyed cluster).

## Recommendation

Re-survey with a larger number of well-spread GCPs (10 or more,
distributed across the full intersection footprint) before treating either
the self-residual or LOOCV error as fully representative of production-level
calibration accuracy.

All calibration-derived position estimates and PET uncertainty bounds
reported in this work should be interpreted with this limitation in mind.

## Related commits

- `ab5116f` -- LOOCV validation + fixed silently-broken calibration schema parsing
- `f8220a5` -- Fixed BEVMapper class ordering bug + implemented two missing methods
- `3d0ece4` -- Documented why `--real-data` is intentionally unimplemented rather than a silent trap

## Related data

- LOOCV results JSON: *(add path/link once available, e.g. `calibration/loocv_results.json`)*
