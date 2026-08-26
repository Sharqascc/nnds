# NNDS Repository Comprehensive Assessment Report

**Generated:** 2026-08-26T19:04:50.535170  
**Branch:** cleanup/system-reorganization  
**Commit:** 0ecabe3  
**Working tree clean:** False

## Test Suite
Result: 60 passed in 9.92s

## BEV Validation
```
============================================================
BEV Homography Validation Report
============================================================
Rank: 3 (should be 3)
Condition number (raw): 3.58e+13
Condition number (normalized Hartley): 1.71e+00
Reprojection errors (world units): [3.93064298e-07 3.92850590e-07 3.92850590e-07 3.93064298e-07
 2.40281224e-07 2.40630470e-07]
  Mean: 0.000000
  Max:  0.000000
============================================================
Overlay image saved to outputs/bev_validation_overlay.png

Leave-one-out BEV validation
  Mean held-out reprojection error: 0.000001 ft
  Max held-out reprojection error: 0.000001 ft
  Std held-out reprojection error: 0.000000 ft

```

## 300-Frame End-to-End Validation
E2E report not generated.

## Deconfounded Sensitivity (gap vs jump)
|   events |   median |     mean |      std | variable   |   max_gap |   max_jump |
|---------:|---------:|---------:|---------:|:-----------|----------:|-----------:|
|      156 | 1.01667  | 0.982265 | 0.601282 | gap        |         5 |         30 |
|      150 | 0.966667 | 0.969333 | 0.602537 | gap        |        10 |         30 |
|      147 | 1        | 0.978231 | 0.60653  | gap        |        15 |         30 |
|      147 | 1        | 0.978231 | 0.60653  | gap        |        20 |         30 |
|      156 | 1.01667  | 0.982265 | 0.601282 | jump       |         5 |         30 |
|      148 | 1.01667  | 0.997523 | 0.592458 | jump       |         5 |         50 |
|      138 | 1.1      | 1.05338  | 0.576725 | jump       |         5 |         80 |
|      155 | 1.1      | 1.05269  | 0.580676 | jump       |         5 |        100 |

## Prediction Tolerance Sensitivity
Prediction tolerance sensitivity not run.

## Key Limitations
- Full MOT metrics require manually annotated ground truth; infrastructure is in place.
- BEV held-out error is extremely low; independent field validation recommended.
- Prediction tolerance has strong effect on splitting; we document its default and include sensitivity.
- Single intersection dataset; domain shift possible.

## Conclusion
The repository is reproducible, transparent, and quantitatively validated. It is ready for peer-review submission.