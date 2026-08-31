> **DEPRECATED** — This document contains historical results. See `STATUS.md` for current results.

# Final Comprehensive Assessment Report (Complete)

**Generated:** 2026-08-26T19:24:31.763852  
**Branch:** cleanup/system-reorganization  
**Commit:** f0c6f12  

## Test Suite
Result: 60 passed in 9.71s

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
```

## BEV Held-out Manual Trace (point 0)
Pixel held out: [194.0, 124.0]
World true: [730900.97, 221998.35]
World projected: [730900.9699992008, 221998.3500000787]
Error: 0.0000008031 ft
Training indices: [1,2,3,4,5] (point 0 not used)

## E2E 300-frame PET Summary
Events: 156
Median PET: 1.017s
Mean PET: 0.982s
Std: 0.601s

## Prediction Tolerance Sensitivity (100 frames)
|   prediction_tolerance |   events |   median |     mean |      std |
|-----------------------:|---------:|---------:|---------:|---------:|
|                      0 |       27 | 1.03333  | 0.955556 | 0.639845 |
|                     40 |       33 | 1.03333  | 0.942424 | 0.574412 |
|                     80 |       75 | 0.9      | 0.903556 | 0.594483 |
|                    120 |       76 | 0.966667 | 0.960526 | 0.621151 |

## Key Limitations
- Full MOT metrics require manually annotated ground truth; infrastructure in place.
- BEV held-out error is extremely low (0.000001 ft) because calibration points and homography share a planar ground assumption. This reflects self-consistency, not independently validated metric accuracy. Field validation against independently measured distances was not performed.
- Prediction tolerance sensitivity is now quantified; default 80.0 chosen as balance.
- Single intersection dataset; domain shift possible.

## Conclusion
All previously flagged gaps have been addressed: prediction tolerance sensitivity run, E2E report regenerated, BEV held-out manually verified. Repository is ready.
## Prediction Tolerance Sensitivity (300 frames)
|   prediction_tolerance |   events |   median |     mean |      std |
|-----------------------:|---------:|---------:|---------:|---------:|
|                      0 |       74 | 0.75     | 0.938739 | 0.581266 |
|                     40 |       75 | 0.866667 | 0.965778 | 0.555182 |
|                     80 |      156 | 1.01667  | 0.982265 | 0.601282 |
|                    120 |      163 | 1.06667  | 1.02045  | 0.583275 |

## Honest Sensitivity Interpretation
Prediction tolerance is the dominant parameter controlling fragmentation. Event counts vary from 27 to 156 across the tested range. We therefore do **not** claim the pipeline is fully robust to all fragmentation parameters; instead, we report the chosen default (80 px) and the full sensitivity table for transparency.


## Steep Sensitivity Region
The prediction tolerance sweep shows a sharp nonlinearity between 40 px (75 events) and 80 px (156 events). Our chosen default (80 px) sits near this steep part of the sensitivity curve; small perturbations around this value can alter event counts more than at 0 or 120 px.
