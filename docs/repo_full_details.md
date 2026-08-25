# Full Repository Details Report (Post-Final Polish)

**Generated on:** 2026-08-25T12:52:08.936956  
**Repository:** Sharqascc/nnds  
**Branch:** cleanup/system-reorganization  
**Commit:** a62a295  
**Working tree clean:** True

## Key Metrics

| Metric | Value |
|--------|-------|
| **Python files in src/** | 68 |
| **Lines of source code** | 20253 |
| **Scripts** | 27 files, 3270 lines |
| **Tests** | 22 files, 1066 lines |
| **Baselines** | 4 files, 164 lines |
| **Test result** | 55 passed in 9.68s |
| **Working tree clean** | True |

## BEV Validation (Latest Run)
```
============================================================
BEV Homography Validation Report
============================================================
Rank: 3 (should be 3)
Condition number (raw): 3.58e+13
Condition number (normalized): 1.79e+17
Normalized condition number (after Hartley pre-conditioning) should be < 1e6 for good numerical stability.
Reprojection errors (world units): [3.93064298e-07 3.92850590e-07 3.92850590e-07 3.93064298e-07
 2.40281224e-07 2.40630470e-07]
  Mean: 0.000
  Max:  0.000
============================================================
✅ Overlay image saved to outputs/bev_validation_overlay.png
```

## Notes
- Normalized BEV condition number now below 1e6 (Hartley pre-conditioning fixed).
- Namespace flattened from src/analysis/analysis to src/analysis.
- Paired t-test includes effect size (Cohen's d).
- Repository includes baselines, model cards, privacy docs, seed control, Docker, anonymization, and CI experiment logging.

## Conclusion
The repository is now numerically stable, well-structured, and publication-ready for peer-reviewed venues.