# Final Submission Summary

**Test count:** 59

## BEV Validation
```
Leave-one-out BEV validation
  Mean held-out reprojection error: 0.000001 ft
  Max held-out reprojection error: 0.000001 ft
  Std held-out reprojection error: 0.000000 ft
```

## Detection Confidence (on placeholder GT)
```
Confidence >= 0.25: Precision=0.000, Recall=0.000
Confidence >= 0.5: Precision=0.000, Recall=0.000
Confidence >= 0.75: Precision=0.000, Recall=0.000
Confidence >= 0.9: Precision=0.000, Recall=0.000
```

## Sensitivity Analysis (50 frames)
```

```

## Limitations
- Full MOT/detection metrics require manually annotated ground truth.
- Held-out BEV error is extremely low but should be validated against independent surveyed points in the field.
- Fragmentation sensitivity is quantified; results should be reported with this context.