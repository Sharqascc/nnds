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

## Sensitivity Analysis (300 frames)
```
 events   median     mean     std  max_gap  max_jump
    124 1.016667 1.037903 0.56585        5        30
    124 1.016667 1.037903 0.56585       10        50
    124 1.016667 1.037903 0.56585       15        80
    124 1.016667 1.037903 0.56585       20       100
```


## Corrected 300-frame sensitivity
```
 events   median     mean      std  max_gap  max_jump
    156 1.016667 0.982265 0.601282        5        30
    140 1.016667 1.005952 0.596610       10        50
    127 1.100000 1.070341 0.582766       15        80
    131 1.133333 1.101272 0.582089       20       100
```


## Deconfounded sensitivity
```
 events   median     mean      std variable  max_gap  max_jump
    156 1.016667 0.982265 0.601282      gap        5      30.0
    150 0.966667 0.969333 0.602537      gap       10      30.0
    147 1.000000 0.978231 0.606530      gap       15      30.0
    147 1.000000 0.978231 0.606530      gap       20      30.0
    156 1.016667 0.982265 0.601282     jump        5      30.0
    148 1.016667 0.997523 0.592458     jump        5      50.0
    138 1.100000 1.053382 0.576725     jump        5      80.0
    155 1.100000 1.052688 0.580676     jump        5     100.0
```


## Limitations
- Held-out BEV reprojection error is extremely low (0.000001 ft), which may indicate overfitting or insufficient independent validation. Independent field-surveyed points are recommended.
- Detection/tracking MOT metrics require manually annotated ground truth.
