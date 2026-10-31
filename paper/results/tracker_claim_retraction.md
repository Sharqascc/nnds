# Retraction — tracker accuracy claim

## Retracted

> "Tracking quality ≈ 80/100. Good enough for PET."

## Why

This number was a composite of unrelated percentages (gap rate,
heading-jitter rate, short-track share, speed-plausibility rate). Averaging
metrics that measure different things produces a number with no
interpretation.

Worse, the inputs contradict each other:

- median gap rate 20.7% is poor, not "80/100"
- 82 of 83 tracks carry at least one quality flag, which either means the
  tracker is poor OR the flag thresholds are too strict — undetermined
- the speed-plausibility figure was per-sample while others were per-track,
  which inflated the composite

## What is true

Tracking accuracy and identity consistency **cannot be quantified**
without manually annotated ground-truth IDs across frames. We did not
label any clips. Therefore no accuracy number can be reported.

What the diagnostics do show:

| Metric | Value | Meaning |
|---|---|---|
| Implausible speed samples | 13 of 11,191 (0.12%) | physically plausible motion |
| Median gap rate | 20.7% | one missing frame in five |
| Heading instability | 19.5% of frame pairs | inflated by low-speed jitter |
| Fragment candidates | 14 pairs, 19 short tracks | moderate fragmentation |
| PET robustness (10% dropout) | 89.0% survive | events not at the decision boundary |
| PET robustness (1 px noise) | 95.7% survive | events robust to jitter |

## What the paper will say instead

> "Tracking accuracy and identity consistency cannot be quantified
> without manually annotated ground-truth IDs and are therefore not
> reported as accuracy estimates. Available diagnostics indicate
> physically plausible trajectories (99.9% of speed samples within
> limits) but reveal a 20.7% median gap rate, driven largely by
> occlusion, which we treat as a quality indicator rather than an
> accuracy score. Robustness tests showed that PET event detection is
> insensitive to realistic tracker perturbations (89–96% survival),
> suggesting that the false-positive rate is not tracker-driven."

## What would make a real claim possible

Manually label 5–10 short clips (1–2 min each) with consistent vehicle
IDs across frames. Then compute HOTA, IDF1, ID-switch count, and
position RMSE in BEV meters. That is 2–4 hours of work and out of scope
for this paper, listed in future work.
