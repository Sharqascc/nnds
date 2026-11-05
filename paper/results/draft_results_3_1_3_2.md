# Results (draft)

## 3.1 Descriptive statistics

The pipeline produced 199 raw conflict candidates across the two sites,
of which 187 passed screening (GITI 153/164, MRC 34/35; 94% retention).
Post-encroachment times ranged from 0.13 s to 3.00 s, with a combined
mean of 1.589 s and median of 1.566 s. Table 1 reports per-site counts and
PET statistics; Table 2 reports the conflict-type and severity breakdowns.

Table 1. Event counts and PET statistics (seconds).

| Site | Raw | Screened | Mean  | Median | Std   | Min   | Max   |
|------|----:|---------:|------:|-------:|------:|------:|------:|
| GITI | 164 |      153 | 1.564 |  1.566 | 0.831 | 0.167 | 2.999 |
| MRC  |  35 |       34 | 1.704 |  1.600 | 0.919 | 0.133 | 2.999 |
| Both | 199 |      187 | 1.589 |  1.566 |  —    | 0.133 | 2.999 |

Table 2. Conflict-type distribution.

| Conflict type | GITI | MRC | Total |
|---------------|-----:|----:|------:|
| rear_end      |    4 |   3 |     7 |
| head_on       |   32 |   1 |    33 |
| crossing      |   30 |   1 |    31 |
| side_swipe    |   34 |   4 |    38 |
| other         |   53 |  25 |    78 |

Table 3. Severity distribution by PET.

| Severity (PET)      | GITI | MRC | Total |
|---------------------|-----:|----:|------:|
| Critical (<1.0 s)   |   49 |   8 |    57 |
| Serious (1.0-1.5 s) |   24 |   8 |    32 |
| Moderate (1.5-3.0 s)|   80 |  18 |    98 |
| Safe (>3.0 s)       |    0 |   0 |     0 |

## 3.2 Cross-site comparison

We tested whether the PET distributions at GITI and MRC are statistically
distinguishable. A two-sided Mann-Whitney U test yielded U = 2318,
p = 0.32 (rank-biserial r = +0.11), and a two-sample Kolmogorov-Smirnov
test yielded D = 0.163, p = 0.41. Bootstrap 95% confidence intervals for
the difference in medians (-0.03 s; CI [-0.87, +0.37]) and means
(-0.14 s; CI [-0.48, +0.20]) both include zero. At this sample size, we
find **no statistically detectable difference** in PET magnitude between
the two sites. Post-hoc power for the observed effect size (Cohen's
d = -0.17) is approximately 0.14, so this is a null result under
substantial uncertainty rather than evidence of equivalence.

The conflict-*type* composition, in contrast, differs significantly
(chi2 = 24.2, df = 4, p = 0.0001). MRC is dominated by "other" (25/34
events), whereas GITI spans all five categories with head_on, crossing,
and side_swipe each contributing ~20% of events. This pattern is
consistent with site geometry and traffic composition rather than method
behavior: the same pipeline, applied to the same detector weights,
yields the same *measure* of conflict (PET) at both sites but naturally
different *kinds* of conflicts.

The severity distribution does not differ between sites
(chi2 = 1.66, df = 2, p = 0.44). The "safe" bin was empty at both sites
and was dropped before the test; all 187 events fall in the critical,
serious, or moderate bands.
