
# Monte Carlo Calibration Benchmark – Paper Notes

## Study goal
Evaluate whether planar homography is sufficient for ground-plane coordinate recovery in traffic-camera calibration, or whether PnP-based camera pose estimation with 3D survey points yields significantly better accuracy under realistic non-idealities (distortion, noisy survey, slight non-planarity).

## Experimental setup
- Synthetic traffic-like scene with a 49.5 × 22 m ground grid sampled at 27 × 12 points (324 world points).
- Camera at ~25 m height, tilted towards the ground; 1920 × 1080 image resolution.
- Intrinsics: focal length ≈ 1600 px, principal point at image center; radial distortion k1 = -0.12.
- Ground non-ideality: linear Z-bias of 1.5 cm across X to mimic imperfect planarity.
- Image-space perturbations: anisotropic Gaussian noise with σx = 2 px, σy = 3 px, ρ = 0.83.
- Monte Carlo: 50 trials, new noise per trial, fixed camera/world geometry.

## Methods
- Homography (biased world): H from noisy image points to biased 2D world (with 1.5 cm Z-slope baked in).
- Homography (Z=0 world): H from same images to ideal Z=0 world; fair planar baseline vs PnP(Z=0).
- PnP ITERATIVE (biased Z): full 3D world (with Z-bias) + 324 points → solvePnP → ray–plane intersection to z=0.
- PnP ITERATIVE (Z=0): same but with all survey Z forced to 0; tests sensitivity to small non-planarity.
- P3P + RANSAC: minimal P3P solver inside simple RANSAC over 4-point subsets; evaluate best hypothesis on all points.

Metric: mean absolute error (MAE) in world (X,Y) over all 324 points.

## Key results (example run, 50 trials)
- Homography (biased):   0.3228 ± 0.0077 m
- Homography (Z=0):      0.3228 ± 0.0077 m
- PnP ITERATIVE (biased): 0.0731 ± 0.0026 m
- PnP ITERATIVE (Z=0):    0.0724 ± 0.0027 m
- P3P + RANSAC:           0.0782 ± 0.0037 m

## Interpretation
- Homography MAE ≈ 32 cm and is identical for biased vs Z=0 targets, so its dominant limitations are projective mapping and distortion rather than small Z-bias.
- PnP reduces MAE to ≈ 7.3 cm, a ~4.4× improvement over homography, with very low variance.
- PnP(biased) vs PnP(Z=0) differ by ~0.07 cm (~1%), showing practical invariance to a 1.5 cm plane bias at 25 m height.
- P3P+RANSAC is slightly worse (~7–8% higher MAE) than full iterative PnP, consistent with minimal vs overdetermined solvers.

## For the paper
- Report paired t-tests: Homography vs PnP (strongly significant, tiny p-value), and PnP(biased) vs PnP(Z=0) (non-significant).
- Report effect size (Cohen’s d) for Homography vs PnP (expected to be “huge”).
- Include a boxplot figure comparing MAE distributions for all methods.
- Main claim: when 3D survey coordinates are available, explicit pose estimation plus ray–plane intersection clearly outperforms direct planar homography under realistic calibration artifacts, and is robust to modest ground non-planarity.
