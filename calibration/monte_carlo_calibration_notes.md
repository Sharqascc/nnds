# Monte Carlo Calibration Benchmark – Paper Notes

## Study goal

Evaluate whether planar homography is sufficient for ground-plane coordinate recovery in traffic-camera calibration, or whether PnP-based camera pose estimation with 3D survey points yields significantly better accuracy under realistic non-idealities (distortion, noisy survey, slight non-planarity).

## Experimental setup

- Synthetic traffic-like scene with a 49.5 × 22 m ground grid sampled at 27 × 12 points (324 world points).
- Camera at ~25 m height, tilted towards the ground; 1920 × 1080 image resolution.
- Intrinsics: focal length ≈ 1600 px, principal point at image center; radial distortion k1 = -0.12 (higher-order coefficient k2 = 0.02).
- Ground non-ideality: linear Z-bias of 1.5 cm across X to mimic imperfect planarity.
- Image-space perturbations: anisotropic Gaussian noise with σx = 2 px, σy = 3 px, ρ = 0.83.
- Monte Carlo: 50 trials by default, new noise per trial, fixed camera/world geometry, all controlled via CLI flags.

Implementation details:

- Code: `calibration/monte_carlo_calibration_benchmark.py` in the NNDS repository.
- Dependencies: Python 3.10+, NumPy, OpenCV, optional `tqdm` (progress bar) and Matplotlib (for plots).
- Configuration: number of trials, random seed, noise levels, and plane bias are exposed as arguments.
- Outputs: per-scenario JSON summaries (`mae_stats`, noise parameters, comparison ratios) plus optional MAE histograms.

## Methods

All methods estimate world (X, Y) coordinates from noisy image points and are evaluated on the same set of 324 ground-truth points.

- **Homography (biased world)**  
  Homography H is estimated from noisy image points to a “biased” 2.5D world where a 1.5 cm Z-slope is baked into the world coordinates.  
  This represents a typical practice where small non-planarity is ignored and the grid is treated as planar even though the survey encodes some Z variation.

- **Homography (Z = 0 world)**  
  Homography H is estimated from the same images to an ideal Z = 0 world (all points projected to a flat plane).  
  This is a fair planar baseline vs PnP(Z=0), isolating the limitations of pure projective mapping under distortion and noise.

- **PnP ITERATIVE (biased Z)**  
  Full 3D world (with Z-bias) + 324 correspondences → `cv2.solvePnP` with ITERATIVE flag → estimated pose (R, t) → ray–plane intersection to Z = 0.  
  This approximates a calibration workflow where 3D survey points are available and a full camera pose is recovered before projecting to the road plane.

- **PnP ITERATIVE (Z = 0)**  
  Same as above but the survey Z is forced to 0 before PnP, simulating the case where ground plane is assumed perfectly planar in the survey.  
  This tests sensitivity of PnP to small non-planarity in the survey.

- **P3P + RANSAC**  
  Minimal P3P solver inside a simple RANSAC loop over 4-point subsets.  
  For each candidate pose, the best hypothesis is evaluated on all points using the same ray–plane back-projection and MAE metric.

Metric:

- Mean absolute error (MAE) in world (X, Y) over all 324 points (meters).

The benchmark script also computes a simple comparative statistic:

- `pnp_vs_homography_factor = PnP_iter_mean / H_biased_mean`  
  Values < 1 indicate that PnP improves over homography by that factor (e.g., ≈0.23 → ~4.4× lower error).

## Key results (example run, 50 trials)

Example numbers from a representative run (as implemented in `monte_carlo_calibration_benchmark.py`):

- Homography (biased):   0.3228 ± 0.0077 m  
- Homography (Z = 0):    0.3228 ± 0.0077 m  
- PnP ITERATIVE (biased): 0.0731 ± 0.0026 m  
- PnP ITERATIVE (Z = 0):  0.0724 ± 0.0027 m  
- P3P + RANSAC:           0.0782 ± 0.0037 m  

Relative improvement:

- PnP ITERATIVE achieves ≈4.4× lower MAE than homography in this setup (pnp_vs_homography_factor ≈ 0.23).

## Interpretation

- Homography MAE ≈ 32 cm and is effectively identical for biased vs Z = 0 targets, so its dominant limitations are projective mapping and distortion rather than the small Z-bias in the survey.
- PnP reduces MAE to ≈ 7.3 cm, a ~4.4× improvement over homography, with very low variance across trials.
- PnP(biased) vs PnP(Z = 0) differ by ~0.07 cm (~1%), showing practical invariance to a 1.5 cm plane bias at 25 m camera height.
- P3P + RANSAC is slightly worse (~7–8% higher MAE) than full iterative PnP, consistent with minimal vs overdetermined solvers.

Connection to NNDS PET/TTC error budget:

- In the NNDS pipeline, a world-coordinate MAE in the range 0.07–0.08 m translates to PET uncertainty on the order of 0.05–0.1 s for typical intersection speeds, which is acceptable for PET thresholds in the 1–3 s range used in safety analysis.

## Reproducing the benchmark in NNDS

From the NNDS project root:

### Single-scenario benchmark

Run the baseline 50-trial Monte Carlo experiment and save a basic MAE histogram:

```bash
PYTHONPATH=. python calibration/monte_carlo_calibration_benchmark.py \
    --num-trials 50 \
    --seed 0 \
    --plot
```

This produces:

- Console summary of MAE for all methods (homography, PnP, P3P).
- `calibration/monte_carlo_calibration_summary.json` with:
  - `camera`, `grid`, and `noise` configuration.
  - `mae_stats` for each method (mean, std).
  - `comparison.pnp_vs_homography_factor`.
- `calibration/calibration_errors_pnp.png` – histogram of PnP ITERATIVE world MAE (if `--plot` is set).

### Multi-noise scenarios (parameter sweep)

To evaluate robustness across different pixel noise levels, enable multi-noise mode:

```bash
PYTHONPATH=. python calibration/monte_carlo_calibration_benchmark.py \
    --num-trials 50 \
    --seed 0 \
    --multi-noise
```

This runs the benchmark for several (σx, σy) settings (e.g., 0.5, 1.0, 2.0, 3.0 px in x with proportional scaling in y) and writes:

- One JSON summary per scenario, e.g.  
  `calibration/monte_carlo_calibration_summary_sigx0p5_sigy0p8.json`  
  `calibration/monte_carlo_calibration_summary_sigx1p0_sigy1p5.json`  
  ...
- A small index file:  
  `calibration/monte_carlo_calibration_summary_index.json`  
  listing available scenarios and the base output stem.

These summaries can be loaded into a notebook to generate boxplots, noise–MAE curves, and statistical tests across noise regimes.

## For the paper

Recommended reporting:

- Present MAE distributions for all methods (homography biased/Z=0, PnP biased/Z=0, P3P) in a single boxplot figure.
- Report paired t-tests:
  - Homography vs PnP (expect very small p-values and large effect sizes).
  - PnP(biased) vs PnP(Z = 0) (expected non-significant difference).
- Report effect size (Cohen’s d) for Homography vs PnP (expected to be “huge”).
- Quote the `pnp_vs_homography_factor` as a concise measure of improvement (e.g., “PnP reduces mean error by ~4.4× relative to homography under the tested conditions”).

Main claim:

> When 3D survey coordinates are available, explicit pose estimation via PnP plus ray–plane intersection clearly outperforms direct planar homography under realistic calibration artifacts (distortion, survey noise, modest non-planarity), and is robust to small deviations from planarity at typical traffic-camera mounting heights.
