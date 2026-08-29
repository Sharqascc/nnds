# GITI Calibration Provenance


## Calibration Sensitivity (Conditional Monte Carlo)

We performed a conditional Monte Carlo sensitivity analysis (200 trials per perturbation level) in which all image-space calibration correspondences were independently perturbed with zero-mean Gaussian noise (σ = 0.5, 1.0, 2.0 px) and the Hartley-normalized homography was re-estimated. For a tested synthetic constant-speed trajectory (36 km/h) within the calibrated ROI, the calibration-image-point perturbation component produced:

| σ (px) | Mean abs error (km/h) | 95th percentile (km/h) | Max (km/h) | Relative error (%) |
|--------|----------------------|------------------------|------------|-------------------|
| 0.5    | 0.034715183243313914              | 0.08364155436377022                | 0.12230561582700261    | 0.09643106456476087           |
| 1.0    | 0.07240660632498429              | 0.17219633309887744                | 0.2565472584896966    | 0.20112946201384527           |
| 2.0    | 0.1484782135259698              | 0.3641160537891394                | 0.4690605494730491    | 0.41243948201658276           |

This result indicates **low local sensitivity** to the specified image-point perturbation model. It does **not** represent externally validated absolute speed accuracy. The analysis does not account for detector, tracker, scale, distortion, timing, or non-planarity errors.

**Interpretation:** The speed estimate is robust to plausible calibration-point pixel noise under the tested conditions, but the calibration itself remains synthetic (20×16 m rectangle) and not field-validated.
