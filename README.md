# 🚦 NNDS: Neural Network for Dynamic Safety
[![CI](https://github.com/Sharqascc/nnds/actions/workflows/ci.yml/badge.svg)](https://github.com/Sharqascc/nnds/actions/workflows/ci.yml)

# NNDS: Neural Network for Dynamic Safety

NNDS is a comprehensive traffic safety analysis pipeline that detects vehicles, pedestrians, and tracks them in 3D space. The pipeline computes safety metrics (PET, TTC, DRAC) and generates bird's-eye-view (BEV) visualizations.

## Installation

```bash
git clone https://github.com/Sharqascc/nnds.git
cd nnds
pip install -r requirements.txt
```

## Evaluation

NNDS ships with a full SSM evaluation framework. Ground-truth CSV
schemas are documented in docs/ANNOTATION_GUIDE.md, and every metric
is defined in docs/METRICS.md.

Quick start once GT CSVs exist:

    python scripts/validate_gt.py --detection gt_detection.csv --tracking gt_tracking.csv --trajectory gt_trajectory.csv --ssm gt_ssm.csv
    python scripts/evaluate_detection_metrics.py  --detections pred_det.csv  --ground-truth gt_detection.csv  --out-json detection.json
    python scripts/evaluate_tracking_metrics.py   --tracked    pred_trk.csv  --ground-truth gt_tracking.csv   --out-json tracking.json
    python scripts/evaluate_trajectory_metrics.py --predicted  pred_traj.csv --ground-truth gt_trajectory.csv --out-json trajectory.json
    python scripts/evaluate_ssm_metrics.py        --predicted  pred_ssm.csv  --ground-truth gt_ssm.csv        --out-json ssm.json
    python scripts/gold_standard_report.py --detection-metrics detection.json --tracking-metrics tracking.json --trajectory-metrics trajectory.json --ssm-metrics ssm.json --out-md outputs/validation_report.md

For a self-contained smoke test on synthetic data:

    python scripts/demo_end_to_end.py

The demo and the test suite verify that the metric implementations are
correct. They do not measure NNDS accuracy. Real accuracy numbers
require real ground truth. See docs/VALIDATION.md.

