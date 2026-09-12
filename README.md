# 🚦 NNDS: Neural Network for Dynamic Safety

[![CI](https://github.com/Sharqascc/nnds/actions/workflows/ci.yml/badge.svg)](https://github.com/Sharqascc/nnds/actions/workflows/ci.yml)

NNDS is a comprehensive traffic safety analysis pipeline that detects vehicles and pedestrians, tracks them in 3D space, computes safety metrics such as PET, TTC, and DRAC, and generates bird's-eye-view (BEV) visualizations.

## Installation

```bash
git clone https://github.com/Sharqascc/nnds.git
cd nnds
pip install -r requirements.txt
```

## GITI intersection video

The repository includes the GITI intersection traffic video at:

```text
data/sample_data/GITI_traffic_video.mp4
```

The video is stored with Git Large File Storage (Git LFS). Install Git LFS before downloading the video:

```bash
sudo apt-get update
sudo apt-get install git-lfs
git lfs install
git lfs pull
```

Verify that the video is available:

```bash
ls -lh data/sample_data/GITI_traffic_video.mp4
git lfs ls-files
```

The anonymized 50-frame fixture remains available for quick smoke tests:

```text
data/sample_data/anonymized_traffic_video_50f.mp4
```

Run the UVH-COCO fused pipeline on the GITI video:

```bash
python -m src.pipeline.traffic_analyzer \
  --video data/sample_data/GITI_traffic_video.mp4 \
  --detector uvh-coco-fused \
  --bev-config configs/bev_config.json \
  --grid-config configs/GITI_grid_config.json \
  --out-csv outputs/GITI_uvh_pet.csv \
  --max-frames 50
```

Remove `--max-frames 50` for a full-video run.

The pipeline writes a detection CSV and a PET CSV:

```text
outputs/GITI_uvh_pet_detections.csv
outputs/GITI_uvh_pet.csv
```

The PET CSV uses a stable 32-column schema, including when no PET events are detected.

## Data paths

| Asset | Path | Purpose |
|---|---|---|
| GITI traffic video | data/sample_data/GITI_traffic_video.mp4 | Primary GITI-intersection input |
| Anonymized fixture | data/sample_data/anonymized_traffic_video_50f.mp4 | Fast smoke-test input |
| UVH detector | data/models/uvh26.pt | UVH vehicle detector |
| COCO person model | data/models/yolo11n.pt | Person detection |
| BEV configuration | configs/bev_config.json | Bird's-eye-view mapping |
| GITI grid configuration | configs/GITI_grid_config.json | GITI grid mapping |

## Evaluation

NNDS ships with a full SSM evaluation framework. Ground-truth CSV
schemas are documented in docs/ANNOTATION_GUIDE.md, and every metric
is defined in docs/METRICS.md.

Quick start once GT CSVs exist:

```bash
# 1. Convert raw pipeline output to the metric schemas
python scripts/pipeline_to_metric_schemas.py \
    --detections-csv outputs/GITI_uvh_pet_detections.csv \
    --pet-csv        outputs/GITI_uvh_pet.csv \
    --bev-config     configs/bev_config.json \
    --out-dir        outputs/schemas

# 2. Validate ground truth
python scripts/validate_gt.py \
    --detection  gt_detection.csv \
    --tracking   gt_tracking.csv \
    --trajectory gt_trajectory.csv \
    --ssm        gt_ssm.csv

# 3. Compute metrics (each writes JSON for the next step)
python scripts/evaluate_detection_metrics.py  --detections outputs/schemas/pred_detection.csv  --ground-truth gt_detection.csv  --out-json detection.json
python scripts/evaluate_tracking_metrics.py   --tracked    outputs/schemas/pred_tracking.csv   --ground-truth gt_tracking.csv   --out-json tracking.json
python scripts/evaluate_trajectory_metrics.py --predicted  outputs/schemas/pred_trajectory.csv --ground-truth gt_trajectory.csv --out-json trajectory.json
python scripts/evaluate_ssm_metrics.py        --predicted  outputs/schemas/pred_ssm.csv        --ground-truth gt_ssm.csv        --out-json ssm.json

# 4. Assemble the 15-row gold-standard table
python scripts/gold_standard_report.py \
    --detection-metrics  detection.json \
    --tracking-metrics   tracking.json \
    --trajectory-metrics trajectory.json \
    --ssm-metrics        ssm.json \
    --out-md outputs/validation_report.md
```

For a self-contained smoke test on synthetic data:

```bash
python scripts/demo_end_to_end.py
```

The demo and the test suite verify that the metric implementations are
correct. They do not measure NNDS accuracy. Real accuracy numbers
require real ground truth. See docs/VALIDATION.md.
