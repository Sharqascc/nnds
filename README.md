# 🚦 NNDS: Neural Network for Dynamic Safety

[![CI](https://github.com/Sharqascc/nnds/actions/workflows/ci.yml/badge.svg)](https://github.com/Sharqascc/nnds/actions/workflows/ci.yml)

NNDS is a comprehensive traffic safety analysis pipeline that detects vehicles and pedestrians, tracks them in 3D space, computes safety metrics such as PET, TTC, and DRAC, and generates bird's-eye-view (BEV) visualizations.

## Installation

```bash
git clone [https://github.com/Sharqascc/nnds.git](https://github.com/Sharqascc/nnds.git)
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
| GITI traffic video | `data/sample_data/GITI_traffic_video.mp4` | Primary GITI-intersection input |
| Anonymized fixture | `data/sample_data/anonymized_traffic_video_50f.mp4` | Fast smoke-test input |
| UVH detector | `data/models/uvh26.pt` | UVH vehicle detector |
| COCO person model | `data/models/yolo11n.pt` | Person detection |
| BEV configuration | `configs/bev_config.json` | Bird's-eye-view mapping |
| GITI grid configuration | `configs/GITI_grid_config.json` | GITI grid mapping |
