# 🚦 NNDS – Neural Network for Driving Safety
[![CI](https://github.com/Sharqascc/nnds/actions/workflows/ci.yml/badge.svg)](https://github.com/Sharqascc/nnds/actions/workflows/ci.yml)

# NNDS: Neural Network for Dynamic Safety

NNDS is a traffic safety analysis pipeline that detects vehicles/pedestrians, tracks them, computes safety metrics (PET, TTC, DRAC), and generates bird's-eye-view visualizations.

## Installation

```bash
git clone https://github.com/Sharqascc/nnds.git
cd nnds
pip install -r requirements.txt
```

For a reproducible environment, use:
```bash
conda env create -f environment.yml
conda activate nnds
```

## Download Models

The required models are automatically downloaded and placed in `data/models/` when using the wrapper script. Alternatively, run:

```bash
bash scripts/download_models.sh
```

This script downloads:
- UVH-26 model from Hugging Face → `data/models/uvh26.pt`
- YOLO11n model from Ultralytics → `data/models/yolo11n.pt`

## Quick Start

### Option 1: Using the Wrapper (Recommended)

```bash
python scripts/run_pipeline.py \
    --video data/sample_data/traffic_video.mp4 \
    --detector yolo-cpu \
    --out-csv outputs/pet_events.csv
```

The wrapper automatically prepares models (including OpenVINO export) and sets the correct Python path.

### Option 2: Direct Script

If you prefer to call the pipeline directly, ensure the project root is in `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/nnds:$PYTHONPATH
python src/pipeline/traffic_analyzer.py \
    --video data/sample_data/traffic_video.mp4 \
    --detector yolo-cpu \
    --yolo-weights data/models/yolo11n.pt \
    --out-csv outputs/pet_events.csv
```

### Other Detectors

- `--detector sam3` (default, requires SAM3 weights)
- `--detector rtdetr` (experimental)
- `--detector uvh-coco-fused` (requires UVH + COCO person model)

## Gate Line Calibration

The gate line defined in `configs/gate_config.yaml` determines where PET events are counted. The default horizontal line may not intersect the traffic in your video. To calibrate:

1. Inspect the detection CSV (e.g., `outputs/*_detections.csv`) to see typical x,y coordinates of objects.
2. Edit `configs/gate_config.yaml` and set `start` and `end` points to form a line crossing the object paths.
   - For horizontal traffic (objects move left/right), use a vertical line (e.g., `start: [x, 0], end: [x, height]`).
   - For vertical traffic, use a horizontal line (e.g., `start: [0, y], end: [width, y]`).
3. Re-run the pipeline.

## Input & Output

- **Input**: Video file (MP4, AVI, etc.)
- **Output**:
  - `outputs/*_detections.csv` – raw object detections
  - `outputs/*.csv` – PET events (post‑encroachment time)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{nnds2025,
  author = {Your Name},
  title = {NNDS: Neural Network for Dynamic Safety},
  year = {2025},
  url = {https://github.com/Sharqascc/nnds}
}
```

## License

See [LICENSE](LICENSE) for details.

## Validation Status

- **BEV Homography:** Hartley normalized condition number 1.71, held-out reprojection error ~0.000001 ft.
- **Pipeline:** 300-frame end-to-end validation passes automatically in CI.
- **PET Analysis:** At our chosen operating point (max_gap=5, max_jump=30), the pipeline detects 156 PET events with median PET 1.017 s (Shapiro-Wilk p=0.0007, non-normal). A deconfounded sensitivity analysis shows PET metrics vary moderately across tracking thresholds (events 156→127; median PET 1.017→1.133 s). We report the full sensitivity table in `docs/sensitivity_deconfounded.tex` and discuss this trade-off in the manuscript.
- **Detection/Tracking Metrics:** Ground truth required for standard MOTA/IDF1/mAP. We explicitly document this limitation and provide scripts for when GT becomes available.
- **Sensitivity:** Track fragmentation impact on PET is quantified via `scripts/sensitivity_pet_fragmentation.py`.

