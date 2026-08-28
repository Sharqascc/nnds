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


### Chosen Tracking Threshold
We select `max_gap=5` frames and `max_jump=30` pixels as our primary operating point. This choice balances identity preservation and fragmentation: stricter settings over-split short tracks, while looser settings merge distinct vehicles, distorting PET. Our sensitivity analysis (Table in docs) shows this configuration provides the highest PET event count while keeping median PET stable at 1.017 s.
### Note on Track Splitting & Occlusion Handling

The track splitter uses a `prediction_tolerance` parameter to avoid splitting tracks during short, predictable occlusions. When an object reappears after a gap (even >15 frames) but its position closely matches a linear motion prediction, the track is intentionally kept intact to reduce fragmentation and preserve identity continuity. This behavior was verified on the 300‑frame validation sequence: tracks with gaps in the 16–20 frame range remained unsplit because their reappearance was consistent with expected motion. Disabling the prediction tolerance (`prediction_tolerance=0`) does split those tracks, confirming the parameter is active. We keep the default tolerance as a balance between identity purity and fragmentation.

For full details, see the sensitivity scripts and `docs/final_submission_summary.md`.

### BEV Calibration Provenance
The six calibration points were surveyed under a planar ground assumption; therefore, a planar homography fits them with very low error. This low error reflects self-consistency of the planarity assumption, not independently validated metric accuracy. Field validation against independently measured distances is recommended.

## Optional: Time-of-Day Estimation (Future Work)

The pipeline includes an **optional** VLM-based time-of-day estimator (`scripts/estimate_time_of_day.py`).
It samples frames and uses a vision-language model to classify the scene as `morning`, `evening`, or `unknown`.

This feature is **not part of the core validated pipeline** and is not required for any results in the paper.
All reported metrics use manually specified time labels (`--time-of-day`).
To enable the experimental VLM estimator, install the optional dependencies:

```bash
pip install -r requirements-vlm.txt
```

If the VLM dependencies are not installed, the estimator returns `"unknown"` and the pipeline continues normally.

## Optional Event Utilities

- `scripts/generate_event_descriptions.py` – deterministic natural language descriptions for each PET event.

- `scripts/extract_event_frames.py` – save before/closest/after frames for manual inspection or VLM audits.

- `scripts/generate_safety_report_groq.py` – optional LLM-based report generation (requires `GROQ_API_KEY`). Not part of core results.

## ⚠️ VLM Utilities – Optional & Experimental

The repository includes optional Vision‑Language Model (VLM) utilities for:

- Time‑of‑day estimation (`scripts/estimate_time_of_day.py`)
- Qualitative scene description (`src/vlm/analyzer.py`)
- Experimental conflict‑type classification (`scripts/classify_conflict_type_vlm.py`)

These are **not used in the main quantitative analysis** of the paper.
All reported PET, conflict type, gate entry, and sensitivity metrics are deterministic and derived from trajectory geometry, not from VLM output.

Preliminary tests showed that small VQA models (e.g., BLIP) cannot reliably distinguish traffic conflict types from static frames, often returning generic labels. Therefore, **conflict type is always computed geometrically** (`src/analysis/conflict_classifier.py`).

To install optional VLM dependencies (if desired):
```bash
pip install -r requirements-vlm.txt
```

## ⚠️ Calibration & Units

- World coordinates in `configs/` are in **US survey feet**.
- The calibration region is a synthetic **20 ft × 16 ft** rectangle.
- BEV visualizations may show local metres after conversion (`1 ft = 0.3048 m`).
- The reported condition number and reprojection residual document **numerical conditioning and internal consistency**, not independent field accuracy.
- See `docs/calibration_provenance.md` for full provenance.
