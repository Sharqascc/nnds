# 🚦 NNDS – Neural Network for Driving Safety

### Intersection Safety & Conflict Detection Pipeline

A **modular, research-ready pipeline** for real‑time detection and analysis of pedestrian‑vehicle interactions at intersections. Combines **UVH + YOLO detectors** with tracking, BEV mapping, and post‑event analysis.

---

## ✨ Features

- **UVH-COCO Fused Detector** – primary backend combining UVH‑26 and YOLO11 for robust pedestrian/vehicle detection
- **BEV Mapping** – bird's‑eye‑view transformation for accurate spatial reasoning
- **Grid‑based PET Event Extraction** – detects and logs pedestrian–vehicle conflict events
- **Diffusion‑based Trajectory Modelling** – for trajectory prediction and safety evaluation
- **VLM Integration (optional)** – vision‑language models for advanced analysis
- **Smoke tests** – lightweight tests to verify repo health

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Sharqascc/nnds.git
cd nnds
git checkout refactor/restructure   # or your preferred branch
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Python 3.10+ is recommended.

### 3. Download pretrained models

```bash
bash scripts/download_models.sh
```

This creates `data/models/` and downloads:

| Model | File | Purpose |
|-------|------|---------|
| UVH‑26 | `data/models/uvh26.pt` | Primary detection model |
| YOLO11n | `data/models/yolo11n.pt` | COCO person fallback detector |

### 4. Run the pipeline

A sample traffic video is included at `data/sample_data/traffic_video.mp4`. You can use it directly or replace it with your own:

```bash
python scripts/run_pipeline.py --video data/sample_data/traffic_video.mp4
```

The pipeline will process the video, detect objects, map to BEV, and produce:

- `outputs/petevents_bev_detections.csv` – all detections
- `outputs/petevents_bev.csv` – PET conflict events

> To process only the first N frames for a quick test, add `--max-frames N`.

---

## 🕹 Detector Backends

You can choose a detector with `--detector`:

| Detector | Description | Required models |
|----------|-------------|-----------------|
| `uvh-coco-fused` | **Default** – UVH + YOLO person fallback | `uvh26.pt`, `yolo11n.pt` |
| `yolo-cpu` | YOLO‑only, CPU‑friendly | `yolo11n.pt` |
| `sam3` | SAM3‑based segmentation (if weights available) | `sam3.pt` |
| `rtdetr` | Experimental RT‑DETR (not fully implemented) | `rtdetr-l.pt` |

Example:

```bash
python scripts/run_pipeline.py --video my_video.mp4 --detector yolo-cpu
```

---

## 🧪 Running Tests

```bash
pytest -q
```

All tests are designed to run without heavyweight models or GPU.

---

## ⚡ Performance Tips

The pipeline automatically chooses the fastest available backend:

| Backend | How to enable | Speed |
|---------|---------------|-------|
| **CUDA (GPU)** | `--device auto` (if GPU available) | Fastest |
| **OpenVINO (CPU)** | Export models once, then `--backend auto` or `--backend openvino` | ~2.3x faster than PyTorch CPU |
| **PyTorch CPU** | `--backend pytorch` | Slowest |

### Export models for OpenVINO

```bash
python scripts/export_openvino.py --uvh data/models/uvh26.pt --yolo data/models/yolo11n.pt
```

After export, the pipeline will use OpenVINO automatically when no GPU is present.

### Reduce image size for faster inference

Add `--imgsz 640` to trade a little accuracy for much faster processing:

```bash
python scripts/run_pipeline.py --video data/sample_data/traffic_video.mp4 --imgsz 640
```

### Show progress bar

The pipeline now displays a `tqdm` progress bar by default. To disable it, add `--no-progress`.

---
## 📊 PET Output Format

The pipeline writes `outputs/petevents_bev.csv` with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `event_id` | int | Sequential event ID |
| `pet` | float | Post‑Encroachment Time (seconds) |
| `frame` | int | Reference frame of the conflict |
| `track_a` | int | First track ID |
| `track_b` | int | Second track ID |
| `conflict_type` | string | Conflict category (`image_intersection`) |
| `grid_cell` | string | Grid cell where trajectories intersected |
| `track_a_entry_frame` | int | First frame that track A was inside the conflict zone |
| `track_a_exit_frame` | int | Last frame that track A was inside the conflict zone |
| `track_b_entry_frame` | int | First frame that track B was inside the conflict zone |
| `track_b_exit_frame` | int | Last frame that track B was inside the conflict zone |
| `world_traj_i` | string | Reference ID for first actor trajectory (`track_<id>`) |
| `world_traj_j` | string | Reference ID for second actor trajectory (`track_<id>`) |
| `traj_a_json` | JSON string | Full trajectory A with `frame`, `x_pixel`, `y_pixel`, `world_x`, `world_y` |
| `traj_b_json` | JSON string | Full trajectory B with `frame`, `x_pixel`, `y_pixel`, `world_x`, `world_y` |

To quickly review events, run:

```bash
python scripts/inspect_pet.py --csv outputs/petevents_bev.csv --top 10
```

## 📁 Repository Structure

```
nnds/
├── configs/                  # JSON/YAML configs (BEV, grid, gates)
├── data/
│   ├── models/               # Downloaded pretrained models (ignored by git)
│   └── sample_data/          # Optional sample videos (ignored)
├── outputs/                  # Generated CSVs and figures (ignored)
├── scripts/
│   ├── run_pipeline.py       # Main entry point
│   ├── download_models.sh    # Download required models
│   └── traffic_analyzer_demo.py
├── src/
│   ├── analysis/             # Tracking, grid, PET logic, analytics
│   ├── bev/                  # BEV mapping and calibration
│   ├── core/                 # Shared types and validation
│   ├── diffusion/            # Trajectory diffusion model and evaluation
│   ├── pipeline/             # CLI entry (traffic_analyzer.py)
│   ├── utils/                # Helpers
│   └── vlm/                  # Optional VLM integration
├── tests/                    # Smoke tests
├── Makefile                  # Convenience targets
├── requirements.txt
└── README.md
```

---

## 🛠 Debugging & Visualization

The repository includes several helper scripts to inspect tracking and PET events:

| Script | Purpose |
|--------|---------|
| `scripts/inspect_pet.py` | Print PET events with grid cell, entry/exit frames, and trajectory summaries |
| `scripts/visualize_pet.py` | Create an annotated video showing full trajectory history for one event |
| `scripts/visualize_pet_live.py` | Create a live tracking video with current markers and short trails |
| `scripts/debug_tracking_video.py` | Draw detection boxes with track IDs on the original video |
| `scripts/diagnose_tracking.py` | Analyze detections CSV for track gaps/jumps and flag suspicious tracks |

### Examples

```bash
python scripts/inspect_pet.py --csv outputs/petevents_bev.csv --top 10
python scripts/visualize_pet.py --csv outputs/petevents_bev.csv --event-id 0 --output outputs/event_0.mp4
python scripts/visualize_pet_live.py --csv outputs/petevents_bev.csv --event-id 0 --output outputs/event_0_live.mp4
python scripts/debug_tracking_video.py --csv outputs/petevents_bev_detections.csv --start 0 --end 150 --output outputs/debug.mp4
python scripts/diagnose_tracking.py --csv outputs/petevents_bev_detections.csv
```

## 🔧 Configuration

Key config files:

| File | Purpose |
|------|---------|
| `configs/bev_config.json` | Homography/ROI settings for BEV |
| `configs/GITI_grid_config.json` | Grid definition |
| `configs/gate_config.yaml` | Gate definitions for counting |
| `configs/giti_calibration_points.json` | Calibration points |

Adjust these to match your intersection geometry.

---

## 🧠 Advanced Usage

### Diffusion training & evaluation

```bash
python src/diffusion/traffic_diffusion/train_trajectory_diffusion.py
python src/analysis/analysis/safety_eval_diffusion.py
```

### VLM integration (optional)

See `src/vlm/` for Groq/Ollama/HuggingFace integrations. Requires additional API keys.

---

## 🏁 Tracking Performance

Final tracking results on the full sample video (1838 frames, 640px, OpenVINO CPU):

| Metric | Value |
|--------|-------|
| Unique tracks | 508 |
| PET events | 649 |
| ID switch candidates | 20 |
| Suspicious tracks | 321 |

The tracker uses a **custom Kalman + Hungarian tracker with appearance disambiguation** to handle overlapping objects and occlusions.

## 🤝 Contributing

Contributions are welcome! Please open an issue or pull request.

---

## 📄 License

MIT