
## Requirements & Compatibility

### Compatibility Matrix

| Component          | Python      | CUDA      | torch        |
|--------------------|-------------|-----------|--------------|
| SAM3 pipeline      | 3.8–3.10    | 11.7+     | 2.0+         |
| Diffusion model    | 3.9–3.11    | 11.8+     | 2.0+         |
| Visualization      | 3.8–3.11    | N/A       | N/A          |

### Example `requirements.txt`

Pin versions for reproducibility:

```txt
torch>=2.0.0,<2.1.0
opencv-python>=4.8.0,<4.9.0
numpy>=1.24.0,<1.26.0
supervision>=0.10.0
```

Adjust to match your actual dependencies.

## Quick Start

### One-Click Colab Setup

Paste this into a fresh Colab cell:

```python
# NNDS Colab bootstrap: clone, install, download demo video + SAM3
import os
import sys
import subprocess
from pathlib import Path

# 1) Clone or update repo on main branch
os.chdir("/content")
repo_dir = Path("nnds")

if repo_dir.exists():
    os.chdir(repo_dir)
    try:
        subprocess.run(["git", "fetch"], check=True, capture_output=True)
        subprocess.run(["git", "checkout", "main"], check=True, capture_output=True)
        subprocess.run(["git", "pull"], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Git fetch/pull failed: {e.stderr.decode()}")
        raise
else:
    subprocess.run(["git", "clone", "https://github.com/Sharqascc/nnds.git", "nnds"], check=True)
    os.chdir(repo_dir)
    subprocess.run(["git", "checkout", "main"], check=True)

# 2) Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

# 3) Download demo video
os.makedirs("videos", exist_ok=True)
video_path = Path("videos/traffic_video.mp4")
if not video_path.exists():
    subprocess.run(
        [
            "wget",
            "-O",
            str(video_path),
            "https://huggingface.co/datasets/sharqascc/traffic-video-dataset/resolve/main/videos/traffic_video.mp4",
        ],
        check=True,
    )

# 4) Download SAM3 weights
sam3_path = Path("sam3.pt")
if not sam3_path.exists():
    subprocess.run(
        [
            "wget",
            "-O",
            str(sam3_path),
            "https://huggingface.co/sharqascc/sam3-traffic-model/resolve/main/sam3.pt",
        ],
        check=True,
    )

print("✅ Setup complete!")
```

### Run the Pipeline

```bash
# Full pipeline
PYTHONPATH=. python traffic_analyzer.py --video videos/traffic_video.mp4

# With frame limit for debugging
PYTHONPATH=. python traffic_analyzer.py \
    --video videos/traffic_video.mp4 \
    --out-csv outputs/petevents_bev_test.csv \
    --pet-threshold 2.0 \
    --max-frames 30
```

## Installation

### Local Development

```bash
git clone https://github.com/Sharqascc/nnds.git
cd nnds
pip install -r requirements.txt
```

### Using Make

```bash
make install      # pip install -r requirements.txt
make grid         # PYTHONPATH=. python traffic_analyzer.py --video videos/traffic_video.mp4
make test         # pytest tests/ -v
```

## Usage Guide

### PET Extraction Only

```bash
PYTHONPATH=. python traffic_analyzer.py \
    --video videos/traffic_video.mp4 \
    --out-csv outputs/petevents_bev.csv \
    --pet-threshold 2.0
```

### PET Summary & Analysis

```bash
# Basic statistics
PYTHONPATH=. python analysis/pet_summary.py \
    --csv-path outputs/petevents_bev.csv

# With risk thresholds and export
PYTHONPATH=. python analysis/pet_summary.py \
    --csv-path outputs/petevents_bev.csv \
    --critical 1.0 --moderate 3.0 \
    --export --output-dir analysis_results/
```

### Research Workflow (Orchestrated)

```bash
# Full research pipeline
PYTHONPATH=. python analysis/research_run.py \
    --video videos/traffic_video.mp4 \
    --train-diffusion --eval-diffusion

# Resume from existing CSV
PYTHONPATH=. python analysis/research_run.py \
    --video videos/traffic_video.mp4 \
    --skip-extraction --train-diffusion

# Dry run (test without executing)
PYTHONPATH=. python analysis/research_run.py \
    --video videos/traffic_video.mp4 --dry-run
```

## Pipeline Components

### Phase 1: Video Input & Preprocessing

| Component     | File                         | Description                                  |
|--------------|------------------------------|----------------------------------------------|
| Input video  | `videos/`                    | Raw traffic video (e.g., `traffic_video.mp4`) |
| Demo video   | HF dataset                   | `sharqascc/traffic-video-dataset`            |
| Frame limiting | `--max-frames`             | Debug / fast-test mode                       |
| Trajectory parser | `analysis/trajectory_parser.py` | Parses raw trajectories into analysis-ready form |

### Phase 2: Segmentation & Tracking (SAM3 / YOLO26)

| Component          | File                               | Description                                       |
|-------------------|------------------------------------|---------------------------------------------------|
| SAM3 weights       | `sam3.pt`                         | SAM3 model (downloaded from Hugging Face)         |
| SAM3 grid-PET      | `grid_trajectory/sam3_grid_pet.py`| SAM3 → grid → PET pipeline                        |
| YOLO26seg grid-PET | `grid_trajectory/yolo26seg_grid_pet.py` | YOLOv8‑26 segmentation → grid → PET (experimental) |
| Contact-point pipeline | `experimental/contact_point_pipeline.py` | Experimental contact-point projection into world coordinates |
| Local motion checker | `experimental/check_local_motion.py` | Tools for inspecting local motion and calibration |

### Phase 3: BEV Transformation & Calibration

| Component           | File                 | Description                             |
|--------------------|----------------------|-----------------------------------------|
| Homography calibration | `giti_bev_calib.py` | Camera-to-world homography estimation   |
| BEV mapper         | `bev_mapper.py`      | Image-plane to world coordinates        |
| World coordinates  | —                    | Outputs (t, x, y) trajectories in meters |

### Phase 4: Grid Mapping & Trajectory Construction

| Component         | File                           | Description                              |
|------------------|--------------------------------|------------------------------------------|
| Spatial grid     | `grid_trajectory/spatial_grid.py` | Intersection grid zones                 |
| PET grid logic   | `grid_trajectory/pet_grid.py`  | Grid cell assignments and PET logic      |
| Trajectory dataset | `traffic_diffusion/data/`    | `trajdiff_*.npy`, `*.parquet` for diffusion training |

### Phase 5: PET Conflict Extraction

| Component          | File                     | Description                              |
|-------------------|--------------------------|------------------------------------------|
| End-to-end pipeline | `traffic_analyzer.py`   | SAM3 / YOLO26seg → BEV → grid → PET      |
| PET computation   | `grid_trajectory/`       | Post Encroachment Time logic             |
| Conflict detection | `pet_conflict_checker.py`| Conflict classification                  |
| Output CSV        | `outputs/petevents_bev.csv` | Events with PET, trajectories           |
| Gate counter      | `gate_counter.py`        | Actor counting through gates             |

### Phase 6: Analysis & Visualization

| Component                  | File                                         | Description                        |
|---------------------------|----------------------------------------------|------------------------------------|
| PET summary               | `analysis/pet_summary.py`                    | Statistics, percentiles, risk      |
| SSM verification          | `analysis/ssm/ssm_verification.py`           | SSM validation framework           |
| Uncertainty quantification| `analysis/ssm/uncertainty_quantifier.py`     | Error and uncertainty analysis     |
| Statistical testing       | `analysis/verification/statistical_testing.py`| Hypothesis tests                  |
| Reproducibility audit     | `analysis/logging/reproducibility_audit.py`  | Run tracking (if present)          |
| Research runner           | `analysis/research_run.py`                   | Orchestrated workflow              |

### Phase 7: Diffusion-Based Trajectory Modeling

| Component        | File                                  | Description                                     |
|-----------------|---------------------------------------|-------------------------------------------------|
| Diffusion model | `traffic_diffusion/trajectory_diffusion.py` | Conditional trajectory diffusion model     |
| Training script | `traffic_diffusion/train_trajectory_diffusion.py` | Train on PET events                    |
| Model & sampler | `traffic_diffusion/model_and_sampler.py` | Checkpoint + sampling utilities              |
| Training utils  | `traffic_diffusion/training_utils.py` | Data loaders and training loops               |
| Sampling utils  | `traffic_diffusion/sampling_utils.py` | Counterfactual futures generation             |
| PET safety metrics | `traffic_diffusion/pet_safety_metrics.py` | PET/TTC from diffusion trajectories     |
| Episode reward  | `traffic_diffusion/episode_reward.py` | Episode-level reward and safety metrics        |
| PET diffusion analysis | `analysis/pet_diffusion_analysis.py` | Real vs generated PET comparison         |

## Visualization Suite

### Visualization Components

| Module             | File                                         | Exports | Description                           |
|--------------------|----------------------------------------------|---------|---------------------------------------|
| SSM Analysis       | `analysis/visualization/industry_standard_viz.py` | 10      | Distributions, time series, heatmaps  |
| Diffusion Evaluation | `analysis/visualization/pet_diffusion_plots.py` | 6       | PET-like metrics, residuals           |
| Conflict Events    | `analysis/visualization/pet_event_plots.py`  | 7       | Individual conflict visualization     |
| Video Overlays     | `analysis/visualization/video_overlays.py`   | 5       | Frame overlays, MP4 generation        |

### Quick Visualization Examples

```python
from analysis.visualization.industry_standard_viz import plot_pet_distribution
from analysis.visualization.pet_event_plots import plot_conflict_event
from analysis.visualization.video_overlays import generate_conflict_video
from analysis.visualization.pet_diffusion_plots import DiffusionPETPlotter

# PET distribution
fig = plot_pet_distribution(df['pet'].values, style='journal')

# Single conflict event
plot_conflict_event(df, event_id=5, show_velocities=True)

# Generate conflict video
generate_conflict_video(
    video_path='videos/traffic_video.mp4',
    frame_range=(1200, 1300),
    trajectories=trajectories,
    output_path='conflict_video.mp4'
)

# Complete diffusion evaluation suite
plotter = DiffusionPETPlotter(dpi=300)
plotter.plot_all(pet_pairs, records, out_dir='diffusion_eval/')
```

### Visualization Standards

- Resolution: 300 DPI minimum for publication  
- Font: 10–12pt serif (Times New Roman / Computer Modern)  
- Color: Colorblind-safe palettes (Okabe-Ito, Viridis)  
- Formats: PNG (raster), PDF/SVG (vector for LaTeX)  
- Dimensions: Single-column (3.375") or double-column (6.875")

## Diffusion-Based Modeling

### Training

```bash
PYTHONPATH=. python traffic_diffusion/train_trajectory_diffusion.py \
    --csv-path outputs/petevents_bev.csv \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --device cuda \
    --save-dir checkpoints/
```

### Safety Evaluation

```bash
# Batch PET/TTC evaluation
PYTHONPATH=. python analysis/safety_eval_diffusion.py

# Notebook-friendly pipeline
PYTHONPATH=. python analysis/safety_eval_diffusion_notebook.py
```

### Outputs

- `outputs/safety_events_diffusion_model.csv` – Sampled futures with PET/TTC  
- `outputs/safety_eval_diffusion_summary.csv` – Aggregated statistics  

## SSM Verification & Validation Methodology

### Surrogate Safety Measures

| Metric | Description                                      | Safety Threshold             | Reference                |
|--------|--------------------------------------------------|-----------------------------|--------------------------|
| PET    | Time difference between leaving/entering conflict zone | < 1.5s: critical, < 3.0s: potential | Allen et al. (1977) |
| TTC    | Time to collision if trajectories maintained     | < 2.0s: critical, < 5.0s: potential | Hyd’en (1987)        |
| DRAC   | Deceleration rate to avoid collision             | > 3.37 m/s²: unsafe          | NHTSA guidelines        |

### Mathematical Formulation

\[
\text{PET}_{i,j} = \min_t \left| t_j^{\text{enter}} - t_i^{\text{exit}} \right|
\]

\[
\text{TTC}_{i,j}(t) = \frac{\lVert \mathbf{x}_i(t) - \mathbf{x}_j(t) \rVert}{\lVert \mathbf{v}_i(t) - \mathbf{v}_j(t) \rVert}
\]

### Error Propagation

| Error Source     | Typical Magnitude | Impact on PET  |
|------------------|-------------------|----------------|
| Detection (ε_d)  | 0.1–0.3 m         | ±0.05 s        |
| Homography (ε_h) | 0.2–0.5 m         | ±0.10 s        |
| Tracking (ε_t)   | 0.05–0.15 m       | ±0.02 s        |
| Total            | 0.25–0.60 m       | ±0.12 s        |

### Validation Protocol

Three-tier validation per FHWA SSAM framework:

1. Theoretical validation – mathematical correctness  
2. Simulation validation – against VISSIM, SUMO  
3. Field validation – correlation with crash data  

## Output Directory Structure

```text
outputs/
├── petevents_bev.csv                     # Main PET events
├── safety_events_diffusion_model.csv     # Sampled futures with PET/TTC
├── trajectories/
│   └── world_trajectories.npy            # BEV trajectories
└── visualizations/
    ├── plots/                            # PDF/PNG figures
    └── videos/                           # Annotated MP4 clips
```

## PET CSV Schema

| Column        | Type     | Nullable | Description                           | Value Range                |
|--------------|----------|----------|---------------------------------------|----------------------------|
| `event_id`   | int      | No       | Integer conflict index                | 1, 2, 3, …               |
| `pet`        | float    | Yes      | Post-Encroachment Time (seconds)      | [0, ∞), NaN if invalid   |
| `frame`      | int      | Yes      | Frame index                           | 0, 1, 2, …, NaN if unknown |
| `track_a`    | int      | No       | Track ID of first actor               | ≥1                       |
| `track_b`    | int      | No       | Track ID of second actor              | ≥1                       |
| `conflict_type` | str    | No       | Grid cell ID                          | e.g. `CELL_C_1`          |
| `world_traj_i` | str    | No       | BEV trajectory for actor a            | Serialized JSON/array    |
| `world_traj_j` | str    | No       | BEV trajectory for actor b            | Serialized JSON/array    |

## Common Issues & Solutions

| Issue                                | Solution                                                                                |
|-------------------------------------|-----------------------------------------------------------------------------------------|
| CUDA out of memory                   | Reduce batch size or use `--max-frames 100`                                            |
| Homography calibration fails         | Ensure at least 4 correspondence points in `giti_bev_calib.py`                          |
| Empty PET CSV output                 | Check `--pet-threshold` is reasonable for your scene                                    |
| Video not found                      | Verify `videos/` directory and file permissions                                         |
| Import errors in visualization       | Use explicit imports as shown in `Visualization Example` section                        |
| Training is too slow                 | Enable GPU via `CUDA_VISIBLE_DEVICES=0` and ensure `torch` uses CUDA                    |

## Environment Variables

| Variable             | Purpose                          | Default     |
|----------------------|----------------------------------|-------------|
| `PYTHONPATH`         | Project root for imports         | `.`         |
| `CUDA_VISIBLE_DEVICES` | GPU selection for diffusion     | `0`         |
| `NNDS_DATA_DIR`      | Override data directory          | `./data/`   |

## Project Structure

```text
nnds/
├── analysis/
│   ├── __init__.py
│   ├── pet_diffusion_analysis.py
│   ├── pet_summary.py
│   ├── research_run.py
│   ├── safety_eval_diffusion_notebook.py
│   ├── safety_eval_diffusion.py
│   ├── trajectory_parser.py
│   ├── ssm/
│   │   ├── __init__.py
│   │   ├── ssm_verification.py
│   │   └── uncertainty_quantifier.py
│   ├── verification/
│   │   ├── __init__.py
│   │   └── statistical_testing.py
│   └── visualization/
│       ├── __init__.py
│       ├── industry_standard_viz.py
│       ├── pet_diffusion_plots.py
│       ├── pet_event_plots.py
│       └── video_overlays.py
├── experimental/
│   ├── check_local_motion.py
│   └── contact_point_pipeline.py
├── grid_trajectory/
│   ├── __init__.py
│   ├── pet_grid.py
│   ├── sam3_grid_pet.py
│   ├── spatial_grid.py
│   └── yolo26seg_grid_pet.py
├── traffic_diffusion/
│   ├── __init__.py
│   ├── data/
│   │   ├── trajdiff_inputs.npy
│   │   ├── trajdiff_meta.parquet
│   │   └── trajdiff_targets.npy
│   ├── episode_reward.py
│   ├── model_and_sampler.py
│   ├── pet_safety_metrics.py
│   ├── sampling_utils.py
│   ├── train_trajectory_diffusion.py
│   ├── training_utils.py
│   └── trajectory_diffusion.py
└── ...
```

## Colab Setup (Recommended)

### One-Cell Bootstrap

The bootstrap script above handles:

- Cloning/updating the repository on `main`  
- Installing Python dependencies  
- Downloading demo video and SAM3 weights  

### Running in Colab

```python
# After bootstrap, run pipeline
!cd /content/nnds && PYTHONPATH=. python traffic_analyzer.py \
    --video videos/traffic_video.mp4

# With frame limit for quick testing
!cd /content/nnds && PYTHONPATH=. python traffic_analyzer.py \
    --video videos/traffic_video.mp4 \
    --max-frames 30 \
    --out-csv outputs/petevents_bev_test.csv

# Summarize results
!cd /content/nnds && PYTHONPATH=. python analysis/pet_summary.py \
    --csv-path outputs/petevents_bev_test.csv \
    --export --output-dir results/
```

## Development

### Running Tests

```bash
# Smoke tests
pytest tests/ -v

# Import tests
PYTHONPATH=. python tests/test_imports_smoke.py
```

### Code Quality

```bash
# Format code
black .

# Type checking (if using mypy)
mypy --ignore-missing-imports .
```

## References

- Allen, B. L., Shin, B. T., & Cooper, P. J. (1977). Analysis of traffic conflicts and collisions. *Transportation Research Record*, 667, 67–74.  
- Gettman, D., & Head, L. (2003). Surrogate safety measures from traffic simulation models. FHWA-RD-03-050.  
- Hyd’en, C. (1987). The Swedish Traffic Conflicts Technique. *Bulletin Lund Institute of Technology*, 70.  
- Zheng, L., Ismail, K., & Meng, X. (2014). Traffic conflict techniques for road safety analysis. *Accident Analysis & Prevention*.  
- FHWA. (2008). *Surrogate Safety Assessment Model and Validation*. FHWA-HRT-08-051.  
- Hayward, J. C. (1972). Near-miss determination through use of a scale of danger.  
- Vogel, K. (2003). A comparison of headway and time to collision as safety indicators.  
- Archer, J. (2005). Indicators for traffic safety assessment methods.  

## Acknowledgments

- SAM3 model from Meta AI  
- Hugging Face for model/dataset hosting  
- FHWA for SSAM methodology guidance  

<!-- AUTO-README-START -->
> The block below is auto-generated to keep high-level structure in sync with the repository.

## Auto-generated structure

- `experimental/contact_point_pipeline.py` — contact-point projection into world coordinates.
- `traffic_analyzer.py` — main video-to-PET entry point.
- `analysis/research_run.py` — orchestrated research workflow.

<!-- AUTO-README-END -->
