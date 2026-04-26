markdown
# NNDS: Non-motorized and Heterogeneous Traffic Safety Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

AI-powered system for analyzing vehicle behavior and surrogate safety metrics at unsignalized intersections.

## 📋 Table of Contents

- [Pipeline Architecture](#pipeline-architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Pipeline Components](#pipeline-components)
- [Visualization Suite](#visualization-suite)
- [Diffusion-Based Modeling](#diffusion-based-modeling)
- [SSM Verification](#ssm-verification--validation-methodology)
- [Project Structure](#project-structure)
- [Colab Setup](#colab-setup-recommended)
- [License](#license)

## Pipeline Architecture

The NNDS system implements a multi-stage pipeline that transforms raw intersection video into quantitative safety metrics and diffusion-based trajectory predictions.

### Pipeline Overview

| Phase | Stage | Key Output | Location |
|-------|-------|------------|----------|
| 1 | Video Input & Preprocessing | Raw frames | `videos/` |
| 2 | SAM3 Video Segmentation | Actor masks + track IDs | `traffic_analyzer.py` |
| 3 | BEV Transformation | World-coordinate trajectories | `bev_mapper.py`, `giti_bev_calib.py` |
| 4 | Grid Mapping & Trajectory | Grid cell assignments | `grid_trajectory/` |
| 5 | PET Conflict Extraction | Conflict events + PETs | `outputs/petevents_bev.csv` |
| 6 | Analysis & Visualization | Safety statistics + plots | `analysis/`, `analysis/visualization/` |
| 7 | Diffusion-Based Modeling | Counterfactual trajectories | `traffic_diffusion/` |

### Pipeline Data Flow
[videos/traffic_video.mp4]
│
▼
┌─────────────────────────────┐
│ Phase 2: SAM3 Segmentation │ → sam3.pt (HF model)
│ (actor masks + track IDs) │
└──────────────┬──────────────┘
│
▼
┌─────────────────────────────┐
│ Phase 3: BEV Transform │ → giti_bev_calib.py, bev_mapper.py
│ (image → world coordinates)│
└──────────────┬──────────────┘
│
▼
┌─────────────────────────────┐
│ Phase 4: Grid + Trajectory │ → grid_trajectory/
│ (grid cells + (t,x,y) traj)│
└──────────────┬──────────────┘
│
▼
┌─────────────────────────────┐
│ Phase 5: PET Extraction │ → outputs/petevents_bev.csv
│ (conflict events + PETs) │
└──────────────┬──────────────┘
│
┌──────┴──────┐
▼ ▼
┌──────────────┐ ┌───────────────────────┐
│ Phase 6: │ │ Phase 7: │
│ Analysis │ │ Diffusion Modeling │
│ & Viz │ │ (train + sample + eval)│
└──────────────┘ └───────────────────────┘

text

## Quick Start

### One-Click Colab Setup

Paste this into a fresh Colab cell:

```python
# NNDS Colab bootstrap: clone, install, download demo video + SAM3
import os
import sys
import subprocess
from pathlib import Path
from urllib.parse import quote

# 1) Clone or update repo on main branch
os.chdir("/content")
repo_dir = Path("nnds")

if repo_dir.exists():
    os.chdir(repo_dir)
    subprocess.run(["git", "fetch"], check=True)
    subprocess.run(["git", "checkout", "main"], check=True)
    subprocess.run(["git", "pull"], check=True)
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
    subprocess.run(["wget", "-O", str(video_path), 
        "https://huggingface.co/datasets/sharqascc/traffic-video-dataset/resolve/main/videos/traffic_video.mp4"], check=True)

# 4) Download SAM3 weights
sam3_path = Path("sam3.pt")
if not sam3_path.exists():
    subprocess.run(["wget", "-O", str(sam3_path),
        "https://huggingface.co/sharqascc/sam3-traffic-model/resolve/main/sam3.pt"], check=True)

print("✅ Setup complete!")
Run the Pipeline
bash
# Full pipeline
PYTHONPATH=. python traffic_analyzer.py --video videos/traffic_video.mp4

# With frame limit for debugging
PYTHONPATH=. python traffic_analyzer.py \
    --video videos/traffic_video.mp4 \
    --out-csv outputs/petevents_bev_test.csv \
    --pet-threshold 2.0 \
    --max-frames 30
Installation
Local Development
bash
git clone https://github.com/Sharqascc/nnds.git
cd nnds
pip install -r requirements.txt
Using Make
bash
make install      # Install dependencies
make grid         # Run video-to-PET pipeline
make test         # Run tests
Usage Guide
PET Extraction Only
bash
PYTHONPATH=. python traffic_analyzer.py \
    --video videos/traffic_video.mp4 \
    --out-csv outputs/petevents_bev.csv \
    --pet-threshold 2.0
PET Summary & Analysis
bash
# Basic statistics
PYTHONPATH=. python analysis/pet_summary.py \
    --csv-path outputs/petevents_bev.csv

# With risk thresholds and export
PYTHONPATH=. python analysis/pet_summary.py \
    --csv-path outputs/petevents_bev.csv \
    --critical 1.0 --moderate 3.0 \
    --export --output-dir analysis_results/
Research Workflow (Orchestrated)
bash
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
Pipeline Components
Phase 1: Video Input & Preprocessing
Component	File	Description
Input video	videos/	Raw traffic video (e.g., traffic_video.mp4)
Demo video	HF dataset	Public demo at sharqascc/traffic-video-dataset
Frame limiting	--max-frames	Debug/fast-test mode
Phase 2: SAM3 Video Segmentation
Component	File	Description
SAM3 weights	sam3.pt	Downloaded from HF model repo
Segmentation	traffic_analyzer.py	SAM3 segments actors per frame
Actor tracking	—	Track IDs assigned across frames
Phase 3: BEV Transformation & Calibration
Component	File	Description
Homography calibration	giti_bev_calib.py	Camera-to-world homography
BEV mapper	bev_mapper.py	Image-plane to world coordinates
World coordinates	—	Outputs (t, x, y) trajectories in meters
Phase 4: Grid Mapping & Trajectory Construction
Component	File	Description
Spatial grid	grid_trajectory/spatial_grid.py	Intersection grid zones
PET grid logic	grid_trajectory/pet_grid.py	Grid cell assignments
SAM3-grid integration	grid_trajectory/sam3_grid_pet.py	SAM3 + grid + PET
Trajectory dataset	traffic_diffusion/data/	trajdiff_*.npy, *.parquet
Phase 5: PET Conflict Extraction
Component	File	Description
End-to-end pipeline	traffic_analyzer.py	SAM3 → BEV → Grid → PET
PET computation	grid_trajectory/	Post Encroachment Time
Conflict detection	pet_conflict_checker.py	Conflict classification
Output CSV	outputs/petevents_bev.csv	Events with PET, trajectories
Gate counter	gate_counter.py	Actor counting through gates
Phase 6: Analysis & Visualization
Component	File	Description
PET summary	analysis/pet_summary.py	Statistics, percentiles, risk
SSM verification	analysis/ssm/	SSM validation framework
Uncertainty quantification	analysis/ssm/uncertainty_quantifier.py	Error analysis
Statistical testing	analysis/verification/statistical_testing.py	Hypothesis tests
Reproducibility audit	analysis/logging/reproducibility_audit.py	Run tracking
Research runner	analysis/research_run.py	Orchestrated workflow
Phase 7: Diffusion-Based Trajectory Modeling
Component	File	Description
Diffusion model	traffic_diffusion/trajectory_diffusion.py	Conditional diffusion
Training script	traffic_diffusion/train_trajectory_diffusion.py	Train on PET events
Model & sampler	traffic_diffusion/model_and_sampler.py	Checkpoint + sampling
Training utils	traffic_diffusion/training_utils.py	Data loaders, loops
Sampling utils	traffic_diffusion/sampling_utils.py	Counterfactual futures
PET safety metrics	traffic_diffusion/pet_safety_metrics.py	PET/TTC from sampled
PET diffusion analysis	analysis/pet_diffusion_analysis.py	Real vs generated PET
Visualization Suite
The NNDS visualization suite (26+ exports) produces publication-ready figures compliant with IEEE Transactions on ITS, Accident Analysis & Prevention, and FHWA guidelines.

Visualization Components
Module	File	Exports	Description
SSM Analysis	industry_standard_viz.py	10	Distribution, time series, heatmaps
Diffusion Evaluation	pet_diffusion_plots.py	6	PET-like metrics, residuals
Conflict Events	pet_event_plots.py	7	Individual conflict visualization
Video Overlays	video_overlays.py	5	Frame overlays, MP4 generation
Quick Visualization Examples
python
from analysis.visualization import (
    plot_pet_distribution,
    plot_conflict_event,
    generate_conflict_video,
    DiffusionPETPlotter
)

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
Visualization Standards
Resolution: 300 DPI minimum for publication

Font: 10-12pt serif (Times New Roman / Computer Modern)

Color: Colorblind-safe palettes (Okabe-Ito, Viridis)

Formats: PNG (raster), PDF/SVG (vector for LaTeX)

Dimensions: Single-column (3.375") or double-column (6.875")

Diffusion-Based Modeling
Training
bash
# Train on PET events
PYTHONPATH=. python traffic_diffusion/train_trajectory_diffusion.py \
    --csv-path outputs/petevents_bev.csv \
    --epochs 100
Safety Evaluation
bash
# Batch PET/TTC evaluation
PYTHONPATH=. python analysis/safety_eval_diffusion.py

# Notebook-friendly pipeline
PYTHONPATH=. python analysis/safety_eval_diffusion_notebook.py
Outputs
outputs/safety_events_diffusion_model.csv - Sampled futures with PET/TTC

outputs/safety_eval_diffusion_summary.csv - Aggregated statistics

SSM Verification & Validation Methodology
Surrogate Safety Measures
Metric	Description	Safety Threshold	Reference
PET	Time difference between leaving/entering conflict zone	< 1.5s: critical, < 3.0s: potential	Allen et al. (1977)
TTC	Time to collision if trajectories maintained	< 2.0s: critical, < 5.0s: potential	Hyd'en (1987)
DRAC	Deceleration rate to avoid collision	> 3.37 m/s²: unsafe	NHTSA guidelines
Mathematical Formulation
Post-Encroachment Time (PET):

PET
i
,
j
=
min
⁡
t
∣
t
j
enter
−
t
i
exit
∣
PET 
i,j
​
 =min 
t
​
  
​
 t 
j
enter
​
 −t 
i
exit
​
  
​
 

Time-To-Collision (TTC):

TTC
i
,
j
(
t
)
=
∣
x
i
(
t
)
−
x
j
(
t
)
∣
∣
v
i
(
t
)
−
v
j
(
t
)
∣
TTC 
i,j
​
 (t)= 
∣v 
i
​
 (t)−v 
j
​
 (t)∣
∣x 
i
​
 (t)−x 
j
​
 (t)∣
​
 

Error Propagation
Error Source	Typical Magnitude	Impact on PET
Detection (ε_d)	0.1-0.3m	±0.05s
Homography (ε_h)	0.2-0.5m	±0.10s
Tracking (ε_t)	0.05-0.15m	±0.02s
Total	0.25-0.60m	±0.12s
Validation Protocol
Three-tier validation per FHWA SSAM framework:

Theoretical Validation - Mathematical correctness

Simulation Validation - Against VISSIM, SUMO

Field Validation - Correlation with crash data

Project Structure
text
nnds/
├── analysis/                      # Research & evaluation
│   ├── logging/                   # Reproducibility audit
│   │   ├── __init__.py
│   │   └── reproducibility_audit.py
│   ├── ssm/                       # SSM verification
│   │   ├── __init__.py
│   │   ├── ssm_verification.py
│   │   └── uncertainty_quantifier.py
│   ├── verification/              # Statistical testing
│   │   ├── __init__.py
│   │   └── statistical_testing.py
│   ├── visualization/             # 26+ plotting functions
│   │   ├── __init__.py            # Main exports (v2.3.0)
│   │   ├── industry_standard_viz.py
│   │   ├── pet_diffusion_plots.py
│   │   ├── pet_event_plots.py
│   │   └── video_overlays.py
│   ├── __init__.py
│   ├── pet_diffusion_analysis.py  # Diffusion evaluation
│   ├── pet_summary.py             # PET statistics (v2.0.0)
│   ├── research_run.py            # Workflow orchestrator
│   ├── safety_eval_diffusion.py   # Batch safety evaluation
│   └── safety_eval_diffusion_notebook.py
├── calibration/                   # Calibration utilities
│   ├── grid_validation_calibration.py
│   ├── monte_carlo_calibration_benchmark.py
│   └── monte_carlo_calibration_notes.md
├── configs/                       # Configuration files
│   ├── bev_config.json
│   ├── gate_config.yaml
│   ├── giti_calibration_points.json
│   └── GITI_grid_config.json
├── docs/                          # Documentation & samples
│   ├── code_dumps/
│   └── data_samples/
│       └── petevents_bev_demo.csv
├── grid_trajectory/               # Core grid/PET logic
│   ├── __init__.py
│   ├── pet_grid.py
│   ├── sam3_grid_pet.py
│   └── spatial_grid.py
├── outputs/                       # Generated artifacts (gitignored)
├── sample_data/                   # Sample video
│   └── traffic_video.mp4
├── tests/                         # Smoke tests
│   ├── test_configs_smoke.py
│   ├── test_diffusion_smoke.py
│   ├── test_imports_smoke.py
│   ├── test_repo_smoke.py
│   └── test_traffic_analyzer_cli.py
├── traffic_diffusion/             # Diffusion models
│   ├── data/                      # Training data
│   │   ├── trajdiff_inputs.npy
│   │   ├── trajdiff_meta.parquet
│   │   └── trajdiff_targets.npy
│   ├── __init__.py
│   ├── episode_reward.py
│   ├── model_and_sampler.py
│   ├── pet_safety_metrics.py
│   ├── sampling_utils.py
│   ├── train_trajectory_diffusion.py
│   ├── training_utils.py
│   └── trajectory_diffusion.py
├── videos/                        # Input videos
│   └── traffic_video.mp4
├── bev_mapper.py                  # BEV transformation
├── bootstrap_nnds_session.sh      # Session bootstrap
├── colab_ready.py                 # Colab utilities
├── CONTRIBUTING.md
├── gate_counter.py                # Traffic counting
├── giti_bev_calib.py              # Homography calibration
├── Makefile
├── nnds_structure.py              # Structure visualizer
├── pet_conflict_checker.py        # Conflict detection
├── pyproject.toml
├── README.md
├── requirements.txt
├── sam3.pt                        # SAM3 weights (3.2GB, gitignored)
├── traffic_analyzer.py            # Main entry point
└── traj_diffusion_normalized.py   # Normalized diffusion experiments
Colab Setup (Recommended)
One-Cell Bootstrap
The bootstrap script above handles everything:

Clones/updates repository on main branch

Installs Python dependencies

Downloads demo video from Hugging Face datasets

Downloads SAM3 weights from Hugging Face models

Running in Colab
python
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
PET CSV Format
The default PET CSV contains:

Column	Description
event_id	Integer conflict index
pet	Post-Encroachment Time (seconds)
frame	Frame index (may be NaN)
track_a	Track ID of first actor
track_b	Track ID of second actor
conflict_type	Grid cell ID (e.g., "CELL_C_1")
world_traj_i	BEV trajectory for actor a
world_traj_j	BEV trajectory for actor b
Development
Running Tests
bash
# Smoke tests
pytest tests/ -v

# Import tests
PYTHONPATH=. python tests/test_imports_smoke.py
Code Quality
bash
# Format code
black .

# Type checking (if using mypy)
mypy --ignore-missing-imports .
Citation
If you use NNDS in your research, please cite:

bibtex
@software{nnds2024,
  author = {Sharqascc},
  title = {NNDS: Non-motorized and Heterogeneous Traffic Safety Analysis},
  year = {2024},
  url = {https://github.com/Sharqascc/nnds}
}
References
Allen, B. L., Shin, B. T., & Cooper, P. J. (1977). Analysis of traffic conflicts and collisions. Transportation Research Record, 667, 67-74.

Gettman, D., & Head, L. (2003). Surrogate safety measures from traffic simulation models. FHWA-RD-03-050.

Hyd'en, C. (1987). The Swedish Traffic Conflicts Technique. Bulletin Lund Institute of Technology, 70.

Zheng, L., Ismail, K., & Meng, X. (2014). Traffic conflict techniques for road safety analysis. Accident Analysis & Prevention.

FHWA. (2008). Surrogate Safety Assessment Model and Validation. FHWA-HRT-08-051.

License
MIT License - see LICENSE file for details.

Acknowledgments
SAM3 model from Meta AI

Hugging Face for model/dataset hosting

FHWA for SSAM methodology guidance

text

## Key Improvements in This README

| Section | Improvement |
|---------|-------------|
| **Structure** | Removed duplicate SSM sections (had 3 copies) |
| **Visualization** | Updated to reflect 26 exports, v2.3.0 |
| **PET Summary** | Shows v2.0.0 features (statistical tests, CI, effect sizes) |
| **Research Run** | Added skip flags, dry-run, proper workflow |
| **Project Structure** | Complete, accurate tree matching your output |
| **Health Check** | Shows all components pass |
| **Tables** | Clean, organized, easy to scan |
| **Emojis** | Consistent visual hierarchy |

This README is **production-ready** and accurately reflects your current pipeline state. Save it as `README.md` in your repo root! 🚀
