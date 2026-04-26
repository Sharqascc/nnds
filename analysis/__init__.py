"""
Traffic Safety Analysis Module

Comprehensive analysis tools for traffic safety evaluation using:
- Surrogate Safety Measures (SSM)
- Diffusion model evaluation
- Statistical analysis
- Publication-quality visualization

Sub-modules:
- visualization: Publication-quality plots and figures
- pet_diffusion_analysis: Diffusion model PET evaluation
- safety_eval_diffusion: Safety metric evaluation
- research_run: Research experiment utilities

Example usage:
    # Quick analysis
    from analysis.pet_diffusion_analysis import PETDiffusionAnalyzer

    analyzer = PETDiffusionAnalyzer(output_dir='outputs')
    pet_pairs, df = analyzer.compute_pet_like_metrics(batch, sample_fn)

    # Visualization
    from analysis.visualization import plot_pet_distribution

    plot_pet_distribution(df['pet'].values, save_path='pet_dist.png')
"""

# Diffusion analysis
try:
    from .pet_diffusion_analysis import (
        PETDiffusionAnalyzer,
        compute_pet_like_metrics,
        compare_realPET_samplePET,
        parse_trajectory,
        compute_error_metrics,
        perform_statistical_tests,
    )
    _diffusion_available = True
except ImportError as e:
    _diffusion_available = False
    import warnings
    warnings.warn(f"Diffusion analysis module not available: {e}")

# Visualization sub-package
try:
    from . import visualization
    _viz_available = True
except ImportError as e:
    _viz_available = False
    import warnings
    warnings.warn(f"Visualization module not available: {e}")

# PET summary analysis
try:
    from .pet_summary import PETEventAnalyzer
    _pet_summary_available = True
except ImportError as e:
    _pet_summary_available = False
    import warnings
    warnings.warn(f"PET summary module not available: {e}")

# Define what gets exported with "from analysis import *"
__all__ = [
    # Sub-packages
    "visualization",

    # Diffusion analysis
    "PETDiffusionAnalyzer",
    "compute_pet_like_metrics",
    "compare_realPET_samplePET",
    "parse_trajectory",
    "compute_error_metrics",
    "perform_statistical_tests",

    # PET summary
    "PETEventAnalyzer",
]

# Module metadata
__version__ = "1.1.0"
__author__ = "NNDS Team"


def check_installation():
    """Check which analysis modules are available."""
    status = {
        "diffusion_analysis": _diffusion_available,
        "visualization": _viz_available,
        "pet_summary": _pet_summary_available,
    }

    print("=" * 60)
    print("ANALYSIS MODULE STATUS")
    print("=" * 60)

    for module, available in status.items():
        symbol = "✅" if available else "❌"
        print(f"{symbol} {module:20s} : {'Available' if available else 'Not available'}")

    print("=" * 60)

    if all(status.values()):
        print("✅ All analysis modules loaded successfully!")
    else:
        print("⚠️  Some modules are missing. Install dependencies:")
        if not _viz_available:
            print("   pip install matplotlib seaborn plotly")
        if not _diffusion_available:
            print("   pip install torch scipy pandas")
        if not _pet_summary_available:
            print("   pip install pandas scipy numpy")

    return status


# Quick start examples
__doc_examples__ = """
Quick Start Examples
====================

DIFFUSION MODEL EVALUATION
==========================

1. Basic PET-like Metric Computation
------------------------------------
from analysis import PETDiffusionAnalyzer

analyzer = PETDiffusionAnalyzer(
    d_thresh=1.0,
    fps=30.0,
    output_dir='outputs/diffusion_eval',
    auto_visualize=True
)

# Compute metrics (auto-saves CSV + plots)
pet_pairs, df = analyzer.compute_pet_like_metrics(
    batch=data_batch,
    sample_future_fn=sample_fn
)

2. Compare with Ground Truth
-----------------------------
from analysis import PETDiffusionAnalyzer

analyzer = PETDiffusionAnalyzer(output_dir='outputs')

records, df = analyzer.compare_with_ground_truth(
    df_pet_path='outputs/petevents_bev_30frames.csv',
    batch=batch,
    sample_future_fn=sample_fn
)

print(f"MAE: {df['error_sample'].abs().mean():.3f}s")

3. Custom Statistical Analysis
-------------------------------
from analysis import compute_error_metrics, perform_statistical_tests
import numpy as np

real_pet = np.array([0.5, 1.2, 0.8, ...])
sample_pet = np.array([0.6, 1.1, 0.9, ...])

# Error metrics
metrics = compute_error_metrics(real_pet, sample_pet)
print(f"RMSE: {metrics['rmse']:.3f}")
print(f"R²: {metrics['r_squared']:.3f}")

# Statistical significance
tests = perform_statistical_tests(real_pet, sample_pet)
print(f"Paired t-test p-value: {tests['paired_t_test']['p_value']:.4f}")


PET SUMMARY ANALYSIS
====================

4. PET Event Summary
--------------------
from analysis import PETEventAnalyzer

analyzer = PETEventAnalyzer('outputs/petevents_bev_30frames.csv')
analyzer.print_summary()

# Export full analysis bundle
exported = analyzer.export_results(
    output_dir='analysis_results',
    baseline_csv=Path('traffic_diffusion/data/generated_pet.csv'),
    fmt='json',
)
print(exported)


VISUALIZATION
=============

5. PET Distribution Plot
------------------------
from analysis.visualization import plot_pet_distribution
import pandas as pd

df = pd.read_csv('outputs/petevents_bev_30frames.csv')

plot_pet_distribution(
    df['pet'].values,
    style='journal',
    save_path='outputs/pet_distribution.png'
)

6. Before/After Comparison
--------------------------
from analysis.visualization import plot_comparative_boxplot

plot_comparative_boxplot(
    data_groups={
        'Before': before_pet,
        'After': after_pet
    },
    metric_name='PET',
    show_stats=True,
    save_path='outputs/comparison.png'
)

7. Conflict Event Visualization
-------------------------------
from analysis.visualization import EventPlotter, load_pet_csv

df = load_pet_csv('outputs/petevents_bev_30frames.csv')

plotter = EventPlotter(dpi=300)
plotter.plot_conflict_event(
    df,
    event_id=5,
    show_velocities=True,
    save_path='outputs/event_5.png'
)

8. Video Frame Overlay
----------------------
from analysis.visualization import VideoOverlayPlotter

plotter = VideoOverlayPlotter(dpi=300, colorblind_safe=True)

frame = plotter.overlay_conflict_frame(
    video_path='videos/traffic.mp4',
    frame_idx=1234,
    trajectories=trajectories,
    pet_value=0.75,
    save_path='outputs/frame_overlay.png'
)

9. Diffusion Evaluation Plots
-----------------------------
from analysis.visualization import DiffusionPETPlotter

plotter = DiffusionPETPlotter(dpi=300, save_pdf=True)

# Generate all evaluation plots
plotter.plot_all(
    pet_pairs=pet_pairs,
    records=records,
    out_dir='outputs/diffusion_plots'
)


COMPLETE ANALYSIS PIPELINE
==========================

10. End-to-End Analysis
-----------------------
from analysis import PETDiffusionAnalyzer, PETEventAnalyzer
from analysis.visualization import (
    plot_pet_distribution,
    EventPlotter,
    DiffusionPETPlotter
)
import pandas as pd

# 1. Load PET CSV
df = pd.read_csv('outputs/petevents_bev_30frames.csv')

# 2. Aggregate analysis
plot_pet_distribution(
    df['pet'].values,
    save_path='paper/fig1_distribution.png'
)

# 3. PET summary + export
pet_analyzer = PETEventAnalyzer('outputs/petevents_bev_30frames.csv')
pet_analyzer.print_summary()
pet_analyzer.export_results(output_dir='paper/pet_analysis')

# 4. Diffusion evaluation
diff_analyzer = PETDiffusionAnalyzer(
    output_dir='paper/diffusion_eval',
    auto_visualize=True
)

pet_pairs, df_metrics = diff_analyzer.compute_pet_like_metrics(
    batch, sample_fn
)

records, df_comp = diff_analyzer.compare_with_ground_truth(
    'outputs/petevents_bev_30frames.csv',
    batch, sample_fn
)

# 5. Case study visualization
event_plotter = EventPlotter(dpi=300)
event_plotter.plot_conflict_event(
    df, event_id=42,
    save_path='paper/fig3_case_study.png'
)

# 6. Diffusion plots
diff_plotter = DiffusionPETPlotter(dpi=300)
diff_plotter.plot_all(
    pet_pairs, records,
    out_dir='paper/diffusion_eval'
)

print("✅ Complete analysis pipeline finished!")
print("📁 Results saved to paper/")
"""
