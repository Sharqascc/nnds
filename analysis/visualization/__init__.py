"""Publication-quality visualization for traffic safety analysis.

Provides industry-standard plots for Surrogate Safety Measures (SSM):
- Distribution plots (histogram + KDE)
- Time series analysis
- Severity scatter plots
- Comparative boxplots
- Cumulative distribution functions (ECDF)
- Correlation heatmaps
- Temporal density heatmaps
- Conflict density maps
- Diffusion model evaluation plots

Features:
- Journal-ready styling (300 DPI, serif fonts, proper sizing)
- Colorblind-safe palettes (Okabe-Ito)
- Vector graphics export (PDF + PNG)
- Statistical annotations (p-values, regression, effect sizes)
- Configurable safety thresholds
- APA/IEEE compliant formatting

Compliant with:
- Transportation Research Board (TRB) standards
- IEEE Transactions on Intelligent Transportation Systems
- Accident Analysis & Prevention journal requirements
"""

# Import SSM visualization functions
from .industry_standard_viz import (
    SSMPlotter,
    plot_pet_distribution,
    plot_ttc_time_series,
    plot_severity_scatter,
    plot_conflict_density_map,
    plot_comparative_boxplot,
    plot_cumulative_distribution,
    plot_correlation_heatmap,
    plot_temporal_heatmap
)

# Import diffusion model evaluation functions
from .pet_diffusion_plots import (
    DiffusionPETPlotter,
    plot_pet_like_histogram,
    plot_true_vs_pet_like,
    plot_true_vs_sample_delta,
    plot_residual_analysis,
    plot_bland_altman
)

__all__ = [
    # SSM Analysis - Main plotter class
    'SSMPlotter',
    
    # SSM Analysis - Individual plot functions
    'plot_pet_distribution',
    'plot_ttc_time_series',
    'plot_severity_scatter',
    'plot_conflict_density_map',
    'plot_comparative_boxplot',
    'plot_cumulative_distribution',
    'plot_correlation_heatmap',
    'plot_temporal_heatmap',
    
    # Diffusion Model Evaluation
    'DiffusionPETPlotter',
    'plot_pet_like_histogram',
    'plot_true_vs_pet_like',
    'plot_true_vs_sample_delta',
    'plot_residual_analysis',
    'plot_bland_altman',
]

__version__ = '2.1.0'  # Updated version


# Quick usage examples
__doc_examples__ = """
Quick Start Examples
====================

SSM ANALYSIS PLOTS
==================

1. Basic PET Distribution
--------------------------
from analysis.visualization import plot_pet_distribution
import pandas as pd

df = pd.read_csv('outputs/petevents_bev_30frames.csv')
fig = plot_pet_distribution(
    df['pet'].values,
    style='journal',
    save_path='outputs/pet_dist.png'
)

2. Time Series with Critical Events
------------------------------------
from analysis.visualization import plot_ttc_time_series

fig = plot_ttc_time_series(
    timestamps=df['frame'].values / 30.0,  # Convert frames to seconds
    ttc_values=df['TTC'].values,
    highlight_critical=True,
    save_path='outputs/ttc_series.png'
)

3. Severity Correlation Scatter
--------------------------------
from analysis.visualization import plot_severity_scatter

fig = plot_severity_scatter(
    pet_values=df['pet'].values,
    ttc_values=df['TTC'].values,
    add_regression=True,
    save_path='outputs/severity.png'
)

4. Before/After Comparison
---------------------------
from analysis.visualization import plot_comparative_boxplot

before = df[df['period']=='before']['pet'].values
after = df[df['period']=='after']['pet'].values

fig = plot_comparative_boxplot(
    data_groups={'Before': before, 'After': after},
    metric_name='PET',
    show_stats=True,
    save_path='outputs/comparison.png'
)

5. Advanced: Custom Thresholds
-------------------------------
from analysis.visualization import SSMPlotter

plotter = SSMPlotter(
    style='journal',
    dpi=300,
    colorblind_safe=True,
    critical_pet=0.75,  # Custom threshold
    safe_pet=4.0
)

fig = plotter.plot_pet_distribution(
    df['pet'].values,
    save_path='outputs/custom_pet.png'
)

6. Multi-Metric Correlation
----------------------------
from analysis.visualization import plot_correlation_heatmap

fig = plot_correlation_heatmap(
    data_dict={
        'PET': df['pet'].values,
        'TTC': df['TTC'].values,
        'DRAC': df['DRAC'].values,
        'Speed': df['speed'].values
    },
    save_path='outputs/correlation.png'
)

7. Temporal Density Heatmap
----------------------------
from analysis.visualization import plot_temporal_heatmap

fig = plot_temporal_heatmap(
    timestamps=df['frame'].values / 30.0,
    pet_values=df['pet'].values,
    time_bins=30,
    severity_bins=15,
    save_path='outputs/temporal_density.png'
)

8. ECDF for Multiple Groups
----------------------------
from analysis.visualization import plot_cumulative_distribution

fig = plot_cumulative_distribution(
    data_groups={
        'Intersection A': df_a['pet'].values,
        'Intersection B': df_b['pet'].values,
        'Intersection C': df_c['pet'].values
    },
    metric_name='PET',
    save_path='outputs/ecdf.png'
)


DIFFUSION MODEL EVALUATION
===========================

9. PET-like Error Distribution
-------------------------------
from analysis.visualization import plot_pet_like_histogram

# pet_pairs = [(real_steps, sample_steps), ...]
fig = plot_pet_like_histogram(
    pet_pairs=pet_pairs,
    bins=30,
    show_stats=True,
    save_path='outputs/diffusion/error_histogram.png'
)

10. Ground Truth vs Generated
------------------------------
from analysis.visualization import plot_true_vs_pet_like

# records = [(idx, true_pet_sec, real_steps, sample_steps), ...]
fig = plot_true_vs_pet_like(
    records=records,
    add_regression=True,
    save_path='outputs/diffusion/scatter.png'
)

11. Generation Error Analysis
------------------------------
from analysis.visualization import plot_true_vs_sample_delta

fig = plot_true_vs_sample_delta(
    records=records,
    add_trend=True,
    save_path='outputs/diffusion/error_vs_truth.png'
)

12. Residual Diagnostics
-------------------------
from analysis.visualization import plot_residual_analysis

# Complete diagnostic suite (4 plots in one figure)
fig = plot_residual_analysis(
    records=records,
    save_path='outputs/diffusion/residuals.png'
)

13. Bland-Altman Agreement
---------------------------
from analysis.visualization import plot_bland_altman

# Agreement analysis between real and sampled
fig = plot_bland_altman(
    records=records,
    save_path='outputs/diffusion/bland_altman.png'
)

14. Complete Diffusion Evaluation Suite
----------------------------------------
from analysis.visualization import DiffusionPETPlotter

plotter = DiffusionPETPlotter(
    style='default',
    dpi=300,
    save_pdf=True  # Save both PNG and PDF
)

# Generate all plots at once
plotter.plot_all(
    pet_pairs=pet_pairs,
    records=records,
    out_dir='outputs/diffusion_eval'
)


Output Formats
==============
All plots automatically save in:
- PNG (high-res raster, 300 DPI)
- PDF (vector graphics for LaTeX/Word)

Both formats are journal-ready and submission-compliant.


Module Organization
===================
- industry_standard_viz.py: General SSM analysis plots
- pet_diffusion_plots.py: Diffusion model evaluation plots

Both modules use:
- Colorblind-safe Okabe-Ito palette
- Publication-quality styling (300 DPI)
- Statistical annotations
- Configurable parameters
"""
