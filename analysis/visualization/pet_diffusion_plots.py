"""
Diffusion Model Evaluation Plots for PET Metrics

Specialized visualization for comparing real vs. generated trajectories
in the context of trajectory diffusion models and safety metrics.

Features:
- PET-like metric comparisons (real vs sampled)
- Distribution analysis of generation quality
- Correlation plots with ground truth
- Residual diagnostics with statistical tests
- Bland-Altman agreement analysis
- Publication-quality styling
- Statistical annotations
"""

import os
from typing import List, Tuple, Optional
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats


__all__ = [
    'plot_pet_like_histogram',
    'plot_true_vs_pet_like',
    'plot_true_vs_sample_delta',
    'plot_residual_analysis',
    'plot_bland_altman',
    'DiffusionPETPlotter'
]


# Colorblind-safe palette (Okabe-Ito - consistent with industry_standard_viz)
COLORS = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'yellow': '#F0E442',
    'purple': '#CC79A7',
    'cyan': '#56B4E9',
    'red': '#D55E00',
    'black': '#000000'
}


# Publication-quality rcParams for this module. NOT applied automatically on
# import (previously this was a bare plt.rcParams.update() at module load,
# which silently mutated global matplotlib state for any other code that
# happened to import this module). Call apply_pet_diffusion_style() explicitly
# if you want these defaults active.
_PET_DIFFUSION_RCPARAMS = {
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
}


def apply_pet_diffusion_style():
    """Explicitly opt in to this module's publication-quality rcParams.

    Intentionally not called at import time -- doing so used to mutate
    global matplotlib state (plt.rcParams) as a side effect of merely
    importing this module, which could unexpectedly change styling in
    unrelated code sharing the same process.
    """
    plt.rcParams.update(_PET_DIFFUSION_RCPARAMS)


def _maybe_save(out_path: Optional[str], save_pdf: bool = True):
    """
    Save figure if path provided.

    Args:
        out_path: Path to save (creates directory if needed)
        save_pdf: Also save PDF version for publications
    """
    if out_path is not None:
        dirname = os.path.dirname(out_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        # Save PNG
        plt.savefig(out_path, dpi=300, bbox_inches="tight", format='png')

        # Also save PDF for publications (configurable)
        if save_pdf:
            pdf_path = out_path.replace('.png', '.pdf')
            if pdf_path != out_path:
                plt.savefig(pdf_path, bbox_inches="tight", format='pdf')


def plot_pet_like_histogram(
    pet_pairs: List[Tuple[float, float]],
    out_path: Optional[str] = None,
    title: str = "PET-like Step Differences (Sampled - Real)",
    bins: int = 30,
    show_stats: bool = True,
    save_pdf: bool = True
):
    """
    Plot histogram of PET-like step differences with statistical annotations.

    Args:
        pet_pairs: List of (pet_real_steps, pet_sample_steps) tuples
        out_path: Save path (optional)
        title: Plot title
        bins: Number of histogram bins
        show_stats: Show mean, median, std on plot
        save_pdf: Save PDF version for publications

    Returns:
        None (displays/saves plot)
    """
    # Extract valid differences
    diffs = [
        (ps - pr)
        for (pr, ps) in pet_pairs
        if pr is not None and ps is not None
    ]

    if not diffs:
        warnings.warn("No examples with both real and sample PET-like defined.")
        return

    diffs = np.array(diffs, dtype=float)

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))

    # Histogram (colorblind-safe)
    counts, bin_edges, patches = ax.hist(
        diffs,
        bins=bins,
        edgecolor="black",
        alpha=0.7,
        color=COLORS['blue'],
        label='Error Distribution'
    )

    # Zero line (perfect match)
    ax.axvline(0.0, color=COLORS['red'], linestyle="--", linewidth=2, label="Perfect Match", zorder=5)

    # Mean and median
    mean_diff = np.mean(diffs)
    median_diff = np.median(diffs)

    ax.axvline(mean_diff, color=COLORS['orange'], linestyle="-.", linewidth=2, label=f"Mean = {mean_diff:.2f}", alpha=0.8)
    ax.axvline(median_diff, color=COLORS['green'], linestyle=":", linewidth=2, label=f"Median = {median_diff:.2f}", alpha=0.8)

    # Statistics box
    if show_stats:
        std_diff = np.std(diffs)
        mae = np.mean(np.abs(diffs))
        rmse = np.sqrt(np.mean(diffs**2))

        stats_text = (
            f"N = {len(diffs)}\n"
            f"Mean = {mean_diff:.2f}\n"
            f"Median = {median_diff:.2f}\n"
            f"Std = {std_diff:.2f}\n"
            f"MAE = {mae:.2f}\n"
            f"RMSE = {rmse:.2f}"
        )

        ax.text(
            0.98, 0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9)
        )

    # Labels
    ax.set_xlabel("Sampled PET-like - Real PET-like (steps)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    _maybe_save(out_path, save_pdf=save_pdf)
    plt.show()
    plt.close()


def plot_true_vs_pet_like(
    records: List[Tuple[int, float, float, float]],
    out_path: Optional[str] = None,
    title: str = "Ground Truth PET vs PET-like Steps",
    add_regression: bool = True,
    save_pdf: bool = True
):
    """
    Scatter plot comparing true PET (seconds) from CSV with PET-like steps
    for real vs sampled trajectories.

    Args:
        records: List of (row_idx, true_pet, pet_like_real, pet_like_sample)
        out_path: Save path (optional)
        title: Plot title
        add_regression: Add regression lines
        save_pdf: Save PDF version

    Returns:
        None (displays/saves plot)
    """
    true_pet = []
    pet_real = []
    pet_sample = []

    for row_idx, t_pet, pr, ps in records:
        if pr is None or ps is None:
            continue
        true_pet.append(float(t_pet))
        pet_real.append(float(pr))
        pet_sample.append(float(ps))

    if not true_pet:
        warnings.warn("No records with both real and sample PET-like defined.")
        return

    true_pet = np.array(true_pet)
    pet_real = np.array(pet_real)
    pet_sample = np.array(pet_sample)

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 6))

    # Scatter plots (colorblind-safe)
    ax.scatter(
        true_pet, pet_real,
        s=30, alpha=0.6,
        color=COLORS['blue'],
        edgecolors='black',
        linewidths=0.5,
        label="Real Trajectories"
    )
    ax.scatter(
        true_pet, pet_sample,
        s=30, alpha=0.6,
        color=COLORS['orange'],
        edgecolors='black',
        linewidths=0.5,
        label="Sampled Trajectories"
    )

    # Regression lines
    if add_regression and len(true_pet) > 2:
        # Real regression
        slope_r, intercept_r, r_r, p_r, _ = stats.linregress(true_pet, pet_real)
        x_line = np.linspace(true_pet.min(), true_pet.max(), 100)
        y_line_r = slope_r * x_line + intercept_r
        ax.plot(
            x_line, y_line_r,
            color=COLORS['blue'], linestyle='--', linewidth=2, alpha=0.7,
            label=f'Real: R²={r_r**2:.3f}, p={p_r:.4f}'
        )

        # Sample regression
        slope_s, intercept_s, r_s, p_s, _ = stats.linregress(true_pet, pet_sample)
        y_line_s = slope_s * x_line + intercept_s
        ax.plot(
            x_line, y_line_s,
            color=COLORS['orange'], linestyle='--', linewidth=2, alpha=0.7,
            label=f'Sample: R²={r_s**2:.3f}, p={p_s:.4f}'
        )

    # Labels
    ax.set_xlabel("Ground Truth PET (seconds)", fontsize=11)
    ax.set_ylabel("PET-like Metric (steps)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    _maybe_save(out_path, save_pdf=save_pdf)
    plt.show()
    plt.close()


def plot_true_vs_sample_delta(
    records: List[Tuple[int, float, float, float]],
    out_path: Optional[str] = None,
    title: str = "Ground Truth PET vs Generation Error",
    add_trend: bool = True,
    save_pdf: bool = True
):
    """
    Plot true PET (seconds) vs difference in PET-like steps (sample - real).

    Useful for identifying if generation quality varies with PET magnitude.

    Args:
        records: List of (row_idx, true_pet, pet_like_real, pet_like_sample)
        out_path: Save path (optional)
        title: Plot title
        add_trend: Add LOWESS trend line
        save_pdf: Save PDF version

    Returns:
        None (displays/saves plot)
    """
    true_pet = []
    delta_steps = []

    for row_idx, t_pet, pr, ps in records:
        if pr is None or ps is None:
            continue
        true_pet.append(float(t_pet))
        delta_steps.append(float(ps - pr))

    if not true_pet:
        warnings.warn("No records with both real and sample PET-like defined.")
        return

    true_pet = np.array(true_pet)
    delta_steps = np.array(delta_steps)

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))

    # Scatter plot (colorblind-safe)
    ax.scatter(
        true_pet, delta_steps,
        s=30, alpha=0.6,
        color=COLORS['blue'],
        edgecolors='black',
        linewidths=0.5
    )

    # Zero line (perfect generation)
    ax.axhline(0.0, color=COLORS['red'], linestyle="--", linewidth=2, label="Perfect Generation", zorder=5)

    # Mean error
    mean_error = np.mean(delta_steps)
    ax.axhline(mean_error, color=COLORS['orange'], linestyle="-.", linewidth=2, label=f"Mean Error = {mean_error:.2f}", alpha=0.8)

    # Trend line (LOWESS or linear regression)
    if add_trend and len(true_pet) > 10:
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed = lowess(delta_steps, true_pet, frac=0.3)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color=COLORS['green'], linewidth=2.5, label='LOWESS Trend', zorder=4)
        except ImportError:
            # Fallback to linear regression
            slope, intercept, r, p, _ = stats.linregress(true_pet, delta_steps)
            x_line = np.linspace(true_pet.min(), true_pet.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=COLORS['green'], linestyle='--', linewidth=2, label=f'Linear Trend (R²={r**2:.3f})', zorder=4)

    # Statistics
    mae = np.mean(np.abs(delta_steps))
    rmse = np.sqrt(np.mean(delta_steps**2))

    stats_text = (
        f"N = {len(delta_steps)}\n"
        f"Mean Error = {mean_error:.2f}\n"
        f"MAE = {mae:.2f}\n"
        f"RMSE = {rmse:.2f}"
    )

    ax.text(
        0.98, 0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9)
    )

    # Labels
    ax.set_xlabel("Ground Truth PET (seconds)", fontsize=11)
    ax.set_ylabel("Generation Error: Sampled - Real (steps)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    _maybe_save(out_path, save_pdf=save_pdf)
    plt.show()
    plt.close()


def plot_residual_analysis(
    records: List[Tuple[int, float, float, float]],
    out_path: Optional[str] = None,
    save_pdf: bool = True
):
    """
    Residual plot for diagnosing systematic errors in generation.

    Includes:
    - Residuals vs fitted
    - Histogram with normality test
    - Q-Q plot
    - Scale-location plot

    Args:
        records: List of (row_idx, true_pet, pet_like_real, pet_like_sample)
        out_path: Save path (optional)
        save_pdf: Save PDF version
    """
    true_pet = []
    residuals = []

    for row_idx, t_pet, pr, ps in records:
        if pr is None or ps is None:
            continue
        true_pet.append(float(t_pet))
        residuals.append(float(ps - pr))

    if not true_pet:
        warnings.warn("No valid records for residual analysis.")
        return

    true_pet = np.array(true_pet)
    residuals = np.array(residuals)

    # Normality test (randomly subsample when over Shapiro-Wilk's N=5000 cap,
    # instead of always taking the first 5000 -- records are often sorted or
    # grouped by row_idx, so a fixed-order slice can bias the normality test)
    if len(residuals) >= 3:
        if len(residuals) > 5000:
            rng = np.random.default_rng(42)
            shapiro_sample = rng.choice(residuals, size=5000, replace=False)
        else:
            shapiro_sample = residuals
        shapiro_stat, shapiro_p = stats.shapiro(shapiro_sample)  # Shapiro max 5000, randomly subsampled
    else:
        shapiro_p = np.nan

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # 1. Residual vs Fitted (colorblind-safe)
    axes[0, 0].scatter(
        true_pet, residuals,
        alpha=0.6, s=30,
        color=COLORS['blue'],
        edgecolors='black',
        linewidths=0.5
    )
    axes[0, 0].axhline(0, color=COLORS['red'], linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Ground Truth PET (s)')
    axes[0, 0].set_ylabel('Residuals (steps)')
    axes[0, 0].set_title('Residuals vs Fitted')
    axes[0, 0].grid(alpha=0.3)

    # 2. Histogram of residuals with normality test
    axes[0, 1].hist(
        residuals,
        bins=25,
        edgecolor='black',
        alpha=0.7,
        color=COLORS['blue']
    )
    axes[0, 1].axvline(0, color=COLORS['red'], linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Residuals (steps)')
    axes[0, 1].set_ylabel('Frequency')

    # Add normality test result to title
    if not np.isnan(shapiro_p):
        normality_text = f'Normal' if shapiro_p > 0.05 else 'Non-normal'
        axes[0, 1].set_title(f'Residual Distribution\n(Shapiro p={shapiro_p:.4f}, {normality_text})')
    else:
        axes[0, 1].set_title('Residual Distribution')

    axes[0, 1].grid(alpha=0.3)

    # 3. Q-Q plot (FIXED: consistent styling)
    res = stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    # Manually style the points for consistency
    axes[1, 0].get_lines()[0].set_markerfacecolor(COLORS['blue'])
    axes[1, 0].get_lines()[0].set_markeredgecolor('black')
    axes[1, 0].get_lines()[0].set_markeredgewidth(0.5)
    axes[1, 0].get_lines()[0].set_markersize(6)
    axes[1, 0].get_lines()[0].set_alpha(0.6)
    axes[1, 0].set_title('Normal Q-Q Plot')
    axes[1, 0].grid(alpha=0.3)

    # 4. Scale-Location plot
    sqrt_abs_resid = np.sqrt(np.abs(residuals))
    axes[1, 1].scatter(
        true_pet, sqrt_abs_resid,
        alpha=0.6, s=30,
        color=COLORS['blue'],
        edgecolors='black',
        linewidths=0.5
    )
    axes[1, 1].set_xlabel('Ground Truth PET (s)')
    axes[1, 1].set_ylabel('√|Residuals|')
    axes[1, 1].set_title('Scale-Location Plot')
    axes[1, 1].grid(alpha=0.3)

    plt.suptitle('Residual Diagnostic Plots', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    _maybe_save(out_path, save_pdf=save_pdf)
    plt.show()
    plt.close()


def plot_bland_altman(
    records: List[Tuple[int, float, float, float]],
    out_path: Optional[str] = None,
    title: str = "Bland-Altman Plot: Real vs Sampled PET-like",
    save_pdf: bool = True
):
    """
    Bland-Altman plot for agreement between real and sampled PET-like metrics.

    Args:
        records: List of (row_idx, true_pet, pet_like_real, pet_like_sample)
        out_path: Save path (optional)
        title: Plot title
        save_pdf: Save PDF version
    """
    pet_real = []
    pet_sample = []

    for row_idx, t_pet, pr, ps in records:
        if pr is None or ps is None:
            continue
        pet_real.append(float(pr))
        pet_sample.append(float(ps))

    if not pet_real:
        warnings.warn("No valid records for Bland-Altman plot.")
        return

    pet_real = np.array(pet_real)
    pet_sample = np.array(pet_sample)

    # Calculate mean and difference
    mean_vals = (pet_real + pet_sample) / 2
    diff_vals = pet_sample - pet_real

    mean_diff = np.mean(diff_vals)
    std_diff = np.std(diff_vals)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter (colorblind-safe)
    ax.scatter(
        mean_vals, diff_vals,
        alpha=0.6, s=30,
        color=COLORS['blue'],
        edgecolors='black',
        linewidths=0.5
    )

    # Mean difference
    ax.axhline(mean_diff, color=COLORS['blue'], linestyle='-', linewidth=2, label=f'Mean Diff = {mean_diff:.2f}')

    # Limits of agreement (±1.96 SD)
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff

    ax.axhline(upper_loa, color=COLORS['red'], linestyle='--', linewidth=2, label=f'+1.96 SD = {upper_loa:.2f}')
    ax.axhline(lower_loa, color=COLORS['red'], linestyle='--', linewidth=2, label=f'-1.96 SD = {lower_loa:.2f}')

    ax.set_xlabel('Mean of Real and Sampled PET-like (steps)', fontsize=11)
    ax.set_ylabel('Difference: Sampled - Real (steps)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    _maybe_save(out_path, save_pdf=save_pdf)
    plt.show()
    plt.close()


class DiffusionPETPlotter:
    """
    Unified plotter for diffusion model PET evaluation.

    Provides all plotting functions with consistent styling and
    configurable export options.
    """

    def __init__(self, style: str = 'default', dpi: int = 300, save_pdf: bool = True):
        """
        Args:
            style: Matplotlib style ('default', 'seaborn', etc.)
            dpi: Resolution for saved figures
            save_pdf: Save PDF versions alongside PNG
        """
        self.style = style
        self.dpi = dpi
        self.save_pdf = save_pdf

        if style != 'default':
            plt.style.use(style)

        plt.rcParams['savefig.dpi'] = dpi

    def plot_all(
        self,
        pet_pairs: List[Tuple[float, float]],
        records: List[Tuple[int, float, float, float]],
        out_dir: str = 'outputs/diffusion_eval'
    ):
        """
        Generate all evaluation plots.

        Args:
            pet_pairs: PET-like pairs for histogram
            records: Full records for scatter plots
            out_dir: Output directory
        """
        os.makedirs(out_dir, exist_ok=True)

        print(f"🎨 Generating diffusion evaluation plots...")

        plot_pet_like_histogram(
            pet_pairs,
            out_path=f'{out_dir}/histogram.png',
            save_pdf=self.save_pdf
        )

        plot_true_vs_pet_like(
            records,
            out_path=f'{out_dir}/scatter.png',
            save_pdf=self.save_pdf
        )

        plot_true_vs_sample_delta(
            records,
            out_path=f'{out_dir}/error_vs_truth.png',
            save_pdf=self.save_pdf
        )

        plot_residual_analysis(
            records,
            out_path=f'{out_dir}/residuals.png',
            save_pdf=self.save_pdf
        )

        plot_bland_altman(
            records,
            out_path=f'{out_dir}/bland_altman.png',
            save_pdf=self.save_pdf
        )

        print(f"✅ All plots saved to: {out_dir}/")
        if self.save_pdf:
            print(f"📄 PDF versions also saved for publication")
