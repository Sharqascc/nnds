"""
Industry-Standard Visualization for Surrogate Safety Measures

Publication-quality plots for traffic safety research:
- Distribution plots (histogram, KDE, violin)
- Time series analysis
- Severity scatter plots
- Heatmaps and spatial conflict density
- Comparative plots (before/after, multi-group)
- Statistical annotations (p-values, confidence intervals)

Compliant with:
- IEEE/Transportation Research Board figure guidelines
- Journal submission requirements (high DPI, vector formats)
- Accessibility standards (colorblind-safe palettes)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from typing import List, Dict, Tuple, Optional, Union
import warnings

# Optional pandas for correlation heatmap
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    warnings.warn("pandas not available - correlation heatmap disabled")


__all__ = [
    'SSMPlotter',
    'plot_pet_distribution',
    'plot_ttc_time_series',
    'plot_severity_scatter',
    'plot_conflict_density_map',
    'plot_comparative_boxplot',
    'plot_cumulative_distribution',
    'plot_correlation_heatmap',
    'plot_temporal_heatmap'
]


class SSMPlotter:
    """
    Industry-standard visualizer for Surrogate Safety Measures.
    
    Features:
    - Journal-quality styling (300+ DPI, vector formats)
    - Colorblind-safe palettes (Okabe-Ito)
    - Statistical annotations
    - Multiple export formats (PNG, PDF, SVG)
    - APA/IEEE compliant formatting
    - Configurable safety thresholds
    """
    
    # Colorblind-safe palette (Okabe-Ito)
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
    
    def __init__(
        self,
        style: str = 'journal',
        dpi: int = 300,
        font_size: float = 10.0,
        colorblind_safe: bool = True,
        # Configurable safety thresholds (FHWA-based)
        critical_pet: float = 0.5,
        serious_pet: float = 1.0,
        moderate_pet: float = 1.5,
        safe_pet: float = 5.0,
        critical_ttc: float = 1.5,
        warning_ttc: float = 3.0,
        safe_ttc: float = 5.0
    ):
        """
        Args:
            style: 'journal' or 'presentation'
            dpi: Resolution (300 for print, 150 for web)
            font_size: Base font size in points
            colorblind_safe: Use colorblind-safe palette
            critical_pet: Critical PET threshold (seconds)
            serious_pet: Serious PET threshold (seconds)
            moderate_pet: Moderate PET threshold (seconds)
            safe_pet: Safe PET threshold (seconds)
            critical_ttc: Critical TTC threshold (seconds)
            warning_ttc: Warning TTC threshold (seconds)
            safe_ttc: Safe TTC threshold (seconds)
        """
        self.style = style
        self.dpi = dpi
        self.font_size = font_size
        self.colorblind_safe = colorblind_safe
        
        # Safety thresholds
        self.thresholds = {
            'pet': {
                'critical': critical_pet,
                'serious': serious_pet,
                'moderate': moderate_pet,
                'safe': safe_pet
            },
            'ttc': {
                'critical': critical_ttc,
                'warning': warning_ttc,
                'safe': safe_ttc
            }
        }
        
        self._setup_style()
    
    def _setup_style(self):
        """Configure matplotlib for publication-quality output."""
        if self.style == 'journal':
            plt.style.use('seaborn-v0_8-paper')
            matplotlib.rcParams.update({
                'font.family': 'serif',
                'font.serif': ['Times New Roman', 'DejaVu Serif'],
                'font.size': self.font_size,
                'axes.titlesize': self.font_size + 2,
                'axes.labelsize': self.font_size + 1,
                'xtick.labelsize': self.font_size - 1,
                'ytick.labelsize': self.font_size - 1,
                'legend.fontsize': self.font_size - 1,
                'figure.figsize': (6.5, 5.0),  # Single column width
                'figure.dpi': self.dpi,
                'savefig.dpi': self.dpi,
                'savefig.format': 'png',
                'axes.linewidth': 1.0,
                'grid.linewidth': 0.5,
                'lines.linewidth': 1.5,
                'patch.linewidth': 0.5,
                'xtick.major.width': 0.8,
                'ytick.major.width': 0.8,
                'axes.grid': True,
                'grid.alpha': 0.3,
                'axes.axisbelow': True,
                'text.usetex': False,  # Set True if LaTeX available
            })
        
        elif self.style == 'presentation':
            plt.style.use('seaborn-v0_8-talk')
            matplotlib.rcParams.update({
                'font.size': self.font_size + 4,
                'figure.figsize': (10.0, 7.0),
                'lines.linewidth': 2.5,
                'axes.linewidth': 1.5,
            })
    
    def validate_ssm_data(
        self,
        data: np.ndarray,
        metric_name: str,
        allow_negative: bool = False
    ) -> Dict:
        """
        Validate and clean SSM data.
        
        Args:
            data: Input array
            metric_name: Name for error messages
            allow_negative: Whether negative values are valid
        
        Returns:
            Validation results with cleaned data
        """
        results = {
            'valid': True,
            'clean_data': None,
            'warnings': [],
            'errors': []
        }
        
        if data is None:
            results['valid'] = False
            results['errors'].append(f"{metric_name}: Input data is None")
            return results
        
        data = np.asarray(data)
        
        if data.ndim != 1:
            results['valid'] = False
            results['errors'].append(f"{metric_name}: Data must be 1D array (got shape {data.shape})")
            return results
        
        if len(data) == 0:
            results['valid'] = False
            results['errors'].append(f"{metric_name}: Data array is empty")
            return results
        
        # Count invalid values
        nan_count = int(np.sum(np.isnan(data)))
        inf_count = int(np.sum(np.isinf(data)))
        
        if nan_count > 0:
            results['warnings'].append(f"{metric_name}: {nan_count} NaN values removed")
        if inf_count > 0:
            results['warnings'].append(f"{metric_name}: {inf_count} Inf values removed")
        
        # Clean data
        clean = data[np.isfinite(data)]
        
        # Check for negative values
        if not allow_negative and np.any(clean < 0):
            neg_count = np.sum(clean < 0)
            results['warnings'].append(f"{metric_name}: {neg_count} negative values found")
        
        if len(clean) == 0:
            results['valid'] = False
            results['errors'].append(f"{metric_name}: No valid finite values after cleaning")
            return results
        
        results['clean_data'] = clean
        results['n_original'] = len(data)
        results['n_clean'] = len(clean)
        results['removal_rate'] = (len(data) - len(clean)) / len(data) * 100
        
        return results
    
    # ===================================================================
    # DISTRIBUTION PLOTS
    # ===================================================================
    
    def plot_pet_distribution(
        self,
        pet_values: np.ndarray,
        bins: int = 40,
        show_kde: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot PET distribution with safety thresholds and statistics.
        
        Args:
            pet_values: Array of PET values (seconds)
            bins: Number of histogram bins
            show_kde: Overlay kernel density estimate
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        validation = self.validate_ssm_data(pet_values, 'PET')
        
        if not validation['valid']:
            raise ValueError(f"Invalid data: {validation['errors']}")
        
        data = validation['clean_data']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=self.dpi)
        
        # Histogram
        counts, bin_edges, patches = ax.hist(
            data,
            bins=bins,
            density=True,
            alpha=0.7,
            color=self.COLORS['blue'],
            edgecolor='black',
            linewidth=0.5,
            label='PET Distribution'
        )
        
        # Color bars by severity (using configurable thresholds)
        pet_thresh = self.thresholds['pet']
        for i, patch in enumerate(patches):
            bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
            if bin_center < pet_thresh['critical']:
                patch.set_facecolor(self.COLORS['red'])
            elif bin_center < pet_thresh['serious']:
                patch.set_facecolor(self.COLORS['orange'])
            elif bin_center < pet_thresh['moderate']:
                patch.set_facecolor(self.COLORS['yellow'])
        
        # Kernel Density Estimate
        if show_kde and len(data) > 1:
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            ax.plot(x_range, kde(x_range), 'k-', linewidth=2, label='KDE')
        
        # Safety threshold lines (configurable)
        threshold_lines = {
            f'Critical (<{pet_thresh["critical"]}s)': (pet_thresh['critical'], self.COLORS['red']),
            f'Serious (<{pet_thresh["serious"]}s)': (pet_thresh['serious'], self.COLORS['orange']),
            f'Moderate (<{pet_thresh["moderate"]}s)': (pet_thresh['moderate'], self.COLORS['yellow']),
            f'Safe (>{pet_thresh["safe"]}s)': (pet_thresh['safe'], self.COLORS['green'])
        }
        
        for label, (val, color) in threshold_lines.items():
            ax.axvline(val, color=color, linestyle='--', linewidth=2, alpha=0.8)
        
        # Statistics box
        mean_val = np.mean(data)
        median_val = np.median(data)
        std_val = np.std(data)
        
        critical_pct = 100 * np.sum(data < pet_thresh['critical']) / len(data)
        serious_pct = 100 * np.sum(data < pet_thresh['serious']) / len(data)
        
        stats_text = (
            f"N = {len(data):,}\n"
            f"Mean = {mean_val:.2f}s\n"
            f"Median = {median_val:.2f}s\n"
            f"SD = {std_val:.2f}s\n"
            f"Critical = {critical_pct:.1f}%\n"
            f"Serious = {serious_pct:.1f}%"
        )
        
        ax.text(
            0.98, 0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=self.font_size - 1,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='white',
                edgecolor='gray',
                alpha=0.9
            )
        )
        
        # Labels and title
        ax.set_xlabel('Post-Encroachment Time (s)', fontsize=self.font_size + 1)
        ax.set_ylabel('Probability Density', fontsize=self.font_size + 1)
        ax.set_title(
            'PET Distribution with Safety Thresholds',
            fontsize=self.font_size + 2,
            fontweight='bold',
            pad=15
        )
        
        # Legend with severity categories
        legend_elements = [
            mpatches.Patch(color=self.COLORS['red'], label=f'Critical (<{pet_thresh["critical"]}s)'),
            mpatches.Patch(color=self.COLORS['orange'], label=f'Serious ({pet_thresh["critical"]}-{pet_thresh["serious"]}s)'),
            mpatches.Patch(color=self.COLORS['yellow'], label=f'Moderate ({pet_thresh["serious"]}-{pet_thresh["moderate"]}s)'),
            mpatches.Patch(color=self.COLORS['green'], label=f'Safe (>{pet_thresh["safe"]}s)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=self.font_size - 1)
        
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    # ===================================================================
    # TIME SERIES PLOTS
    # ===================================================================
    
    def plot_ttc_time_series(
        self,
        timestamps: np.ndarray,
        ttc_values: np.ndarray,
        highlight_critical: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot TTC time series with critical event highlighting.
        
        Args:
            timestamps: Time values (seconds or frame numbers)
            ttc_values: TTC values (seconds)
            highlight_critical: Highlight critical events
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        validation = self.validate_ssm_data(ttc_values, 'TTC')
        
        if not validation['valid']:
            raise ValueError(f"Invalid data: {validation['errors']}")
        
        data = validation['clean_data']
        
        # Match timestamp length
        if len(timestamps) != len(data):
            timestamps = np.arange(len(data))
        
        fig, ax = plt.subplots(figsize=(10.0, 5.0), dpi=self.dpi)
        
        ttc_thresh = self.thresholds['ttc']
        
        # Main time series
        ax.plot(
            timestamps,
            data,
            color=self.COLORS['blue'],
            linewidth=1.5,
            label='TTC',
            alpha=0.8
        )
        
        # Highlight critical events
        if highlight_critical:
            critical_mask = data < ttc_thresh['critical']
            if np.any(critical_mask):
                ax.scatter(
                    timestamps[critical_mask],
                    data[critical_mask],
                    color=self.COLORS['red'],
                    s=50,
                    marker='o',
                    label=f'Critical Events (<{ttc_thresh["critical"]}s)',
                    zorder=5
                )
        
        # Safety zones (configurable thresholds)
        ax.axhspan(0, ttc_thresh['critical'], alpha=0.2, color=self.COLORS['red'], label='Imminent Collision')
        ax.axhspan(ttc_thresh['critical'], ttc_thresh['warning'], alpha=0.15, color=self.COLORS['orange'], label='Critical')
        ax.axhspan(ttc_thresh['warning'], ttc_thresh['safe'], alpha=0.1, color=self.COLORS['yellow'], label='Warning')
        
        # Threshold lines
        ax.axhline(ttc_thresh['critical'], color=self.COLORS['red'], linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(ttc_thresh['warning'], color=self.COLORS['orange'], linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(ttc_thresh['safe'], color=self.COLORS['yellow'], linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Labels
        ax.set_xlabel('Time (s)', fontsize=self.font_size + 1)
        ax.set_ylabel('Time-To-Collision (s)', fontsize=self.font_size + 1)
        ax.set_title(
            'TTC Time Series with Safety Zones',
            fontsize=self.font_size + 2,
            fontweight='bold',
            pad=15
        )
        
        ax.legend(loc='upper right', fontsize=self.font_size - 1, framealpha=0.9)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    # ===================================================================
    # SCATTER AND CORRELATION PLOTS
    # ===================================================================
    
    def plot_severity_scatter(
        self,
        pet_values: np.ndarray,
        ttc_values: np.ndarray,
        add_regression: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Scatter plot of PET vs TTC with severity coloring.
        
        Args:
            pet_values: PET values (seconds)
            ttc_values: TTC values (seconds)
            add_regression: Add regression line
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        pet_val = self.validate_ssm_data(pet_values, 'PET')
        ttc_val = self.validate_ssm_data(ttc_values, 'TTC')
        
        # Align lengths
        min_len = min(len(pet_val['clean_data']), len(ttc_val['clean_data']))
        pet_data = pet_val['clean_data'][:min_len]
        ttc_data = ttc_val['clean_data'][:min_len]
        
        fig, ax = plt.subplots(figsize=(7.0, 6.5), dpi=self.dpi)
        
        # Severity index (inverse of sum, higher = more severe)
        severity = 1.0 / (pet_data + ttc_data + 0.1)
        
        # FIXED: Use perceptual colormap (plasma instead of YlOrRd)
        scatter = ax.scatter(
            pet_data,
            ttc_data,
            c=severity,
            cmap='plasma',  # Perceptually uniform, colorblind-safe
            s=50,
            alpha=0.6,
            edgecolors='black',
            linewidths=0.5
        )
        
        # Regression line
        if add_regression and len(pet_data) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(pet_data, ttc_data)
            x_line = np.linspace(pet_data.min(), pet_data.max(), 100)
            y_line = slope * x_line + intercept
            
            ax.plot(
                x_line,
                y_line,
                'k--',
                linewidth=2,
                alpha=0.7,
                label=f'Linear fit (R²={r_value**2:.3f}, p={p_value:.4f})'
            )
        
        # Safety threshold lines (configurable)
        pet_safe = self.thresholds['pet']['safe']
        ttc_safe = self.thresholds['ttc']['safe']
        
        ax.axvline(pet_safe, color=self.COLORS['green'], linestyle='--', linewidth=2, alpha=0.7, label=f'PET Safe ({pet_safe}s)')
        ax.axhline(ttc_safe, color=self.COLORS['green'], linestyle='--', linewidth=2, alpha=0.7, label=f'TTC Safe ({ttc_safe}s)')
        
        # Quadrant shading (critical zone)
        ax.axvspan(0, self.thresholds['pet']['serious'], alpha=0.1, color=self.COLORS['red'])
        ax.axhspan(0, self.thresholds['ttc']['critical'], alpha=0.1, color=self.COLORS['red'])
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label('Severity Index (1 / (PET + TTC))', fontsize=self.font_size)
        
        # Labels
        ax.set_xlabel('Post-Encroachment Time (s)', fontsize=self.font_size + 1)
        ax.set_ylabel('Time-To-Collision (s)', fontsize=self.font_size + 1)
        ax.set_title(
            'Conflict Severity: PET vs TTC',
            fontsize=self.font_size + 2,
            fontweight='bold',
            pad=15
        )
        
        ax.legend(loc='upper right', fontsize=self.font_size - 1, framealpha=0.9)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    # ===================================================================
    # CATEGORICAL AND COMPARISON PLOTS
    # ===================================================================
    
    def plot_comparative_boxplot(
        self,
        data_groups: Dict[str, np.ndarray],
        metric_name: str = 'PET',
        show_stats: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Comparative boxplot for multiple groups with statistical annotations.
        
        Args:
            data_groups: Dict of {group_name: data_array}
            metric_name: Name of metric being plotted
            show_stats: Annotate with p-values
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        # Validate all groups
        clean_data = {}
        for name, data in data_groups.items():
            validation = self.validate_ssm_data(data, name)
            if validation['valid']:
                clean_data[name] = validation['clean_data']
        
        if len(clean_data) < 2:
            raise ValueError("Need at least 2 valid groups for comparison")
        
        fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=self.dpi)
        
        # Prepare data
        labels = list(clean_data.keys())
        data_list = [clean_data[label] for label in labels]
        
        # Boxplot
        bp = ax.boxplot(
            data_list,
            labels=labels,
            patch_artist=True,
            notch=True,
            showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='red', markersize=6),
            medianprops=dict(linewidth=2, color='black'),
            boxprops=dict(facecolor=self.COLORS['blue'], alpha=0.7),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5)
        )
        
        # Statistical testing (if requested)
        if show_stats and len(data_list) == 2:
            # Two-sample t-test
            t_stat, p_value = stats.ttest_ind(data_list[0], data_list[1], equal_var=False)
            
            # Add p-value annotation
            y_max = max(np.max(d) for d in data_list)
            y_pos = y_max * 1.1
            
            ax.plot([1, 2], [y_pos, y_pos], 'k-', linewidth=1.5)
            ax.plot([1, 1], [y_pos * 0.98, y_pos], 'k-', linewidth=1.5)
            ax.plot([2, 2], [y_pos * 0.98, y_pos], 'k-', linewidth=1.5)
            
            sig_text = self._format_p_value(p_value)
            ax.text(1.5, y_pos * 1.02, sig_text, ha='center', fontsize=self.font_size)
        
        # Labels
        ax.set_ylabel(f'{metric_name} (s)', fontsize=self.font_size + 1)
        ax.set_title(
            f'{metric_name} Comparison Across Groups',
            fontsize=self.font_size + 2,
            fontweight='bold',
            pad=15
        )
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    # ===================================================================
    # HEATMAPS AND DENSITY PLOTS
    # ===================================================================
    
    def plot_conflict_density_map(
        self,
        pet_values: np.ndarray,
        custom_bands: Optional[List[float]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Horizontal bar chart showing conflict density by severity bands.
        
        Args:
            pet_values: PET values (seconds)
            custom_bands: Custom bin edges (default: FHWA guidelines)
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        validation = self.validate_ssm_data(pet_values, 'PET')
        
        if not validation['valid']:
            raise ValueError(f"Invalid data: {validation['errors']}")
        
        data = validation['clean_data']
        pet_thresh = self.thresholds['pet']
        
        # Define severity bands (using configurable thresholds)
        if custom_bands is None:
            bands = [0, pet_thresh['critical'], pet_thresh['serious'], 
                     pet_thresh['moderate'], 2.0, 3.0, pet_thresh['safe'], float('inf')]
            labels = [
                f'0-{pet_thresh["critical"]}s (Critical)',
                f'{pet_thresh["critical"]}-{pet_thresh["serious"]}s (Serious)',
                f'{pet_thresh["serious"]}-{pet_thresh["moderate"]}s (Moderate)',
                f'{pet_thresh["moderate"]}-2.0s (Slight)',
                '2.0-3.0s (Low)',
                f'3.0-{pet_thresh["safe"]}s (Minor)',
                f'>{pet_thresh["safe"]}s (Safe)'
            ]
            colors = [
                self.COLORS['red'],
                self.COLORS['orange'],
                self.COLORS['yellow'],
                '#90EE90',  # Light green
                self.COLORS['green'],
                self.COLORS['cyan'],
                self.COLORS['blue']
            ]
        else:
            bands = custom_bands
            labels = [f'{bands[i]:.1f}-{bands[i+1]:.1f}s' for i in range(len(bands)-1)]
            colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(labels)))
        
        # Count events in each band
        counts = []
        for i in range(len(bands) - 1):
            if bands[i+1] == float('inf'):
                count = int(np.sum(data >= bands[i]))
            else:
                count = int(np.sum((data >= bands[i]) & (data < bands[i+1])))
            counts.append(count)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(9.0, 6.0), dpi=self.dpi)
        
        y_pos = np.arange(len(labels))
        bars = ax.barh(
            y_pos,
            counts,
            color=colors,
            edgecolor='black',
            linewidth=0.8
        )
        
        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            if count > 0:
                ax.text(
                    bar.get_width() + max(counts) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{count} ({100*count/len(data):.1f}%)',
                    va='center',
                    fontsize=self.font_size - 1
                )
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=self.font_size - 1)
        ax.set_xlabel('Number of Conflicts', fontsize=self.font_size + 1)
        ax.set_title(
            f'Conflict Severity Distribution (N={len(data):,})',
            fontsize=self.font_size + 2,
            fontweight='bold',
            pad=15
        )
        
        ax.invert_yaxis()  # Most severe at top
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    # ===================================================================
    # CUMULATIVE DISTRIBUTION
    # ===================================================================
    
    def plot_cumulative_distribution(
        self,
        data_groups: Dict[str, np.ndarray],
        metric_name: str = 'PET',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Empirical cumulative distribution function (ECDF) plot.
        
        Args:
            data_groups: Dict of {group_name: data_array}
            metric_name: Name of metric
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=self.dpi)
        
        color_cycle = [self.COLORS[c] for c in ['blue', 'orange', 'green', 'purple', 'red']]
        
        for i, (name, data) in enumerate(data_groups.items()):
            validation = self.validate_ssm_data(data, name)
            if not validation['valid']:
                continue
            
            clean = validation['clean_data']
            sorted_data = np.sort(clean)
            y_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            
            ax.plot(
                sorted_data,
                y_values,
                linewidth=2,
                label=f'{name} (N={len(clean):,})',
                color=color_cycle[i % len(color_cycle)]
            )
        
        # Safety thresholds (configurable)
        if metric_name.upper() == 'PET':
            thresh = self.thresholds['pet']
            ax.axvline(thresh['serious'], color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Serious ({thresh["serious"]}s)')
            ax.axvline(thresh['safe'], color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Safe ({thresh["safe"]}s)')
        elif metric_name.upper() == 'TTC':
            thresh = self.thresholds['ttc']
            ax.axvline(thresh['critical'], color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Critical ({thresh["critical"]}s)')
            ax.axvline(thresh['safe'], color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Safe ({thresh["safe"]}s)')
        
        ax.set_xlabel(f'{metric_name} (s)', fontsize=self.font_size + 1)
        ax.set_ylabel('Cumulative Probability', fontsize=self.font_size + 1)
        ax.set_title(
            f'Empirical CDF: {metric_name}',
            fontsize=self.font_size + 2,
            fontweight='bold',
            pad=15
        )
        
        ax.legend(loc='best', fontsize=self.font_size - 1)
        ax.set_ylim(0, 1)
        ax.set_xlim(left=0)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    # ===================================================================
    # HELPER METHODS
    # ===================================================================
    
    def _format_p_value(self, p: float) -> str:
        """Format p-value for display (APA style)."""
        if p < 0.001:
            return 'p < 0.001***'
        elif p < 0.01:
            return f'p = {p:.3f}**'
        elif p < 0.05:
            return f'p = {p:.3f}*'
        else:
            return f'p = {p:.3f} (ns)'
    
    def _save_figure(self, fig: plt.Figure, save_path: str):
        """Save figure in PNG and PDF formats."""
        # Save PNG
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight', format='png')
        
        # Save PDF (vector graphics)
        pdf_path = save_path.replace('.png', '.pdf')
        if pdf_path != save_path:  # Only if .png was in filename
            fig.savefig(pdf_path, bbox_inches='tight', format='pdf')


# ===================================================================
# CONVENIENCE FUNCTIONS
# ===================================================================

def plot_pet_distribution(
    pet_values: np.ndarray,
    style: str = 'journal',
    save_path: Optional[str] = None
) -> plt.Figure:
    """Standalone PET distribution plot."""
    plotter = SSMPlotter(style=style)
    return plotter.plot_pet_distribution(pet_values, save_path=save_path)


def plot_ttc_time_series(
    timestamps: np.ndarray,
    ttc_values: np.ndarray,
    style: str = 'journal',
    save_path: Optional[str] = None
) -> plt.Figure:
    """Standalone TTC time series plot."""
    plotter = SSMPlotter(style=style)
    return plotter.plot_ttc_time_series(timestamps, ttc_values, save_path=save_path)


def plot_severity_scatter(
    pet_values: np.ndarray,
    ttc_values: np.ndarray,
    style: str = 'journal',
    save_path: Optional[str] = None
) -> plt.Figure:
    """Standalone severity scatter plot."""
    plotter = SSMPlotter(style=style)
    return plotter.plot_severity_scatter(pet_values, ttc_values, save_path=save_path)


def plot_conflict_density_map(
    pet_values: np.ndarray,
    style: str = 'journal',
    save_path: Optional[str] = None
) -> plt.Figure:
    """Standalone conflict density plot."""
    plotter = SSMPlotter(style=style)
    return plotter.plot_conflict_density_map(pet_values, save_path=save_path)


def plot_comparative_boxplot(
    data_groups: Dict[str, np.ndarray],
    metric_name: str = 'PET',
    style: str = 'journal',
    save_path: Optional[str] = None
) -> plt.Figure:
    """Standalone comparative boxplot."""
    plotter = SSMPlotter(style=style)
    return plotter.plot_comparative_boxplot(data_groups, metric_name, save_path=save_path)


def plot_cumulative_distribution(
    data_groups: Dict[str, np.ndarray],
    metric_name: str = 'PET',
    style: str = 'journal',
    save_path: Optional[str] = None
) -> plt.Figure:
    """Standalone ECDF plot."""
    plotter = SSMPlotter(style=style)
    return plotter.plot_cumulative_distribution(data_groups, metric_name, save_path=save_path)


def plot_correlation_heatmap(
    data_dict: Dict[str, np.ndarray],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Correlation heatmap for multiple SSM metrics."""
    if not HAS_PANDAS:
        raise ImportError("pandas required for correlation heatmap. Install with: pip install pandas")
    
    # Create DataFrame
    df = pd.DataFrame(data_dict)
    
    # Compute correlation matrix
    corr = df.corr()
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Annotations
    for i in range(len(corr)):
        for j in range(len(corr)):
            text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=9)
    
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.columns)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Pearson Correlation', fontsize=10)
    
    ax.set_title('SSM Correlation Matrix', fontsize=12, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', format='pdf')
    
    return fig


def plot_temporal_heatmap(
    timestamps: np.ndarray,
    pet_values: np.ndarray,
    time_bins: int = 20,
    severity_bins: int = 10,
    save_path: Optional[str] = None
) -> plt.Figure:
    """2D heatmap showing temporal evolution of conflict severity."""
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Create 2D histogram
    H, xedges, yedges = np.histogram2d(
        timestamps,
        pet_values,
        bins=[time_bins, severity_bins]
    )
    
    # Smooth with Gaussian filter
    H_smooth = gaussian_filter(H.T, sigma=1.0)
    
    im = ax.imshow(
        H_smooth,
        aspect='auto',
        origin='lower',
        extent=[timestamps.min(), timestamps.max(), pet_values.min(), pet_values.max()],
        cmap='YlOrRd',
        interpolation='bilinear'
    )
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Conflict Density', fontsize=10)
    
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('PET (s)', fontsize=11)
    ax.set_title('Temporal Conflict Density Heatmap', fontsize=12, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', format='pdf')
    
    return fig
