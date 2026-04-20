import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.ticker import MaxNLocator
from typing import List, Dict, Tuple, Optional
import warnings

__all__ = ['SSMPlotter', 'plot_pet_distribution', 'plot_ttc_time_series', 'plot_severity_scatter', 'plot_conflict_density_map']


class SSMPlotter:
    """Industry-standard visualizer for Surrogate Safety Measures.
    Designed for peer-reviewed traffic safety journals.
    Supports PET (Post-Encroachment Time), TTC (Time-To-Collision).
    """

    def __init__(self, style: str = 'journal', dpi: int = 300, font_size: float = 10.0):
        self.style = style
        self.dpi = dpi
        self.font_size = font_size
        self._setup_style()

    def _setup_style(self):
        if self.style == 'journal':
            plt.style.use('classic')
            matplotlib.rcParams.update({
                'font.size': self.font_size,
                'axes.titlesize': self.font_size + 2,
                'axes.labelsize': self.font_size + 1,
                'xtick.labelsize': self.font_size - 1,
                'ytick.labelsize': self.font_size - 1,
                'legend.fontsize': self.font_size - 1,
                'figure.figsize': (6.5, 5.0),
                'axes.linewidth': 1.0,
            })
        elif self.style == 'presentation':
            plt.style.use('seaborn-v0_8-whitegrid')
            matplotlib.rcParams.update({'font.size': self.font_size + 2, 'figure.figsize': (8.0, 6.0)})

    def validate_ssm_data(self, data: np.ndarray, metric_name: str) -> Dict:
        results = {'valid': True, 'clean_data': data.copy(), 'warnings': [], 'errors': []}
        if data is None:
            results['valid'] = False
            results['errors'].append(f"{metric_name}: Input data is None")
            return results
        data = np.asarray(data)
        if data.ndim != 1:
            results['valid'] = False
            results['errors'].append(f"{metric_name}: Data must be 1D array")
            return results
        if len(data) == 0:
            results['valid'] = False
            results['errors'].append(f"{metric_name}: Data array is empty")
            return results
        nan_count = int(np.sum(np.isnan(data)))
        inf_count = int(np.sum(np.isinf(data)))
        if nan_count > 0:
            results['warnings'].append(f"{metric_name}: {nan_count} NaN values removed")
        if inf_count > 0:
            results['warnings'].append(f"{metric_name}: {inf_count} Inf values removed")
        results['clean_data'] = data[np.isfinite(data)]
        if len(results['clean_data']) == 0:
            results['valid'] = False
            results['errors'].append(f"{metric_name}: No valid finite values")
        return results

    def plot_pet_distribution(self, pet_values: np.ndarray, save_path: Optional[str] = None) -> plt.Figure:
        validation = self.validate_ssm_data(pet_values, 'PET')
        if not validation['valid']:
            raise ValueError(validation['errors'])
        data = validation['clean_data']
        fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=self.dpi)
        ax.hist(data, bins=40, density=True, alpha=0.7, color='steelblue',
                edgecolor='black', linewidth=0.5, label='PET Distribution')
        thresholds = {'Critical': 1.0, 'Warning': 3.0, 'Safe': 5.0}
        colors = {'Critical': 'red', 'Warning': 'orange', 'Safe': 'green'}
        for label, val in thresholds.items():
            ax.axvline(val, color=colors[label], linestyle='--', linewidth=2.0,
                       label=f'{label} ({val}s)')
        stats_text = f"N={len(data):,}\nMean={np.mean(data):.2f}s\nStd={np.std(data):.2f}s\nMin={np.min(data):.2f}s"
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=self.font_size - 1,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))
        ax.set_xlabel('Post-Encroachment Time (seconds)', fontsize=self.font_size + 1)
        ax.set_ylabel('Probability Density', fontsize=self.font_size + 1)
        ax.set_title('PET Distribution with Safety Thresholds', fontsize=self.font_size + 2, fontweight='bold')
        ax.legend(loc='upper right', fontsize=self.font_size - 1)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        return fig

    def plot_ttc_time_series(self, timestamps: np.ndarray, ttc_values: np.ndarray,
                             save_path: Optional[str] = None) -> plt.Figure:
        validation = self.validate_ssm_data(ttc_values, 'TTC')
        if not validation['valid']:
            raise ValueError(validation['errors'])
        data = validation['clean_data']
        if len(timestamps) != len(data):
            timestamps = np.arange(len(data))
        fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=self.dpi)
        ax.plot(timestamps, data, color='navy', linewidth=1.5, label='TTC')
        ax.axhspan(0, 2.0, alpha=0.3, color='red', label='Critical Zone (<2s)')
        ax.axhspan(2.0, 5.0, alpha=0.2, color='orange', label='Warning Zone (2-5s)')
        ax.axhline(2.0, color='red', linestyle='--', linewidth=1.5)
        ax.axhline(5.0, color='orange', linestyle='--', linewidth=1.5)
        ax.set_xlabel('Time (seconds)', fontsize=self.font_size + 1)
        ax.set_ylabel('Time-To-Collision (seconds)', fontsize=self.font_size + 1)
        ax.set_title('TTC Time Series Analysis', fontsize=self.font_size + 2, fontweight='bold')
        ax.legend(loc='upper right', fontsize=self.font_size - 1)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        return fig

    def plot_severity_scatter(self, pet_values: np.ndarray, ttc_values: np.ndarray,
                              save_path: Optional[str] = None) -> plt.Figure:
        pet_val = self.validate_ssm_data(pet_values, 'PET')
        ttc_val = self.validate_ssm_data(ttc_values, 'TTC')
        min_len = min(len(pet_val['clean_data']), len(ttc_val['clean_data']))
        pet_data = pet_val['clean_data'][:min_len]
        ttc_data = ttc_val['clean_data'][:min_len]
        fig, ax = plt.subplots(figsize=(6.5, 6.0), dpi=self.dpi)
        severity = 1.0 / (pet_data + ttc_data + 0.1)
        scatter = ax.scatter(pet_data, ttc_data, c=severity, cmap='Reds', s=40, alpha=0.7)
        ax.axvline(5.0, color='green', linestyle='--', linewidth=2.0, label='PET Safe (5s)')
        ax.axhline(5.0, color='green', linestyle='--', linewidth=2.0)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Severity Index', fontsize=self.font_size)
        ax.set_xlabel('PET (seconds)', fontsize=self.font_size + 1)
        ax.set_ylabel('TTC (seconds)', fontsize=self.font_size + 1)
        ax.set_title('Conflict Severity: PET vs TTC', fontsize=self.font_size + 2, fontweight='bold')
        ax.legend(loc='upper right', fontsize=self.font_size - 1)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        return fig

    def plot_conflict_density_map(self, pet_values: np.ndarray, save_path: Optional[str] = None) -> plt.Figure:
        validation = self.validate_ssm_data(pet_values, 'PET')
        if not validation['valid']:
            raise ValueError(validation['errors'])
        data = validation['clean_data']
        bands = [0, 1, 2, 3, 5, 10, float('inf')]
        labels = ['0-1s (Critical)', '1-2s (Severe)', '2-3s (Warning)',
                  '3-5s (Caution)', '5-10s (Safe)', '>10s (Very Safe)']
        counts = [int(np.sum((data >= bands[i]) & (data < bands[i+1]))) if bands[i+1] != float('inf')
                  else int(np.sum(data >= bands[i])) for i in range(len(bands)-1)]
        fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=self.dpi)
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, counts, edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=self.font_size - 1)
        ax.set_xlabel('Number of Conflicts', fontsize=self.font_size + 1)
        ax.set_title('Conflict Severity Distribution by PET Bands', fontsize=self.font_size + 2, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        return fig


def plot_pet_distribution(pet_values: np.ndarray, style: str = 'journal',
                          save_path: Optional[str] = None) -> plt.Figure:
    plotter = SSMPlotter(style=style)
    return plotter.plot_pet_distribution(pet_values, save_path)


def plot_ttc_time_series(timestamps: np.ndarray, ttc_values: np.ndarray,
                         style: str = 'journal', save_path: Optional[str] = None) -> plt.Figure:
    plotter = SSMPlotter(style=style)
    return plotter.plot_ttc_time_series(timestamps, ttc_values, save_path)
