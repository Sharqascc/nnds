"""
PET Diffusion Analysis Module

Advanced analysis and comparison of real vs diffusion-generated trajectories
using PET-like metrics. Provides:
- PET metric computation for generated trajectories
- Real vs sampled trajectory comparison
- Statistical analysis and significance testing
- Automatic visualization generation
- Results export (CSV, JSON, Parquet)
- Publication-ready reports

Integrates with:
- analysis.visualization.pet_diffusion_plots for automatic plotting
- traffic_diffusion for model evaluation
"""

import ast
import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Callable

import numpy as np
import pandas as pd
import torch
from scipy import stats

# Optional progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    warnings.warn("tqdm not available - progress bars disabled. Install with: pip install tqdm")
    # Fallback: dummy tqdm
    def tqdm(iterable, **kwargs):
        return iterable

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = [
    'PETDiffusionAnalyzer',
    'compute_pet_like_metrics',
    'compare_realPET_samplePET',
    'parse_trajectory',
    'compute_error_metrics',
    'perform_statistical_tests'
]


def parse_trajectory(cell: Any) -> Optional[np.ndarray]:
    """
    Parse trajectory from CSV cell (string representation of list/array).

    Args:
        cell: Cell value (string, list, or array)

    Returns:
        Numpy array or None if parsing fails
    """
    try:
        if isinstance(cell, str):
            return np.array(ast.literal_eval(cell), dtype=float)
        elif isinstance(cell, (list, np.ndarray)):
            return np.array(cell, dtype=float)
        else:
            return None
    except Exception as e:
        logger.warning(f"Failed to parse trajectory: {e}")
        return None


def compute_distance_to_threshold(
    traj: np.ndarray,
    d_thresh: float = 1.0
) -> Tuple[np.ndarray, Optional[int]]:
    """
    Compute pairwise distances and find first threshold crossing.

    Args:
        traj: Trajectory array of shape (T, 4) [x1, y1, x2, y2]
        d_thresh: Distance threshold in meters

    Returns:
        Tuple of (distances array, first hit step index or None)

    Raises:
        ValueError: If trajectory shape is invalid
    """
    if traj.shape[1] != 4:
        raise ValueError(f"Expected trajectory with 4 columns, got {traj.shape[1]}")

    distances = np.linalg.norm(traj[:, 0:2] - traj[:, 2:4], axis=-1)

    hit_idxs = np.where(distances < d_thresh)[0]
    first_hit = int(hit_idxs[0]) if len(hit_idxs) > 0 else None

    return distances, first_hit


def compute_error_metrics(
    real_values: np.ndarray,
    predicted_values: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive error metrics.

    Args:
        real_values: Ground truth values
        predicted_values: Predicted/sampled values

    Returns:
        Dictionary of error metrics

    Raises:
        ValueError: If arrays have different lengths or are empty
    """
    if len(real_values) == 0 or len(predicted_values) == 0:
        raise ValueError("Input arrays cannot be empty")

    if len(real_values) != len(predicted_values):
        raise ValueError(f"Array length mismatch: {len(real_values)} vs {len(predicted_values)}")

    errors = predicted_values - real_values
    abs_errors = np.abs(errors)
    sq_errors = errors ** 2

    # Avoid division by zero for R²
    var_real = np.sum((real_values - np.mean(real_values))**2)
    r_squared = 1 - (np.sum(sq_errors) / var_real) if var_real > 0 else 0.0

    metrics = {
        'mae': float(np.mean(abs_errors)),
        'rmse': float(np.sqrt(np.mean(sq_errors))),
        'mse': float(np.mean(sq_errors)),
        'mean_error': float(np.mean(errors)),
        'std_error': float(np.std(errors)),
        'median_error': float(np.median(errors)),
        'max_error': float(np.max(abs_errors)),
        'r_squared': float(r_squared)
    }

    return metrics


def perform_statistical_tests(
    real_values: np.ndarray,
    predicted_values: np.ndarray
) -> Dict[str, Any]:
    """
    Perform statistical significance tests.

    Args:
        real_values: Ground truth values
        predicted_values: Predicted/sampled values

    Returns:
        Dictionary of test results

    Raises:
        ValueError: If arrays have different lengths or are too small
    """
    if len(real_values) < 3 or len(predicted_values) < 3:
        raise ValueError("Need at least 3 samples for statistical tests")

    if len(real_values) != len(predicted_values):
        raise ValueError(f"Array length mismatch: {len(real_values)} vs {len(predicted_values)}")

    # Paired t-test
    t_stat, t_pval = stats.ttest_rel(predicted_values, real_values)

    # Wilcoxon signed-rank test (non-parametric)
    w_stat, w_pval = stats.wilcoxon(predicted_values, real_values)

    # Pearson correlation
    r_corr, r_pval = stats.pearsonr(real_values, predicted_values)

    # Spearman correlation
    rho_corr, rho_pval = stats.spearmanr(real_values, predicted_values)

    return {
        'paired_t_test': {'statistic': float(t_stat), 'p_value': float(t_pval)},
        'wilcoxon_test': {'statistic': float(w_stat), 'p_value': float(w_pval)},
        'pearson_corr': {'coefficient': float(r_corr), 'p_value': float(r_pval)},
        'spearman_corr': {'coefficient': float(rho_corr), 'p_value': float(rho_pval)}
    }


class PETDiffusionAnalyzer:
    """
    Comprehensive analyzer for diffusion-generated trajectories using PET metrics.

    Features:
    - PET-like metric computation
    - Real vs sampled comparison
    - Statistical analysis
    - Automatic visualization
    - Results export
    """

    def __init__(
        self,
        d_thresh: float = 1.0,
        fps: float = 30.0,
        scale: float = 1.0,
        output_dir: str = 'outputs/diffusion_analysis',
        auto_visualize: bool = True,
        save_results: bool = True
    ):
        """
        Args:
            d_thresh: Distance threshold in meters for PET
            fps: Frames per second for time conversion
            scale: Scaling factor for trajectory normalization
            output_dir: Directory for saving results
            auto_visualize: Automatically generate plots
            save_results: Save results to files
        """
        self.d_thresh = d_thresh
        self.fps = fps
        self.scale = scale
        self.output_dir = Path(output_dir)
        self.auto_visualize = auto_visualize
        self.save_results = save_results

        # Create output directory
        if self.save_results:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Output directory: {self.output_dir}")

        # Try to import visualization module
        self.viz_available = False
        if auto_visualize:
            try:
                from analysis.visualization import DiffusionPETPlotter
                self.plotter = DiffusionPETPlotter(dpi=300, save_pdf=True)
                self.viz_available = True
                logger.info("✅ Visualization module loaded")
            except ImportError as e:
                logger.warning(f"⚠️ Visualization module not available: {e}")
                logger.warning("   Install dependencies: pip install matplotlib seaborn")
                logger.info("   Continuing without automatic visualization...")

    def steps_to_seconds(self, steps: Optional[float]) -> Optional[float]:
        """Convert step index to seconds."""
        return steps / self.fps if steps is not None else None

    def _validate_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        """
        Validate batch structure.

        Raises:
            ValueError: If batch is invalid
        """
        if batch is None:
            raise ValueError("Batch cannot be None")

        required_keys = ['past', 'future']
        for key in required_keys:
            if key not in batch:
                raise KeyError(f"Batch must contain '{key}' key. Found keys: {list(batch.keys())}")

        # Check shapes
        if batch['past'].shape[0] == 0:
            raise ValueError("Batch is empty (batch size = 0)")

        if batch['past'].shape[0] != batch['future'].shape[0]:
            raise ValueError(
                f"Batch size mismatch: past={batch['past'].shape[0]}, future={batch['future'].shape[0]}"
            )

    def compute_pet_like_metrics(
        self,
        batch: Dict[str, torch.Tensor],
        sample_future_fn: Callable,
        noise_scale: float = 0.01,
        verbose: bool = True
    ) -> Tuple[List[Tuple[Optional[float], Optional[float]]], pd.DataFrame]:
        """
        Compute PET-like metrics comparing real and sampled futures.

        Args:
            batch: Batch dictionary with 'past', 'future', 'idx'
            sample_future_fn: Function to generate sampled futures
            noise_scale: Noise scale for sampling
            verbose: Print detailed output

        Returns:
            Tuple of (pet_pairs list, results DataFrame)

        Raises:
            ValueError: If batch is invalid or sample_future_fn is None
        """
        # Input validation
        if sample_future_fn is None:
            raise ValueError("sample_future_fn cannot be None")

        self._validate_batch(batch)

        B = batch["past"].shape[0]

        # Handle empty batch
        if B == 0:
            logger.warning("Empty batch provided, returning empty results")
            return [], pd.DataFrame()

        # Get trajectories in world coordinates
        future_norm = batch["future"]
        real_world = future_norm * self.scale

        try:
            sample_world = sample_future_fn(batch, max_B=B, noise_scale=noise_scale)
        except Exception as e:
            logger.error(f"Error in sample_future_fn: {e}")
            raise

        real_np = real_world.cpu().numpy()
        sample_np = sample_world.cpu().numpy()

        pet_pairs = []
        records = []

        # Process each example
        iterator = range(B)
        if verbose and TQDM_AVAILABLE:
            iterator = tqdm(iterator, desc="Computing PET metrics")

        for b in iterator:
            real_traj = real_np[b]
            sample_traj = sample_np[b]

            try:
                # Compute distances and find PET
                _, pet_real = compute_distance_to_threshold(real_traj, self.d_thresh)
                _, pet_sample = compute_distance_to_threshold(sample_traj, self.d_thresh)
            except Exception as e:
                logger.warning(f"Error computing PET for example {b}: {e}")
                pet_real, pet_sample = None, None

            pet_pairs.append((pet_real, pet_sample))

            records.append({
                'example_idx': b,
                'pet_real_steps': pet_real,
                'pet_sample_steps': pet_sample,
                'pet_real_sec': self.steps_to_seconds(pet_real),
                'pet_sample_sec': self.steps_to_seconds(pet_sample),
                'has_real_pet': pet_real is not None,
                'has_sample_pet': pet_sample is not None,
                'both_defined': pet_real is not None and pet_sample is not None
            })

        df = pd.DataFrame(records)

        # Compute statistics
        both_defined = df[df['both_defined']]

        if verbose:
            self._print_pet_summary(df, both_defined)

        # Save results
        if self.save_results:
            csv_path = self.output_dir / 'pet_like_metrics.csv'
            df.to_csv(csv_path, index=False)
            logger.info(f"✅ Results saved to {csv_path}")

        # Generate visualizations
        if self.viz_available and self.auto_visualize and len(both_defined) > 0:
            self._generate_visualizations(pet_pairs, both_defined)

        return pet_pairs, df

    def compare_with_ground_truth(
        self,
        df_pet_path: str,
        batch: Dict[str, torch.Tensor],
        sample_future_fn: Callable,
        noise_scale: float = 0.01,
        verbose: bool = True
    ) -> Tuple[List[Tuple], pd.DataFrame]:
        """
        Compare ground truth PET (from CSV) with PET-like metrics from model.

        Args:
            df_pet_path: Path to CSV with ground truth PET values
            batch: Batch dictionary
            sample_future_fn: Sampling function
            noise_scale: Noise scale
            verbose: Print output

        Returns:
            Tuple of (records list, results DataFrame)

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If batch is invalid
        """
        # Input validation
        if sample_future_fn is None:
            raise ValueError("sample_future_fn cannot be None")

        self._validate_batch(batch)

        # Load ground truth
        pet_path = Path(df_pet_path)
        if not pet_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {df_pet_path}")

        df_pet = pd.read_csv(df_pet_path)
        logger.info(f"Loaded {len(df_pet)} ground truth PET records from {pet_path.name}")

        idxs = batch["idx"].cpu().numpy()
        B = len(idxs)

        # Handle empty batch
        if B == 0:
            logger.warning("Empty batch provided, returning empty results")
            return [], pd.DataFrame()

        # Generate trajectories
        future_norm = batch["future"]
        real_world = future_norm * self.scale

        try:
            sample_world = sample_future_fn(batch, max_B=B, noise_scale=noise_scale)
        except Exception as e:
            logger.error(f"Error in sample_future_fn: {e}")
            raise

        real_np = real_world.cpu().numpy()
        sample_np = sample_world.cpu().numpy()

        records = []

        iterator = range(B)
        if verbose and TQDM_AVAILABLE:
            iterator = tqdm(iterator, desc="Comparing with ground truth")

        for b in iterator:
            row_idx = int(idxs[b])

            # Ground truth PET from CSV
            if row_idx >= len(df_pet):
                logger.warning(f"Row index {row_idx} out of bounds (CSV has {len(df_pet)} rows)")
                continue

            true_pet_sec = float(df_pet.loc[row_idx, "PET"])

            # Compute PET-like from trajectories
            real_traj = real_np[b]
            sample_traj = sample_np[b]

            try:
                _, pet_like_real = compute_distance_to_threshold(real_traj, self.d_thresh)
                _, pet_like_sample = compute_distance_to_threshold(sample_traj, self.d_thresh)
            except Exception as e:
                logger.warning(f"Error computing PET for example {b}: {e}")
                pet_like_real, pet_like_sample = None, None

            records.append({
                'row_idx': row_idx,
                'true_pet_sec': true_pet_sec,
                'pet_like_real_steps': pet_like_real,
                'pet_like_sample_steps': pet_like_sample,
                'pet_like_real_sec': self.steps_to_seconds(pet_like_real),
                'pet_like_sample_sec': self.steps_to_seconds(pet_like_sample),
                'error_real': (self.steps_to_seconds(pet_like_real) - true_pet_sec) if pet_like_real else None,
                'error_sample': (self.steps_to_seconds(pet_like_sample) - true_pet_sec) if pet_like_sample else None
            })

        df = pd.DataFrame(records)

        # Compute error metrics
        valid_real = df[df['pet_like_real_sec'].notna()]
        valid_sample = df[df['pet_like_sample_sec'].notna()]

        if verbose:
            self._print_ground_truth_summary(df, valid_real, valid_sample)

        # Save results
        if self.save_results:
            csv_path = self.output_dir / 'ground_truth_comparison.csv'
            df.to_csv(csv_path, index=False)
            logger.info(f"✅ Results saved to {csv_path}")

        # Generate visualizations
        if self.viz_available and self.auto_visualize:
            self._generate_ground_truth_visualizations(df)

        # Convert to tuple format for backward compatibility
        records_tuple = [
            (r['row_idx'], r['true_pet_sec'], r['pet_like_real_steps'], r['pet_like_sample_steps'])
            for _, r in df.iterrows()
        ]

        return records_tuple, df

    def _print_pet_summary(self, df: pd.DataFrame, both_defined: pd.DataFrame):
        """Print summary statistics for PET metrics."""
        print("\n" + "="*60)
        print("PET-LIKE METRICS SUMMARY")
        print("="*60)
        print(f"Distance threshold: {self.d_thresh} m")
        print(f"Total examples: {len(df)}")
        print(f"Real PET defined: {df['has_real_pet'].sum()} ({100*df['has_real_pet'].mean():.1f}%)")
        print(f"Sample PET defined: {df['has_sample_pet'].sum()} ({100*df['has_sample_pet'].mean():.1f}%)")
        print(f"Both defined: {len(both_defined)} ({100*len(both_defined)/len(df):.1f}%)")

        if len(both_defined) > 0:
            errors_steps = both_defined['pet_sample_steps'] - both_defined['pet_real_steps']
            errors_sec = both_defined['pet_sample_sec'] - both_defined['pet_real_sec']

            print(f"\nError Statistics (steps):")
            print(f"  Mean: {errors_steps.mean():.3f} ± {errors_steps.std():.3f}")
            print(f"  Median: {errors_steps.median():.3f}")
            print(f"  Range: [{errors_steps.min():.3f}, {errors_steps.max():.3f}]")

            print(f"\nError Statistics (seconds):")
            print(f"  Mean: {errors_sec.mean():.3f} ± {errors_sec.std():.3f}")
            print(f"  Median: {errors_sec.median():.3f}")
            print(f"  MAE: {errors_sec.abs().mean():.3f}")
            print(f"  RMSE: {np.sqrt((errors_sec**2).mean()):.3f}")

        print("="*60 + "\n")

    def _print_ground_truth_summary(
        self,
        df: pd.DataFrame,
        valid_real: pd.DataFrame,
        valid_sample: pd.DataFrame
    ):
        """Print summary for ground truth comparison."""
        print("\n" + "="*60)
        print("GROUND TRUTH COMPARISON SUMMARY")
        print("="*60)
        print(f"Total examples: {len(df)}")
        print(f"PET-like real valid: {len(valid_real)} ({100*len(valid_real)/len(df):.1f}%)")
        print(f"PET-like sample valid: {len(valid_sample)} ({100*len(valid_sample)/len(df):.1f}%)")

        if len(valid_real) > 0:
            print(f"\nReal Trajectory Error vs Ground Truth:")
            print(f"  MAE: {valid_real['error_real'].abs().mean():.3f} s")
            print(f"  RMSE: {np.sqrt((valid_real['error_real']**2).mean()):.3f} s")

        if len(valid_sample) > 0:
            print(f"\nSampled Trajectory Error vs Ground Truth:")
            print(f"  MAE: {valid_sample['error_sample'].abs().mean():.3f} s")
            print(f"  RMSE: {np.sqrt((valid_sample['error_sample']**2).mean()):.3f} s")

        print("="*60 + "\n")

    def _generate_visualizations(self, pet_pairs: List, both_defined: pd.DataFrame):
        """Generate automatic visualizations."""
        try:
            # Histogram of errors
            self.plotter.plot_pet_like_histogram(
                pet_pairs,
                save_path=str(self.output_dir / 'pet_error_histogram.png')
            )
            logger.info("✅ Generated error histogram")
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate histogram: {e}")

    def _generate_ground_truth_visualizations(self, df: pd.DataFrame):
        """Generate visualizations for ground truth comparison."""
        try:
            # Convert to records format (optimized)
            records = list(df[['row_idx', 'true_pet_sec', 'pet_like_real_steps', 'pet_like_sample_steps']].itertuples(index=False, name=None))

            # Scatter plot
            self.plotter.plot_true_vs_pet_like(
                records,
                add_regression=True,
                save_path=str(self.output_dir / 'true_vs_petlike_scatter.png')
            )

            # Error analysis
            self.plotter.plot_true_vs_sample_delta(
                records,
                add_trend=True,
                save_path=str(self.output_dir / 'error_vs_truth.png')
            )

            logger.info("✅ Generated ground truth comparison plots")
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate plots: {e}")


# Backward compatible functions
def compute_pet_like_metrics(
    batch: Dict[str, torch.Tensor],
    sample_future_fn: Callable,
    scale: float,
    noise_scale: float = 0.01,
    d_thresh: float = 1.0
) -> List[Tuple[Optional[float], Optional[float]]]:
    """
    Backward compatible function for computing PET-like metrics.

    Recommended: Use PETDiffusionAnalyzer class for more features.
    """
    analyzer = PETDiffusionAnalyzer(
        d_thresh=d_thresh,
        scale=scale,
        auto_visualize=False,
        save_results=False
    )

    pet_pairs, _ = analyzer.compute_pet_like_metrics(
        batch,
        sample_future_fn,
        noise_scale=noise_scale,
        verbose=True
    )

    return pet_pairs


def compare_realPET_samplePET(
    df_pet_path: str,
    batch: Dict[str, torch.Tensor],
    sample_future_fn: Callable,
    scale: float,
    noise_scale: float = 0.01,
    d_thresh: float = 1.0
) -> List[Tuple]:
    """
    Backward compatible function for ground truth comparison.

    Recommended: Use PETDiffusionAnalyzer class for more features.
    """
    analyzer = PETDiffusionAnalyzer(
        d_thresh=d_thresh,
        scale=scale,
        auto_visualize=False,
        save_results=False
    )

    records, _ = analyzer.compare_with_ground_truth(
        df_pet_path,
        batch,
        sample_future_fn,
        noise_scale=noise_scale,
        verbose=True
    )

    return records
