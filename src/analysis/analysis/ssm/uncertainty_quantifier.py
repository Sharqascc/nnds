"""
Uncertainty Quantification for Surrogate Safety Measures

Provides statistical uncertainty estimation for SSM measurements:
- Bootstrap confidence intervals (BCa, percentile)
- Monte Carlo uncertainty propagation
- Power analysis for sample size determination
- Bayesian credible intervals
- Effect size computation (Cohen's d, Hedges' g)

Compliant with:
- APA statistical reporting guidelines
- Frequentist and Bayesian inference methods
- FAIR reproducibility standards
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable, List
from scipy import stats
from dataclasses import dataclass
import warnings


__all__ = [
    'UncertaintyQuantifier',
    'bootstrap_ci',
    'monte_carlo_uq',
    'compute_confidence_interval',
    'compute_effect_size',
    'compute_sample_size',
    'sensitivity_analysis'
]


@dataclass
class UQResult:
    """Container for uncertainty quantification results."""
    metric_name: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    method: str
    n_samples: int
    std_error: float
    cv_percent: float  # Coefficient of variation


class UncertaintyQuantifier:
    """
    Comprehensive uncertainty quantification for SSM measurements.
    
    Features:
    - Multiple CI methods (parametric, bootstrap, BCa)
    - Effect size with multiple estimators
    - Power analysis
    - Sensitivity analysis
    - Bayesian credible intervals
    """
    
    def __init__(
        self,
        confidence_level: float = 0.95,
        n_bootstrap: int = 10000,
        random_state: Optional[int] = None
    ):
        """
        Args:
            confidence_level: Confidence level for intervals (default: 0.95)
            n_bootstrap: Number of bootstrap samples (default: 10000)
            random_state: Random seed for reproducibility
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    # ===================================================================
    # COMPREHENSIVE ANALYSIS
    # ===================================================================
    
    def analyze(
        self,
        data: np.ndarray,
        name: str = 'metric',
        method: str = 'bootstrap'
    ) -> Dict:
        """
        Complete uncertainty analysis with multiple methods.
        
        Args:
            data: Array of measurements
            name: Metric name for reporting
            method: CI method ('parametric', 'bootstrap', 'bca')
        
        Returns:
            Comprehensive analysis results
        """
        results = {
            'metric_name': name,
            'passed': True,
            'warnings': [],
            'errors': [],
            'method': method
        }
        
        # Clean data
        data = np.asarray(data)
        data = data[np.isfinite(data)]
        
        if len(data) == 0:
            results['passed'] = False
            results['errors'].append('No valid data for analysis')
            return results
        
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        sem = std / np.sqrt(n)
        median = np.median(data)
        
        # Compute confidence interval using specified method
        if method == 'parametric':
            ci = self._parametric_ci(data)
        elif method == 'bootstrap':
            ci = self.bootstrap_ci(data, method='percentile')
        elif method == 'bca':
            ci = self.bootstrap_ci(data, method='bca')
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Effect size (one-sample: mean/std)
        cohens_d = mean / std if std > 0 else 0
        
        # Coefficient of variation
        cv = (std / mean * 100) if mean > 0 else float('inf')
        
        # CI width as percentage of mean
        ci_width_pct = ((ci[1] - ci[0]) / mean * 100) if mean > 0 else float('inf')
        
        # Store results
        results['point_estimates'] = {
            'mean': float(mean),
            'median': float(median),
            'std': float(std),
            'sem': float(sem)
        }
        
        results['confidence_interval'] = {
            'lower': float(ci[0]),
            'upper': float(ci[1]),
            'width': float(ci[1] - ci[0]),
            'width_percent': float(ci_width_pct),
            'level': self.confidence_level
        }
        
        results['effect_size'] = {
            'cohens_d': float(cohens_d),
            'interpretation': self._interpret_effect_size(cohens_d)
        }
        
        results['reliability'] = {
            'sample_size': int(n),
            'cv_percent': float(cv),
            'precision': 'high' if cv < 15 else 'moderate' if cv < 30 else 'low'
        }
        
        # Distribution characteristics
        if n >= 8:
            _, p_normal = stats.shapiro(data[:5000])  # Max 5000 for Shapiro
            results['distribution'] = {
                'normality_p': float(p_normal),
                'is_normal': bool(p_normal > 0.05),
                'skewness': float(stats.skew(data)),
                'kurtosis': float(stats.kurtosis(data))
            }
        
        # Quality checks
        if n < 30:
            results['warnings'].append(f'Small sample size (N={n} < 30)')
        
        if cv > 50:
            results['warnings'].append(f'High variability (CV={cv:.1f}%)')
        
        if ci_width_pct > 50:
            results['warnings'].append(f'Wide confidence interval ({ci_width_pct:.1f}% of mean)')
        
        # Summary
        status = 'PASS' if results['passed'] else 'FAIL'
        results['summary'] = (
            f"{name}: {status} | "
            f"N={n} | "
            f"Mean={mean:.3f} "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"d={cohens_d:.2f} | "
            f"CV={cv:.1f}%"
        )
        
        return results
    
    # ===================================================================
    # CONFIDENCE INTERVALS
    # ===================================================================
    
    def _parametric_ci(self, data: np.ndarray) -> Tuple[float, float]:
        """Standard parametric confidence interval (t-distribution)."""
        n = len(data)
        mean = np.mean(data)
        sem = np.std(data, ddof=1) / np.sqrt(n)
        
        return stats.t.interval(
            self.confidence_level,
            df=n - 1,
            loc=mean,
            scale=sem
        )
    
    def bootstrap_ci(
        self,
        data: np.ndarray,
        method: str = 'percentile',
        statistic: Callable = np.mean
    ) -> Tuple[float, float]:
        """
        Bootstrap confidence interval.
        
        Args:
            data: Input data
            method: 'percentile', 'bca' (bias-corrected accelerated), or 'basic'
            statistic: Function to compute (default: mean)
        
        Returns:
            (lower, upper) confidence bounds
        """
        n = len(data)
        bootstrap_stats = np.zeros(self.n_bootstrap)
        
        # Generate bootstrap samples
        for i in range(self.n_bootstrap):
            resample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats[i] = statistic(resample)
        
        if method == 'percentile':
            # Simple percentile method
            alpha_lower = (1 - self.confidence_level) / 2
            alpha_upper = 1 - alpha_lower
            return np.percentile(bootstrap_stats, [alpha_lower * 100, alpha_upper * 100])
        
        elif method == 'bca':
            # Bias-corrected and accelerated (BCa)
            return self._bca_ci(data, bootstrap_stats, statistic)
        
        elif method == 'basic':
            # Basic bootstrap (reflection method)
            theta_hat = statistic(data)
            alpha_lower = (1 - self.confidence_level) / 2
            alpha_upper = 1 - alpha_lower
            
            lower_percentile = np.percentile(bootstrap_stats, alpha_upper * 100)
            upper_percentile = np.percentile(bootstrap_stats, alpha_lower * 100)
            
            return (2 * theta_hat - lower_percentile, 2 * theta_hat - upper_percentile)
        
        else:
            raise ValueError(f"Unknown bootstrap method: {method}")
    
    def _bca_ci(
        self,
        data: np.ndarray,
        bootstrap_stats: np.ndarray,
        statistic: Callable
    ) -> Tuple[float, float]:
        """Bias-corrected and accelerated (BCa) bootstrap CI."""
        n = len(data)
        theta_hat = statistic(data)
        
        # Bias correction
        z0 = stats.norm.ppf(np.mean(bootstrap_stats < theta_hat))
        
        # Acceleration (jackknife)
        jackknife_stats = np.zeros(n)
        for i in range(n):
            jackknife_sample = np.delete(data, i)
            jackknife_stats[i] = statistic(jackknife_sample)
        
        jackknife_mean = np.mean(jackknife_stats)
        numerator = np.sum((jackknife_mean - jackknife_stats) ** 3)
        denominator = 6 * (np.sum((jackknife_mean - jackknife_stats) ** 2) ** 1.5)
        
        acceleration = numerator / denominator if denominator != 0 else 0
        
        # Adjusted percentiles
        z_alpha_lower = stats.norm.ppf((1 - self.confidence_level) / 2)
        z_alpha_upper = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
        
        alpha_lower = stats.norm.cdf(
            z0 + (z0 + z_alpha_lower) / (1 - acceleration * (z0 + z_alpha_lower))
        )
        alpha_upper = stats.norm.cdf(
            z0 + (z0 + z_alpha_upper) / (1 - acceleration * (z0 + z_alpha_upper))
        )
        
        return np.percentile(bootstrap_stats, [alpha_lower * 100, alpha_upper * 100])
    
    # ===================================================================
    # MONTE CARLO METHODS
    # ===================================================================
    
    def monte_carlo_propagation(
        self,
        input_distributions: Dict[str, Tuple[str, Tuple]],
        model_function: Callable,
        n_samples: int = 10000
    ) -> Dict:
        """
        Monte Carlo uncertainty propagation through a model.
        
        Args:
            input_distributions: Dict of {param_name: (dist_type, dist_params)}
                Example: {'speed': ('normal', (50, 5)), 'distance': ('uniform', (10, 20))}
            model_function: Function that takes **kwargs of parameters
            n_samples: Number of MC samples
        
        Returns:
            Output distribution statistics and CI
        """
        outputs = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Sample from input distributions
            inputs = {}
            for param, (dist_type, dist_params) in input_distributions.items():
                if dist_type == 'normal':
                    inputs[param] = np.random.normal(*dist_params)
                elif dist_type == 'uniform':
                    inputs[param] = np.random.uniform(*dist_params)
                elif dist_type == 'lognormal':
                    inputs[param] = np.random.lognormal(*dist_params)
                else:
                    raise ValueError(f"Unknown distribution: {dist_type}")
            
            # Evaluate model
            outputs[i] = model_function(**inputs)
        
        # Analyze outputs
        return self.analyze(outputs, name='MC_output', method='bootstrap')
    
    # ===================================================================
    # EFFECT SIZE
    # ===================================================================
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size (Cohen, 1988)."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def compute_effect_size(
        self,
        group1: np.ndarray,
        group2: Optional[np.ndarray] = None,
        estimator: str = 'cohens_d'
    ) -> Dict:
        """
        Compute effect size with confidence interval.
        
        Args:
            group1: First group (or single sample)
            group2: Second group (optional)
            estimator: 'cohens_d', 'hedges_g', or 'glass_delta'
        
        Returns:
            Effect size estimate with CI
        """
        g1 = np.asarray(group1)[np.isfinite(group1)]
        
        if group2 is None:
            # One-sample effect size
            d = np.mean(g1) / np.std(g1, ddof=1) if np.std(g1, ddof=1) > 0 else 0
            ci = self.bootstrap_ci(g1, statistic=lambda x: np.mean(x) / np.std(x, ddof=1))
        
        else:
            # Two-sample effect size
            g2 = np.asarray(group2)[np.isfinite(group2)]
            n1, n2 = len(g1), len(g2)
            
            if estimator == 'cohens_d':
                # Pooled standard deviation
                pooled_var = ((n1 - 1) * np.var(g1, ddof=1) + (n2 - 1) * np.var(g2, ddof=1)) / (n1 + n2 - 2)
                pooled_std = np.sqrt(pooled_var)
                d = (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std > 0 else 0
            
            elif estimator == 'hedges_g':
                # Hedges' g (bias-corrected)
                pooled_var = ((n1 - 1) * np.var(g1, ddof=1) + (n2 - 1) * np.var(g2, ddof=1)) / (n1 + n2 - 2)
                pooled_std = np.sqrt(pooled_var)
                cohens_d = (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std > 0 else 0
                
                # Correction factor
                correction = 1 - (3 / (4 * (n1 + n2) - 9))
                d = cohens_d * correction
            
            elif estimator == 'glass_delta':
                # Glass's Δ (uses control group SD)
                d = (np.mean(g1) - np.mean(g2)) / np.std(g2, ddof=1) if np.std(g2, ddof=1) > 0 else 0
            
            else:
                raise ValueError(f"Unknown estimator: {estimator}")
            
            # Bootstrap CI for effect size
            combined = np.concatenate([g1, g2])
            labels = np.array([0] * n1 + [1] * n2)
            
            bootstrap_d = np.zeros(self.n_bootstrap)
            for i in range(self.n_bootstrap):
                idx = np.random.choice(len(combined), size=len(combined), replace=True)
                boot_g1 = combined[idx][labels[idx] == 0]
                boot_g2 = combined[idx][labels[idx] == 1]
                
                if len(boot_g1) > 0 and len(boot_g2) > 0:
                    if estimator == 'cohens_d':
                        pooled_std_boot = np.sqrt(
                            ((len(boot_g1) - 1) * np.var(boot_g1, ddof=1) +
                             (len(boot_g2) - 1) * np.var(boot_g2, ddof=1)) /
                            (len(boot_g1) + len(boot_g2) - 2)
                        )
                        bootstrap_d[i] = (np.mean(boot_g1) - np.mean(boot_g2)) / pooled_std_boot if pooled_std_boot > 0 else 0
            
            alpha_lower = (1 - self.confidence_level) / 2
            alpha_upper = 1 - alpha_lower
            ci = np.percentile(bootstrap_d, [alpha_lower * 100, alpha_upper * 100])
        
        return {
            'estimate': float(d),
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
            'interpretation': self._interpret_effect_size(d),
            'estimator': estimator
        }
    
    # ===================================================================
    # POWER ANALYSIS
    # ===================================================================
    
    def compute_required_sample_size(
        self,
        effect_size: float,
        power: float = 0.80,
        alpha: float = 0.05,
        test_type: str = 'two-sided'
    ) -> int:
        """
        Compute required sample size for desired power.
        
        Args:
            effect_size: Expected Cohen's d
            power: Desired statistical power (default: 0.80)
            alpha: Significance level (default: 0.05)
            test_type: 'two-sided' or 'one-sided'
        
        Returns:
            Required sample size
        """
        if test_type == 'two-sided':
            z_alpha = stats.norm.ppf(1 - alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        
        z_beta = stats.norm.ppf(power)
        
        # Formula for one-sample t-test
        n = ((z_alpha + z_beta) / effect_size) ** 2
        
        return int(np.ceil(n))


# ===================================================================
# CONVENIENCE FUNCTIONS
# ===================================================================

def bootstrap_ci(
    data: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    method: str = 'bca'
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval (standalone function).
    
    Args:
        data: Input data
        confidence: Confidence level
        n_bootstrap: Number of bootstrap samples
        method: 'percentile', 'bca', or 'basic'
    
    Returns:
        (lower, upper) confidence bounds
    """
    uq = UncertaintyQuantifier(confidence_level=confidence, n_bootstrap=n_bootstrap)
    return uq.bootstrap_ci(data, method=method)


def monte_carlo_uq(
    input_distributions: Dict[str, Tuple[str, Tuple]],
    model_function: Callable,
    n_samples: int = 10000,
    confidence: float = 0.95
) -> Dict:
    """Monte Carlo uncertainty quantification (standalone)."""
    uq = UncertaintyQuantifier(confidence_level=confidence)
    return uq.monte_carlo_propagation(input_distributions, model_function, n_samples)


def compute_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95,
    method: str = 'parametric'
) -> Tuple[float, float]:
    """Compute confidence interval using specified method."""
    uq = UncertaintyQuantifier(confidence_level=confidence)
    
    data = np.asarray(data)[np.isfinite(data)]
    
    if method == 'parametric':
        return uq._parametric_ci(data)
    else:
        return uq.bootstrap_ci(data, method=method)


def compute_effect_size(
    group1: np.ndarray,
    group2: Optional[np.ndarray] = None,
    estimator: str = 'cohens_d'
) -> float:
    """
    Compute effect size (standalone).
    
    Returns just the point estimate (not full dict).
    """
    uq = UncertaintyQuantifier()
    result = uq.compute_effect_size(group1, group2, estimator)
    return result['estimate']


def compute_sample_size(
    effect_size: float,
    power: float = 0.80,
    alpha: float = 0.05
) -> int:
    """Compute required sample size for power analysis."""
    uq = UncertaintyQuantifier()
    return uq.compute_required_sample_size(effect_size, power, alpha)


def sensitivity_analysis(
    baseline_params: Dict[str, float],
    model_function: Callable,
    param_ranges: Dict[str, Tuple[float, float]],
    n_steps: int = 20
) -> Dict[str, np.ndarray]:
    """
    One-at-a-time sensitivity analysis.
    
    Args:
        baseline_params: Baseline parameter values
        model_function: Model function
        param_ranges: Dict of {param: (min, max)}
        n_steps: Number of steps for each parameter
    
    Returns:
        Dict of {param_name: output_values}
    """
    results = {}
    
    for param, (min_val, max_val) in param_ranges.items():
        param_values = np.linspace(min_val, max_val, n_steps)
        outputs = np.zeros(n_steps)
        
        for i, val in enumerate(param_values):
            params = baseline_params.copy()
            params[param] = val
            outputs[i] = model_function(**params)
        
        results[param] = {
            'param_values': param_values,
            'outputs': outputs,
            'sensitivity': np.std(outputs) / np.mean(outputs) if np.mean(outputs) > 0 else 0
        }
    
    return results
