"""Uncertainty Quantification for SSM Metrics.

Provides Monte Carlo simulation and bootstrap confidence intervals
for surrogate safety metric uncertainty estimation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable


@dataclass
class UQResult:
    """Container for uncertainty quantification results."""
    metric_name: str
    estimate: float
    std_error: float
    ci_lower: float
    ci_upper: float
    n_samples: int
    method: str


class UncertaintyQuantifier:
    """Uncertainty quantifier for safety metrics.
    
    Supports Monte Carlo simulation and bootstrap resampling
    for estimating confidence intervals of SSM values.
    """
    
    def __init__(self, random_state: Optional[int] = None):
        self.rng = np.random.default_rng(random_state)
    
    def monte_carlo_uq(
        self,
        data: np.ndarray,
        metric_fn: Callable[[np.ndarray], float],
        n_simulations: int = 1000,
        noise_std: float = 0.1,
    ) -> UQResult:
        """Estimate uncertainty via Monte Carlo simulation.
        
        Args:
            data: Input data array
            metric_fn: Function computing the metric
            n_simulations: Number of Monte Carlo iterations
            noise_std: Standard deviation of noise to add
            
        Returns:
            UQResult with estimate and confidence interval
        """
        data = np.asarray(data)
        estimates = []
        
        for _ in range(n_simulations):
            noisy_data = data + self.rng.normal(0, noise_std, size=data.shape)
            try:
                estimates.append(metric_fn(noisy_data))
            except (ValueError, RuntimeError):
                continue
        
        estimates = np.array(estimates)
        if len(estimates) == 0:
            return UQResult(
                metric_name="MC_UQ",
                estimate=np.nan,
                std_error=np.nan,
                ci_lower=np.nan,
                ci_upper=np.nan,
                n_samples=0,
                method="monte_carlo",
            )
        
        mean = float(np.mean(estimates))
        std = float(np.std(estimates))
        ci_lower = float(np.percentile(estimates, 2.5))
        ci_upper = float(np.percentile(estimates, 97.5))
        
        return UQResult(
            metric_name="MC_UQ",
            estimate=mean,
            std_error=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_samples=len(estimates),
            method="monte_carlo",
        )
    
    def bootstrap_ci(
        self,
        data: np.ndarray,
        metric_fn: Callable[[np.ndarray], float],
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
    ) -> UQResult:
        """Estimate confidence interval via bootstrap resampling.
        
        Args:
            data: Input data array
            metric_fn: Function computing the metric
            n_bootstrap: Number of bootstrap samples
            ci_level: Confidence level (e.g., 0.95)
            
        Returns:
            UQResult with bootstrap confidence interval
        """
        data = np.asarray(data)
        n = len(data)
        estimates = []
        
        for _ in range(n_bootstrap):
            indices = self.rng.integers(0, n, size=n)
            resampled = data[indices]
            try:
                estimates.append(metric_fn(resampled))
            except (ValueError, RuntimeError):
                continue
        
        estimates = np.array(estimates)
        if len(estimates) == 0:
            return UQResult(
                metric_name="bootstrap",
                estimate=np.nan,
                std_error=np.nan,
                ci_lower=np.nan,
                ci_upper=np.nan,
                n_samples=0,
                method="bootstrap",
            )
        
        alpha = 1 - ci_level
        mean = float(np.mean(estimates))
        std = float(np.std(estimates))
        ci_lower = float(np.percentile(estimates, alpha / 2 * 100))
        ci_upper = float(np.percentile(estimates, (1 - alpha / 2) * 100))
        
        return UQResult(
            metric_name="bootstrap",
            estimate=mean,
            std_error=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_samples=len(estimates),
            method="bootstrap",
        )
    
    def quantify_pet_uncertainty(
        self,
        pet_values: np.ndarray,
        method: str = "bootstrap",
    ) -> UQResult:
        """Quantify uncertainty in PET statistics.
        
        Args:
            pet_values: PET values
            method: 'bootstrap' or 'monte_carlo'
            
        Returns:
            UQResult for PET mean uncertainty
        """
        pet_values = np.asarray(pet_values)
        
        if method == "bootstrap":
            return self.bootstrap_ci(
                pet_values,
                metric_fn=np.nanmean,
            )
        else:
            return self.monte_carlo_uq(
                pet_values,
                metric_fn=np.nanmean,
            )
