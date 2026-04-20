import numpy as np
from typing import Dict, Tuple, Optional
from scipy import stats
import warnings

__all__ = ['UncertaintyQuantifier', 'compute_confidence_interval', 'compute_effect_size']


class UncertaintyQuantifier:
    """Quantify uncertainty in SSM measurements for research reporting.
    Provides confidence intervals, effect sizes, and power analysis.
    """

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def analyze(self, data: np.ndarray, name: str = 'metric') -> Dict:
        results = {'metric_name': name, 'passed': True, 'warnings': [], 'errors': []}
        data = np.asarray(data)[np.isfinite(data)]
        if len(data) == 0:
            results['passed'] = False
            results['errors'].append('No valid data for analysis')
            return results
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        sem = std / np.sqrt(n)
        ci = stats.t.interval(self.confidence_level, df=n-1, loc=mean, scale=sem)
        cohens_d = mean / std if std > 0 else 0
        ci_width_pct = float((ci[1]-ci[0])/mean*100) if mean > 0 else float('inf')
        results['confidence_interval'] = ci
        results['effect_size'] = {'cohens_d': float(cohens_d),
                                  'interpretation': 'negligible' if abs(cohens_d) < 0.2
                                  else 'small' if abs(cohens_d) < 0.5
                                  else 'medium' if abs(cohens_d) < 0.8 else 'large'}
        results['reliability'] = {'sample_size': n, 'mean': float(mean), 'std': float(std),
                                  'sem': float(sem), 'ci': ci, 'cv': float(std/mean*100) if mean > 0 else float('inf')}
        if n < 30:
            results['warnings'].append(f'Sample size ({n}) is small')
        if results['reliability']['cv'] > 50:
            results['warnings'].append(f'High variability (CV={results["reliability"]["cv"]:.1f}%)')
        status = 'PASS' if results['passed'] else 'FAIL'
        results['summary'] = f"{name}: {status} | N={n} | Mean={mean:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | d={cohens_d:.2f}"
        return results


def compute_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    data = np.asarray(data)[np.isfinite(data)]
    n, mean = len(data), np.mean(data)
    sem = np.std(data, ddof=1) / np.sqrt(n)
    return stats.t.interval(confidence, df=n-1, loc=mean, scale=sem)


def compute_effect_size(group1: np.ndarray, group2: np.ndarray = None) -> float:
    g1 = np.asarray(group1)[np.isfinite(group1)]
    if group2 is not None:
        g2 = np.asarray(group2)[np.isfinite(group2)]
        pooled = np.sqrt((np.var(g1)*len(g1) + np.var(g2)*len(g2)) / (len(g1)+len(g2)-2))
        return float((np.mean(g1) - np.mean(g2)) / pooled) if pooled > 0 else 0.0
    return float(np.mean(g1) / np.std(g1)) if np.std(g1) > 0 else 0.0
