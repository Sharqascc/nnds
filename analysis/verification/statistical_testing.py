import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy import stats
import warnings

__all__ = ['StatisticalTester', 'test_pet_difference', 'test_ttc_difference', 'chi_square_test']


class StatisticalTester:
    """Statistical hypothesis testing for traffic safety analysis.
    Provides peer-review-ready statistical tests for SSM comparisons.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def _clean_data(self, *arrays) -> List[np.ndarray]:
        return [np.asarray(a)[np.isfinite(a)] for a in arrays]

    def t_test(self, group1: np.ndarray, group2: np.ndarray = None,
               alternative: str = 'two-sided') -> Dict:
        g1, = self._clean_data(group1)
        result = {'test': 't-test', 'alpha': self.alpha, 'passed': True, 'warnings': [], 'errors': []}
        if len(g1) < 2:
            result['passed'] = False
            result['errors'].append('Insufficient data in group 1')
            return result
        if group2 is None:
            t_stat, p_value = stats.ttest_1samp(g1, 0)
            result['type'] = 'one-sample'
        else:
            g2, = self._clean_data(group2)
            if len(g2) < 2:
                result['passed'] = False
                result['errors'].append('Insufficient data in group 2')
                return result
            t_stat, p_value = stats.ttest_ind(g1, g2, equal_var=False)
            result['type'] = 'two-sample (Welch)'
        pooled_std = np.sqrt((np.var(g1)*len(g1) + np.var(g2)*len(g2)) / (len(g1)+len(g2)-2)) if group2 is not None else np.std(g1)
        cohens_d = float((np.mean(g1) - np.mean(g2)) / pooled_std) if pooled_std > 0 and group2 is not None else float(np.mean(g1) / np.std(g1))
        ci = stats.t.interval(1-self.alpha, df=len(g1)-1, loc=np.mean(g1), scale=np.std(g1)/np.sqrt(len(g1)))
        normal_check = stats.shapiro(g1[:min(5000, len(g1))])[1] > 0.05
        result['statistics'] = {'t_statistic': float(t_stat), 'p_value': float(p_value),
                                'significant': p_value < self.alpha, 'effect_size_cohens_d': cohens_d,
                                'group1': {'n': len(g1), 'mean': float(np.mean(g1)), 'std': float(np.std(g1))},
                                'confidence_interval': ci, 'assumptions_normal': normal_check}
        if group2 is not None:
            result['statistics']['group2'] = {'n': len(g2), 'mean': float(np.mean(g2)), 'std': float(np.std(g2))}
        if not normal_check:
            result['warnings'].append('Data may not be normally distributed; consider Mann-Whitney U')
        sig = 'significant' if p_value < self.alpha else 'not significant'
        result['summary'] = f"t-test ({result['type']}): t={t_stat:.3f}, p={p_value:.4f} ({sig}), d={cohens_d:.2f}"
        return result

    def mann_whitney(self, group1: np.ndarray, group2: np.ndarray) -> Dict:
        g1, g2 = self._clean_data(group1, group2)
        if len(g1) < 2 or len(g2) < 2:
            return {'test': 'Mann-Whitney U', 'error': 'Insufficient data', 'passed': False}
        stat, p_value = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        return {'test': 'Mann-Whitney U', 'statistic': float(stat), 'p_value': float(p_value),
                'significant': p_value < self.alpha, 'summary': f"U={stat:.1f}, p={p_value:.4f}"}

    def chi_square_test(self, observed: np.ndarray, expected: np.ndarray = None) -> Dict:
        obs = np.asarray(observed)[np.isfinite(observed)]
        if expected is None:
            expected = np.full_like(obs, np.mean(obs))
        else:
            expected = np.asarray(expected)
        chi2, p_value = stats.chisquare(obs, expected)
        return {'test': 'Chi-square', 'chi2_statistic': float(chi2), 'p_value': float(p_value),
                'significant': p_value < self.alpha, 'summary': f"chi2={chi2:.2f}, p={p_value:.4f}"}

    def anova(self, *groups) -> Dict:
        if len(groups) < 2:
            return {'test': 'ANOVA', 'error': 'Need at least 2 groups', 'passed': False}
        clean_groups = self._clean_data(*groups)
        if any(len(g) < 2 for g in clean_groups):
            return {'test': 'ANOVA', 'error': 'All groups need at least 2 observations', 'passed': False}
        f_stat, p_value = stats.f_oneway(*clean_groups)
        return {'test': 'One-way ANOVA', 'f_statistic': float(f_stat), 'p_value': float(p_value),
                'significant': p_value < self.alpha, 'n_groups': len(groups),
                'summary': f"F={f_stat:.2f}, p={p_value:.4f} ({'significant' if p_value < self.alpha else 'not significant'})"}


def test_pet_difference(pet_group1: np.ndarray, pet_group2: np.ndarray, alpha: float = 0.05) -> Dict:
    return StatisticalTester(alpha=alpha).t_test(pet_group1, pet_group2)


def test_ttc_difference(ttc_group1: np.ndarray, ttc_group2: np.ndarray, alpha: float = 0.05) -> Dict:
    return StatisticalTester(alpha=alpha).t_test(ttc_group1, ttc_group2)


def chi_square_test(observed: np.ndarray, expected: np.ndarray = None, alpha: float = 0.05) -> Dict:
    return StatisticalTester(alpha=alpha).chi_square_test(observed, expected)
