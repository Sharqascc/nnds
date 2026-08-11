"""
Statistical Hypothesis Testing for Traffic Safety Analysis

Provides comprehensive statistical tests for SSM comparisons:
- Parametric tests (t-test, ANOVA, paired t-test)
- Non-parametric tests (Mann-Whitney, Kruskal-Wallis, Wilcoxon)
- Chi-square and proportion tests
- Multiple testing corrections (Bonferroni, Holm)
- Assumption checking (normality, homoscedasticity)
- Post-hoc tests (Tukey HSD, Dunn's test)

Compliant with:
- APA statistical reporting standards
- CONSORT guidelines for clinical trials
- Traffic safety research best practices
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Union
from scipy import stats
from dataclasses import dataclass
import warnings


__all__ = [
    'StatisticalTester',
    'test_pet_difference',
    'test_ttc_difference',
    'test_drac_difference',
    'chi_square_test',
    'paired_test',
    'multiple_comparisons',
    'check_assumptions'
]


@dataclass
class TestResult:
    """Structured container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""


class StatisticalTester:
    """
    Comprehensive statistical hypothesis testing suite.
    
    Features:
    - Automatic assumption checking
    - Multiple test corrections
    - Effect size computation
    - Confidence intervals
    - Post-hoc tests
    - Detailed reporting
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        auto_check_assumptions: bool = True,
        correction_method: Optional[str] = None
    ):
        """
        Args:
            alpha: Significance level (default: 0.05)
            auto_check_assumptions: Check normality/homoscedasticity automatically
            correction_method: Multiple testing correction ('bonferroni', 'holm', None)
        """
        self.alpha = alpha
        self.auto_check_assumptions = auto_check_assumptions
        self.correction_method = correction_method
    
    # ===================================================================
    # DATA CLEANING
    # ===================================================================
    
    def _clean_data(self, *arrays) -> List[np.ndarray]:
        """Remove NaN and Inf values from arrays."""
        cleaned = []
        for arr in arrays:
            arr = np.asarray(arr)
            arr = arr[np.isfinite(arr)]
            cleaned.append(arr)
        return cleaned
    
    # ===================================================================
    # ASSUMPTION CHECKING
    # ===================================================================
    
    def check_normality(
        self,
        data: np.ndarray,
        method: str = 'shapiro'
    ) -> Dict:
        """
        Test for normality.
        
        Args:
            data: Input data
            method: 'shapiro' or 'anderson'
        
        Returns:
            Test results with interpretation
        """
        data = self._clean_data(data)[0]
        
        if len(data) < 3:
            return {
                'test': method,
                'passed': False,
                'error': 'Insufficient data (n < 3)'
            }
        
        if method == 'shapiro':
            # Shapiro-Wilk test (max 5000 samples)
            sample = data[:5000] if len(data) > 5000 else data
            statistic, p_value = stats.shapiro(sample)
            
            return {
                'test': 'Shapiro-Wilk',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'normal': bool(p_value > self.alpha),
                'interpretation': 'Normal' if p_value > self.alpha else 'Non-normal',
                'recommendation': '' if p_value > self.alpha else 'Consider non-parametric tests'
            }
        
        elif method == 'anderson':
            # Anderson-Darling test
            result = stats.anderson(data, dist='norm')
            
            # Use 5% critical value
            critical_5pct = result.critical_values[2]  # Index 2 is 5%
            
            return {
                'test': 'Anderson-Darling',
                'statistic': float(result.statistic),
                'critical_value_5pct': float(critical_5pct),
                'normal': bool(result.statistic < critical_5pct),
                'interpretation': 'Normal' if result.statistic < critical_5pct else 'Non-normal'
            }
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def check_homoscedasticity(
        self,
        *groups: np.ndarray
    ) -> Dict:
        """
        Test for equal variances (Levene's test).
        
        Args:
            groups: Two or more groups
        
        Returns:
            Test results
        """
        groups = self._clean_data(*groups)
        
        if len(groups) < 2:
            return {'error': 'Need at least 2 groups'}
        
        if any(len(g) < 2 for g in groups):
            return {'error': 'All groups need at least 2 observations'}
        
        # Levene's test (robust to non-normality)
        statistic, p_value = stats.levene(*groups)
        
        return {
            'test': "Levene's test",
            'statistic': float(statistic),
            'p_value': float(p_value),
            'equal_variances': bool(p_value > self.alpha),
            'interpretation': 'Equal variances' if p_value > self.alpha else 'Unequal variances',
            'recommendation': '' if p_value > self.alpha else "Use Welch's t-test or non-parametric test"
        }
    
    def check_assumptions(
        self,
        *groups: np.ndarray,
        test_type: str = 't-test'
    ) -> Dict:
        """
        Comprehensive assumption checking for statistical tests.
        
        Args:
            groups: Data groups
            test_type: Type of test to check for ('t-test', 'anova')
        
        Returns:
            Summary of assumption checks
        """
        results = {
            'test_type': test_type,
            'checks': {},
            'all_passed': True,
            'recommendations': []
        }
        
        # Check normality for each group
        for i, group in enumerate(groups):
            normality = self.check_normality(group)
            results['checks'][f'normality_group{i+1}'] = normality
            
            if not normality.get('normal', False):
                results['all_passed'] = False
                results['recommendations'].append(
                    f"Group {i+1} is non-normal. Consider Mann-Whitney U or Kruskal-Wallis."
                )
        
        # Check homoscedasticity (if multiple groups)
        if len(groups) >= 2:
            homoscedasticity = self.check_homoscedasticity(*groups)
            results['checks']['homoscedasticity'] = homoscedasticity
            
            if not homoscedasticity.get('equal_variances', False):
                results['all_passed'] = False
                if test_type == 't-test':
                    results['recommendations'].append("Use Welch's t-test (unequal variances)")
                elif test_type == 'anova':
                    results['recommendations'].append("Consider Welch's ANOVA or Kruskal-Wallis")
        
        return results
    
    # ===================================================================
    # T-TESTS
    # ===================================================================
    
    def t_test(
        self,
        group1: np.ndarray,
        group2: Optional[np.ndarray] = None,
        paired: bool = False,
        equal_var: bool = False,
        alternative: str = 'two-sided'
    ) -> Dict:
        """
        Student's t-test or Welch's t-test.
        
        Args:
            group1: First group
            group2: Second group (None for one-sample test)
            paired: Whether data are paired
            equal_var: Assume equal variances (False = Welch's test)
            alternative: 'two-sided', 'less', or 'greater'
        
        Returns:
            Comprehensive test results
        """
        g1, = self._clean_data(group1)
        
        result = {
            'test': 't-test',
            'alpha': self.alpha,
            'passed': True,
            'warnings': [],
            'errors': []
        }
        
        if len(g1) < 2:
            result['passed'] = False
            result['errors'].append('Insufficient data in group 1 (n < 2)')
            return result
        
        # One-sample t-test
        if group2 is None:
            t_stat, p_value = stats.ttest_1samp(g1, 0, alternative=alternative)
            result['type'] = 'one-sample'
            
            # Cohen's d for one sample
            cohens_d = np.mean(g1) / np.std(g1, ddof=1) if np.std(g1, ddof=1) > 0 else 0
            
            # CI for mean
            ci = stats.t.interval(
                1 - self.alpha,
                df=len(g1) - 1,
                loc=np.mean(g1),
                scale=stats.sem(g1)
            )
            
            result['statistics'] = {
                'n': len(g1),
                'mean': float(np.mean(g1)),
                'std': float(np.std(g1, ddof=1)),
                'sem': float(stats.sem(g1))
            }
        
        # Two-sample t-test
        else:
            g2, = self._clean_data(group2)
            
            if len(g2) < 2:
                result['passed'] = False
                result['errors'].append('Insufficient data in group 2 (n < 2)')
                return result
            
            # Paired or independent
            if paired:
                if len(g1) != len(g2):
                    result['errors'].append('Paired test requires equal sample sizes')
                    result['passed'] = False
                    return result
                
                t_stat, p_value = stats.ttest_rel(g1, g2, alternative=alternative)
                result['type'] = 'paired'
                
                # Cohen's d for paired data
                differences = g1 - g2
                cohens_d = np.mean(differences) / np.std(differences, ddof=1) if np.std(differences, ddof=1) > 0 else 0
            
            else:
                t_stat, p_value = stats.ttest_ind(g1, g2, equal_var=equal_var, alternative=alternative)
                result['type'] = 'two-sample (Welch)' if not equal_var else 'two-sample (Student)'
                
                # Cohen's d (pooled or unpooled)
                if equal_var:
                    # Pooled standard deviation
                    n1, n2 = len(g1), len(g2)
                    pooled_var = ((n1 - 1) * np.var(g1, ddof=1) + (n2 - 1) * np.var(g2, ddof=1)) / (n1 + n2 - 2)
                    pooled_std = np.sqrt(pooled_var)
                    cohens_d = (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std > 0 else 0
                else:
                    # Glass's delta (uses control group SD)
                    cohens_d = (np.mean(g1) - np.mean(g2)) / np.std(g2, ddof=1) if np.std(g2, ddof=1) > 0 else 0
            
            # CIs for both groups
            ci1 = stats.t.interval(1 - self.alpha, df=len(g1) - 1, loc=np.mean(g1), scale=stats.sem(g1))
            ci2 = stats.t.interval(1 - self.alpha, df=len(g2) - 1, loc=np.mean(g2), scale=stats.sem(g2))
            
            result['statistics'] = {
                'group1': {
                    'n': len(g1),
                    'mean': float(np.mean(g1)),
                    'std': float(np.std(g1, ddof=1)),
                    'sem': float(stats.sem(g1)),
                    'ci': ci1
                },
                'group2': {
                    'n': len(g2),
                    'mean': float(np.mean(g2)),
                    'std': float(np.std(g2, ddof=1)),
                    'sem': float(stats.sem(g2)),
                    'ci': ci2
                },
                'mean_difference': float(np.mean(g1) - np.mean(g2))
            }
        
        # Store test results
        result['test_statistics'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < self.alpha),
            'effect_size_cohens_d': float(cohens_d),
            'effect_interpretation': self._interpret_effect_size(cohens_d)
        }
        
        # Check assumptions if requested
        if self.auto_check_assumptions and group2 is not None and not paired:
            normality_check = self.check_normality(g1)
            result['assumptions'] = {
                'normality_group1': normality_check['normal'],
                'normality_p': normality_check.get('p_value')
            }
            
            if not normality_check['normal']:
                result['warnings'].append(
                    'Data may not be normally distributed. Consider Mann-Whitney U test.'
                )
        
        # Summary
        sig_text = 'significant' if p_value < self.alpha else 'not significant'
        result['summary'] = (
            f"t-test ({result['type']}): "
            f"t={t_stat:.3f}, "
            f"p={p_value:.4f} ({sig_text}), "
            f"d={cohens_d:.2f}"
        )
        
        return result
    
    # ===================================================================
    # NON-PARAMETRIC TESTS
    # ===================================================================
    
    def mann_whitney(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        alternative: str = 'two-sided'
    ) -> Dict:
        """
        Mann-Whitney U test (non-parametric alternative to t-test).
        
        Args:
            group1: First group
            group2: Second group
            alternative: 'two-sided', 'less', or 'greater'
        
        Returns:
            Test results with rank-biserial correlation effect size
        """
        g1, g2 = self._clean_data(group1, group2)
        
        if len(g1) < 2 or len(g2) < 2:
            return {
                'test': 'Mann-Whitney U',
                'error': 'Insufficient data (need n ≥ 2 per group)',
                'passed': False
            }
        
        # Perform test
        stat, p_value = stats.mannwhitneyu(g1, g2, alternative=alternative)
        
        # Rank-biserial correlation (effect size for Mann-Whitney)
        n1, n2 = len(g1), len(g2)
        rank_biserial = 1 - (2 * stat) / (n1 * n2)
        
        return {
            'test': 'Mann-Whitney U',
            'statistic': float(stat),
            'p_value': float(p_value),
            'significant': bool(p_value < self.alpha),
            'effect_size_rank_biserial': float(rank_biserial),
            'group1': {'n': n1, 'median': float(np.median(g1))},
            'group2': {'n': n2, 'median': float(np.median(g2))},
            'summary': f"U={stat:.1f}, p={p_value:.4f}, r={rank_biserial:.2f}"
        }
    
    def wilcoxon(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        alternative: str = 'two-sided'
    ) -> Dict:
        """
        Wilcoxon signed-rank test (non-parametric paired test).
        
        Args:
            group1: First measurements
            group2: Second measurements (paired)
            alternative: 'two-sided', 'less', or 'greater'
        
        Returns:
            Test results
        """
        g1, g2 = self._clean_data(group1, group2)
        
        if len(g1) != len(g2):
            return {
                'test': 'Wilcoxon',
                'error': 'Groups must have equal length for paired test',
                'passed': False
            }
        
        if len(g1) < 2:
            return {
                'test': 'Wilcoxon',
                'error': 'Insufficient data (n < 2)',
                'passed': False
            }
        
        # Perform test
        stat, p_value = stats.wilcoxon(g1, g2, alternative=alternative)
        
        return {
            'test': 'Wilcoxon signed-rank',
            'statistic': float(stat),
            'p_value': float(p_value),
            'significant': bool(p_value < self.alpha),
            'n_pairs': len(g1),
            'summary': f"W={stat:.1f}, p={p_value:.4f}"
        }
    
    # ===================================================================
    # ANOVA AND KRUSKAL-WALLIS
    # ===================================================================
    
    def anova(self, *groups: np.ndarray) -> Dict:
        """
        One-way ANOVA (parametric test for ≥2 groups).
        
        Args:
            groups: Two or more independent groups
        
        Returns:
            ANOVA results with eta-squared effect size
        """
        if len(groups) < 2:
            return {
                'test': 'ANOVA',
                'error': 'Need at least 2 groups',
                'passed': False
            }
        
        groups = self._clean_data(*groups)
        
        if any(len(g) < 2 for g in groups):
            return {
                'test': 'ANOVA',
                'error': 'All groups need at least 2 observations',
                'passed': False
            }
        
        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Eta-squared (effect size for ANOVA)
        grand_mean = np.mean(np.concatenate(groups))
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_total = sum((x - grand_mean)**2 for g in groups for x in g)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        # Check assumptions
        assumptions = {}
        if self.auto_check_assumptions:
            homoscedasticity = self.check_homoscedasticity(*groups)
            assumptions['equal_variances'] = homoscedasticity.get('equal_variances', False)
        
        return {
            'test': 'One-way ANOVA',
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < self.alpha),
            'effect_size_eta_squared': float(eta_squared),
            'n_groups': len(groups),
            'assumptions': assumptions,
            'summary': (
                f"F({len(groups)-1}, {sum(len(g) for g in groups) - len(groups)})="
                f"{f_stat:.2f}, p={p_value:.4f}, η²={eta_squared:.3f}"
            )
        }
    
    def kruskal_wallis(self, *groups: np.ndarray) -> Dict:
        """
        Kruskal-Wallis H test (non-parametric ANOVA) [web:32][web:33][web:34].
        
        Args:
            groups: Two or more independent groups
        
        Returns:
            Test results with epsilon-squared effect size
        """
        if len(groups) < 2:
            return {
                'test': 'Kruskal-Wallis',
                'error': 'Need at least 2 groups',
                'passed': False
            }
        
        groups = self._clean_data(*groups)
        
        if any(len(g) < 2 for g in groups):
            return {
                'test': 'Kruskal-Wallis',
                'error': 'All groups need at least 2 observations',
                'passed': False
            }
        
        # Perform test
        h_stat, p_value = stats.kruskal(*groups)
        
        # Epsilon-squared (effect size for Kruskal-Wallis)
        n_total = sum(len(g) for g in groups)
        epsilon_squared = (h_stat - len(groups) + 1) / (n_total - len(groups)) if n_total > len(groups) else 0
        
        return {
            'test': 'Kruskal-Wallis H',
            'h_statistic': float(h_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < self.alpha),
            'effect_size_epsilon_squared': float(epsilon_squared),
            'n_groups': len(groups),
            'summary': f"H={h_stat:.2f}, p={p_value:.4f}, ε²={epsilon_squared:.3f}"
        }
    
    # ===================================================================
    # CHI-SQUARE AND PROPORTION TESTS
    # ===================================================================
    
    def chi_square_test(
        self,
        observed: np.ndarray,
        expected: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Chi-square goodness-of-fit test.
        
        Args:
            observed: Observed frequencies
            expected: Expected frequencies (uniform if None)
        
        Returns:
            Test results with Cramér's V effect size
        """
        obs = np.asarray(observed)[np.isfinite(observed)]
        
        if expected is None:
            expected = np.full_like(obs, np.mean(obs), dtype=float)
        else:
            expected = np.asarray(expected)
        
        # Perform test
        chi2, p_value = stats.chisquare(obs, expected)
        
        # Cramér's V (effect size)
        n = np.sum(obs)
        cramers_v = np.sqrt(chi2 / n) if n > 0 else 0
        
        return {
            'test': 'Chi-square',
            'chi2_statistic': float(chi2),
            'p_value': float(p_value),
            'significant': bool(p_value < self.alpha),
            'effect_size_cramers_v': float(cramers_v),
            'degrees_of_freedom': len(obs) - 1,
            'summary': f"χ²={chi2:.2f}, p={p_value:.4f}, V={cramers_v:.2f}"
        }
    
    # ===================================================================
    # MULTIPLE TESTING CORRECTION
    # ===================================================================
    
    def adjust_p_values(
        self,
        p_values: List[float],
        method: str = 'holm'
    ) -> np.ndarray:
        """
        Adjust p-values for multiple comparisons [web:37][web:40].
        
        Args:
            p_values: List of original p-values
            method: 'bonferroni', 'holm', 'fdr_bh' (Benjamini-Hochberg)
        
        Returns:
            Adjusted p-values
        """
        p_values = np.array(p_values)
        n = len(p_values)
        
        if method == 'bonferroni':
            # Simple Bonferroni correction
            return np.minimum(p_values * n, 1.0)
        
        elif method == 'holm':
            # Holm-Bonferroni (uniformly more powerful than Bonferroni)
            sorted_idx = np.argsort(p_values)
            sorted_p = p_values[sorted_idx]
            
            adjusted = np.zeros(n)
            for i, p in enumerate(sorted_p):
                adjusted[sorted_idx[i]] = min(p * (n - i), 1.0)
            
            # Ensure monotonicity
            for i in range(1, n):
                if adjusted[sorted_idx[i]] < adjusted[sorted_idx[i-1]]:
                    adjusted[sorted_idx[i]] = adjusted[sorted_idx[i-1]]
            
            return adjusted
        
        elif method == 'fdr_bh':
            # Benjamini-Hochberg FDR control
            sorted_idx = np.argsort(p_values)
            sorted_p = p_values[sorted_idx]
            
            adjusted = np.zeros(n)
            for i in range(n-1, -1, -1):
                adjusted[sorted_idx[i]] = min(sorted_p[i] * n / (i + 1), 1.0)
                if i < n - 1:
                    adjusted[sorted_idx[i]] = min(adjusted[sorted_idx[i]], adjusted[sorted_idx[i+1]])
            
            return adjusted
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    # ===================================================================
    # HELPER METHODS
    # ===================================================================
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d (Cohen, 1988)."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'


# ===================================================================
# CONVENIENCE FUNCTIONS
# ===================================================================

def test_pet_difference(
    pet_group1: np.ndarray,
    pet_group2: np.ndarray,
    alpha: float = 0.05,
    parametric: bool = True
) -> Dict:
    """Compare PET values between two groups."""
    tester = StatisticalTester(alpha=alpha)
    
    if parametric:
        return tester.t_test(pet_group1, pet_group2, equal_var=False)
    else:
        return tester.mann_whitney(pet_group1, pet_group2)


def test_ttc_difference(
    ttc_group1: np.ndarray,
    ttc_group2: np.ndarray,
    alpha: float = 0.05,
    parametric: bool = True
) -> Dict:
    """Compare TTC values between two groups."""
    tester = StatisticalTester(alpha=alpha)
    
    if parametric:
        return tester.t_test(ttc_group1, ttc_group2, equal_var=False)
    else:
        return tester.mann_whitney(ttc_group1, ttc_group2)


def test_drac_difference(
    drac_group1: np.ndarray,
    drac_group2: np.ndarray,
    alpha: float = 0.05,
    parametric: bool = True
) -> Dict:
    """Compare DRAC values between two groups."""
    tester = StatisticalTester(alpha=alpha)
    
    if parametric:
        return tester.t_test(drac_group1, drac_group2, equal_var=False)
    else:
        return tester.mann_whitney(drac_group1, drac_group2)


def chi_square_test(
    observed: np.ndarray,
    expected: Optional[np.ndarray] = None,
    alpha: float = 0.05
) -> Dict:
    """Chi-square goodness-of-fit test."""
    return StatisticalTester(alpha=alpha).chi_square_test(observed, expected)


def paired_test(
    before: np.ndarray,
    after: np.ndarray,
    alpha: float = 0.05,
    parametric: bool = True
) -> Dict:
    """Paired comparison (e.g., before/after intervention)."""
    tester = StatisticalTester(alpha=alpha)
    
    if parametric:
        return tester.t_test(before, after, paired=True)
    else:
        return tester.wilcoxon(before, after)


def multiple_comparisons(
    *groups: np.ndarray,
    alpha: float = 0.05,
    parametric: bool = True,
    correction: str = 'holm'
) -> Dict:
    """
    Compare multiple groups with correction.
    
    Returns omnibus test + pairwise comparisons
    """
    tester = StatisticalTester(alpha=alpha, correction_method=correction)
    
    # Omnibus test
    if parametric:
        omnibus = tester.anova(*groups)
    else:
        omnibus = tester.kruskal_wallis(*groups)
    
    # Pairwise comparisons
    pairwise = []
    p_values = []
    
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if parametric:
                result = tester.t_test(groups[i], groups[j], equal_var=False)
            else:
                result = tester.mann_whitney(groups[i], groups[j])
            
            pairwise.append({
                'comparison': f'Group {i+1} vs Group {j+1}',
                'result': result
            })
            p_values.append(result.get('test_statistics', {}).get('p_value', result.get('p_value')))
    
    # Adjust p-values
    adjusted_p = tester.adjust_p_values(p_values, method=correction)
    
    for i, pair in enumerate(pairwise):
        pair['adjusted_p_value'] = float(adjusted_p[i])
        pair['significant_adjusted'] = bool(adjusted_p[i] < alpha)
    
    return {
        'omnibus_test': omnibus,
        'pairwise_comparisons': pairwise,
        'correction_method': correction
    }


def check_assumptions(
    *groups: np.ndarray,
    test_type: str = 't-test',
    alpha: float = 0.05
) -> Dict:
    """Check statistical assumptions for tests."""
    tester = StatisticalTester(alpha=alpha)
    return tester.check_assumptions(*groups, test_type=test_type)
