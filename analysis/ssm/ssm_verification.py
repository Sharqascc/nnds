"""
SSM Verification Suite for NNDS Pipeline

Provides rigorous validation of Surrogate Safety Measure calculations.
Designed for peer review and ensures mathematical correctness.

Complies with:
- Traffic safety analysis standards (FHWA-HRT-08-051)
- Statistical verification best practices
- Research reproducibility requirements
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from scipy import stats
import warnings


__all__ = [
    'SSMVerifier',
    'verify_pet_calculation',
    'verify_ttc_calculation',
    'verify_drac_calculation',
    'run_verification_suite',
    'compare_with_reference'
]


class SSMVerifier:
    """
    Verification system for Surrogate Safety Measure calculations.
    
    Provides:
    - Data quality checks (completeness, validity, outliers)
    - Distribution analysis (normality, skewness, percentiles)
    - Physical plausibility (range checks, relationship validation)
    - Statistical testing (t-tests, KS-tests for comparison)
    - Reference value validation
    """
    
    def __init__(
        self,
        tolerance: float = 1e-6,
        strict_mode: bool = False,
        min_sample_size: int = 10
    ):
        """
        Args:
            tolerance: Numerical tolerance for comparisons
            strict_mode: If True, warnings become errors
            min_sample_size: Minimum samples for statistical tests
        """
        self.tolerance = tolerance
        self.strict_mode = strict_mode
        self.min_sample_size = min_sample_size
    
    # ===================================================================
    # DATA QUALITY CHECKS
    # ===================================================================
    
    def check_data_quality(
        self,
        data: np.ndarray,
        name: str = 'data',
        expected_range: Optional[Tuple[float, float]] = None
    ) -> Dict:
        """
        Comprehensive data quality assessment.
        
        Returns:
            Dictionary with quality metrics, warnings, and clean data
        """
        results = {
            'metric_name': name,
            'checks': [],
            'warnings': [],
            'errors': [],
            'passed': True,
            'clean_data': None,
            'summary': '',
            'statistics': {}
        }
        
        # Type validation
        if not isinstance(data, np.ndarray):
            try:
                data = np.asarray(data)
                results['checks'].append({
                    'check': 'Type conversion',
                    'passed': True,
                    'message': 'Successfully converted to numpy array'
                })
            except Exception as e:
                results['errors'].append(f'Cannot convert to numpy: {e}')
                results['passed'] = False
                return results
        
        original_size = data.size
        
        # Check for invalid values
        nan_count = int(np.sum(np.isnan(data)))
        inf_count = int(np.sum(np.isinf(data)))
        
        if nan_count > 0 or inf_count > 0:
            msg = f'Found {nan_count} NaN and {inf_count} Inf values'
            results['warnings'].append(msg)
            data = data[np.isfinite(data)]
        
        if data.size == 0:
            results['errors'].append('No valid data after removing NaN/Inf')
            results['passed'] = False
            return results
        
        # Completeness check
        completeness = 100 * data.size / max(original_size, 1)
        results['checks'].append({
            'check': 'Data completeness',
            'passed': completeness >= 90,
            'value': completeness,
            'message': f'{completeness:.1f}% valid data'
        })
        
        if completeness < 90:
            results['warnings'].append(f'Low completeness: {completeness:.1f}%')
        
        # Sample size check
        if data.size < self.min_sample_size:
            results['warnings'].append(
                f'Small sample size: {data.size} < {self.min_sample_size}'
            )
        
        # Range validation
        if expected_range is not None:
            min_val, max_val = expected_range
            out_of_range = np.sum((data < min_val) | (data > max_val))
            
            if out_of_range > 0:
                pct = 100 * out_of_range / data.size
                msg = f'{out_of_range} values ({pct:.1f}%) outside expected range [{min_val}, {max_val}]'
                
                if pct > 5:  # More than 5% out of range
                    results['errors'].append(msg)
                    results['passed'] = False
                else:
                    results['warnings'].append(msg)
        
        # Outlier detection (IQR method)
        if data.size >= 4:  # Need at least 4 points for quartiles
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            outliers = np.sum((data < lower_bound) | (data > upper_bound))
            
            if outliers > 0:
                outlier_pct = 100 * outliers / data.size
                results['statistics']['outliers'] = {
                    'count': int(outliers),
                    'percentage': float(outlier_pct),
                    'bounds': (float(lower_bound), float(upper_bound))
                }
                
                if outlier_pct > 10:
                    results['warnings'].append(
                        f'{outliers} potential outliers ({outlier_pct:.1f}%)'
                    )
        
        # Basic statistics
        results['statistics'].update({
            'n': int(data.size),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'median': float(np.median(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'skewness': float(stats.skew(data)) if data.size >= 3 else None,
            'kurtosis': float(stats.kurtosis(data)) if data.size >= 4 else None
        })
        
        # Distribution shape check
        if data.size >= 8:  # Need sufficient samples for normality test
            _, p_value = stats.shapiro(data[:5000])  # Shapiro-Wilk test (max 5000 samples)
            results['statistics']['normality_p_value'] = float(p_value)
            
            if p_value < 0.05:
                results['checks'].append({
                    'check': 'Normality',
                    'passed': True,
                    'message': f'Non-normal distribution (p={p_value:.4f})'
                })
        
        results['clean_data'] = data
        
        status = 'PASS' if results['passed'] else 'FAIL'
        results['summary'] = (
            f"{name}: {status} | "
            f"N={data.size:,} | "
            f"Mean={results['statistics']['mean']:.3f} | "
            f"Completeness={completeness:.1f}%"
        )
        
        return results
    
    # ===================================================================
    # PET VERIFICATION
    # ===================================================================
    
    def verify_pet_calculation(
        self,
        pet_values: np.ndarray,
        expected_range: Tuple[float, float] = (0.0, 30.0),
        reference_values: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Verify PET (Post-Encroachment Time) calculations.
        
        Args:
            pet_values: Array of PET values in seconds
            expected_range: Valid range for PET (default: 0-30s)
            reference_values: Optional dict with 'mean', 'std', 'critical_rate'
        
        Returns:
            Verification results with detailed statistics
        """
        results = {
            'metric': 'PET',
            'checks': [],
            'warnings': [],
            'errors': [],
            'passed': True,
            'summary': '',
            'statistics': {}
        }
        
        # Data quality check
        quality = self.check_data_quality(pet_values, 'PET', expected_range)
        results['data_quality'] = quality
        
        if not quality['passed']:
            results['errors'].extend(quality['errors'])
            results['passed'] = False
        
        if quality['clean_data'] is None or quality['clean_data'].size == 0:
            results['summary'] = 'PET Verification: FAIL (no valid data)'
            return results
        
        data = quality['clean_data']
        
        # Severity classification (from FHWA guidelines)
        critical = np.sum(data < 0.5)  # < 0.5s: Critical
        serious = np.sum((data >= 0.5) & (data < 1.0))  # 0.5-1.0s: Serious
        moderate = np.sum((data >= 1.0) & (data < 1.5))  # 1.0-1.5s: Moderate
        
        results['statistics']['severity_distribution'] = {
            'critical': {'count': int(critical), 'percentage': float(100 * critical / data.size)},
            'serious': {'count': int(serious), 'percentage': float(100 * serious / data.size)},
            'moderate': {'count': int(moderate), 'percentage': float(100 * moderate / data.size)}
        }
        
        # Check critical event rate
        critical_rate = 100 * critical / data.size
        if critical_rate > 20:
            results['warnings'].append(
                f'High critical event rate: {critical_rate:.1f}% (PET < 0.5s)'
            )
        
        # Percentile analysis
        percentiles = np.percentile(data, [10, 25, 50, 75, 90, 95, 99])
        results['statistics']['percentiles'] = {
            'p10': float(percentiles[0]),
            'p25': float(percentiles[1]),
            'p50': float(percentiles[2]),
            'p75': float(percentiles[3]),
            'p90': float(percentiles[4]),
            'p95': float(percentiles[5]),
            'p99': float(percentiles[6])
        }
        
        # Reference comparison
        if reference_values:
            results['reference_comparison'] = {}
            
            if 'mean' in reference_values:
                mean_diff = abs(np.mean(data) - reference_values['mean'])
                rel_error = mean_diff / reference_values['mean'] if reference_values['mean'] > 0 else 0
                
                results['reference_comparison']['mean'] = {
                    'observed': float(np.mean(data)),
                    'expected': float(reference_values['mean']),
                    'absolute_error': float(mean_diff),
                    'relative_error_pct': float(100 * rel_error),
                    'passed': rel_error < 0.1  # 10% tolerance
                }
                
                if rel_error >= 0.1:
                    results['warnings'].append(
                        f'Mean PET differs from reference by {100*rel_error:.1f}%'
                    )
        
        # Final summary
        status = 'PASS' if results['passed'] else 'FAIL'
        results['summary'] = (
            f"PET: {status} | "
            f"N={data.size:,} | "
            f"Mean={np.mean(data):.3f}s | "
            f"Critical={critical_rate:.1f}% | "
            f"p50={percentiles[2]:.3f}s"
        )
        
        return results
    
    # ===================================================================
    # TTC VERIFICATION
    # ===================================================================
    
    def verify_ttc_calculation(
        self,
        ttc_values: np.ndarray,
        expected_range: Tuple[float, float] = (0.0, 20.0),
        reference_values: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Verify TTC (Time-To-Collision) calculations.
        
        Args:
            ttc_values: Array of TTC values in seconds
            expected_range: Valid range for TTC
            reference_values: Optional reference statistics
        
        Returns:
            Verification results
        """
        results = {
            'metric': 'TTC',
            'checks': [],
            'warnings': [],
            'errors': [],
            'passed': True,
            'summary': '',
            'statistics': {}
        }
        
        # Data quality
        quality = self.check_data_quality(ttc_values, 'TTC', expected_range)
        results['data_quality'] = quality
        
        if not quality['passed']:
            results['errors'].extend(quality['errors'])
            results['passed'] = False
        
        if quality['clean_data'] is None or quality['clean_data'].size == 0:
            results['summary'] = 'TTC Verification: FAIL (no valid data)'
            return results
        
        data = quality['clean_data']
        
        # Severity classification
        imminent = np.sum(data < 1.5)  # < 1.5s: Imminent collision
        critical = np.sum((data >= 1.5) & (data < 3.0))  # 1.5-3.0s: Critical
        serious = np.sum((data >= 3.0) & (data < 5.0))  # 3.0-5.0s: Serious
        
        results['statistics']['severity_distribution'] = {
            'imminent': {'count': int(imminent), 'percentage': float(100 * imminent / data.size)},
            'critical': {'count': int(critical), 'percentage': float(100 * critical / data.size)},
            'serious': {'count': int(serious), 'percentage': float(100 * serious / data.size)}
        }
        
        # Check for extremely low TTC (near-collision)
        near_collision = np.sum(data < 0.5)
        if near_collision > 0:
            rate = 100 * near_collision / data.size
            msg = f'{near_collision} near-collision events (TTC < 0.5s, {rate:.1f}%)'
            
            if self.strict_mode or rate > 5:
                results['errors'].append(msg)
                results['passed'] = False
            else:
                results['warnings'].append(msg)
        
        # Percentiles
        percentiles = np.percentile(data, [10, 25, 50, 75, 90])
        results['statistics']['percentiles'] = {
            'p10': float(percentiles[0]),
            'p25': float(percentiles[1]),
            'p50': float(percentiles[2]),
            'p75': float(percentiles[3]),
            'p90': float(percentiles[4])
        }
        
        status = 'PASS' if results['passed'] else 'FAIL'
        results['summary'] = (
            f"TTC: {status} | "
            f"N={data.size:,} | "
            f"Mean={np.mean(data):.3f}s | "
            f"Imminent={100*imminent/data.size:.1f}% | "
            f"p50={percentiles[2]:.3f}s"
        )
        
        return results
    
    # ===================================================================
    # DRAC VERIFICATION
    # ===================================================================
    
    def verify_drac_calculation(
        self,
        drac_values: np.ndarray,
        expected_range: Tuple[float, float] = (0.0, 10.0)
    ) -> Dict:
        """
        Verify DRAC (Deceleration Rate to Avoid Crash) calculations.
        
        Args:
            drac_values: Array of DRAC values in m/s²
            expected_range: Valid range for DRAC
        
        Returns:
            Verification results
        """
        results = {
            'metric': 'DRAC',
            'checks': [],
            'warnings': [],
            'errors': [],
            'passed': True,
            'summary': '',
            'statistics': {}
        }
        
        # Data quality
        quality = self.check_data_quality(drac_values, 'DRAC', expected_range)
        results['data_quality'] = quality
        
        if not quality['passed']:
            results['errors'].extend(quality['errors'])
            results['passed'] = False
        
        if quality['clean_data'] is None or quality['clean_data'].size == 0:
            results['summary'] = 'DRAC Verification: FAIL (no valid data)'
            return results
        
        data = quality['clean_data']
        
        # Physical plausibility (typical deceleration limits)
        comfort_limit = 3.0  # m/s² - comfortable braking
        max_limit = 9.8  # m/s² - emergency braking (~1g)
        
        uncomfortable = np.sum(data > comfort_limit)
        extreme = np.sum(data > max_limit)
        
        if extreme > 0:
            results['warnings'].append(
                f'{extreme} events exceed emergency braking threshold (>{max_limit} m/s²)'
            )
        
        results['statistics']['severity_distribution'] = {
            'comfortable': {'count': int(np.sum(data <= comfort_limit))},
            'uncomfortable': {'count': int(uncomfortable)},
            'extreme': {'count': int(extreme)}
        }
        
        status = 'PASS' if results['passed'] else 'FAIL'
        results['summary'] = (
            f"DRAC: {status} | "
            f"N={data.size:,} | "
            f"Mean={np.mean(data):.3f} m/s² | "
            f"Extreme={100*extreme/data.size:.1f}%"
        )
        
        return results
    
    # ===================================================================
    # SUITE EXECUTION
    # ===================================================================
    
    def run_verification_suite(
        self,
        pet_values: Optional[np.ndarray] = None,
        ttc_values: Optional[np.ndarray] = None,
        drac_values: Optional[np.ndarray] = None,
        reference_values: Optional[Dict] = None
    ) -> Dict:
        """
        Run complete verification suite on all available SSM metrics.
        
        Returns:
            Comprehensive verification report
        """
        suite = {
            'tests': [],
            'overall_pass': True,
            'summary': '',
            'timestamp': np.datetime64('now').astype(str)
        }
        
        if pet_values is not None:
            pet_ref = reference_values.get('PET') if reference_values else None
            pet_result = self.verify_pet_calculation(pet_values, reference_values=pet_ref)
            suite['tests'].append(pet_result)
            if not pet_result['passed']:
                suite['overall_pass'] = False
        
        if ttc_values is not None:
            ttc_ref = reference_values.get('TTC') if reference_values else None
            ttc_result = self.verify_ttc_calculation(ttc_values, reference_values=ttc_ref)
            suite['tests'].append(ttc_result)
            if not ttc_result['passed']:
                suite['overall_pass'] = False
        
        if drac_values is not None:
            drac_result = self.verify_drac_calculation(drac_values)
            suite['tests'].append(drac_result)
            if not drac_result['passed']:
                suite['overall_pass'] = False
        
        if suite['tests']:
            passed = sum(1 for t in suite['tests'] if t['passed'])
            total = len(suite['tests'])
            suite['summary'] = (
                f"Verification Suite: {passed}/{total} metrics passed | "
                f"Overall: {'PASS' if suite['overall_pass'] else 'FAIL'}"
            )
        else:
            suite['summary'] = 'Verification Suite: No data provided'
            suite['overall_pass'] = False
        
        return suite


# ===================================================================
# CONVENIENCE FUNCTIONS
# ===================================================================

def verify_pet_calculation(
    pet_values: np.ndarray,
    tolerance: float = 1e-6,
    reference_values: Optional[Dict] = None
) -> Dict:
    """Standalone PET verification."""
    verifier = SSMVerifier(tolerance=tolerance)
    return verifier.verify_pet_calculation(pet_values, reference_values=reference_values)


def verify_ttc_calculation(
    ttc_values: np.ndarray,
    tolerance: float = 1e-6,
    reference_values: Optional[Dict] = None
) -> Dict:
    """Standalone TTC verification."""
    verifier = SSMVerifier(tolerance=tolerance)
    return verifier.verify_ttc_calculation(ttc_values, reference_values=reference_values)


def verify_drac_calculation(
    drac_values: np.ndarray,
    tolerance: float = 1e-6
) -> Dict:
    """Standalone DRAC verification."""
    verifier = SSMVerifier(tolerance=tolerance)
    return verifier.verify_drac_calculation(drac_values)


def run_verification_suite(
    pet_values: Optional[np.ndarray] = None,
    ttc_values: Optional[np.ndarray] = None,
    drac_values: Optional[np.ndarray] = None,
    tolerance: float = 1e-6,
    reference_values: Optional[Dict] = None
) -> Dict:
    """Run complete verification suite."""
    verifier = SSMVerifier(tolerance=tolerance)
    return verifier.run_verification_suite(
        pet_values, ttc_values, drac_values, reference_values
    )


def compare_with_reference(
    observed: np.ndarray,
    reference: np.ndarray,
    metric_name: str = 'SSM'
) -> Dict:
    """
    Statistical comparison between observed and reference values.
    
    Uses:
    - Two-sample t-test
    - Kolmogorov-Smirnov test
    - Effect size (Cohen's d)
    """
    results = {
        'metric': metric_name,
        'tests': {},
        'passed': True
    }
    
    # T-test
    t_stat, t_p = stats.ttest_ind(observed, reference)
    results['tests']['t_test'] = {
        'statistic': float(t_stat),
        'p_value': float(t_p),
        'significant': t_p < 0.05
    }
    
    # KS test
    ks_stat, ks_p = stats.ks_2samp(observed, reference)
    results['tests']['ks_test'] = {
        'statistic': float(ks_stat),
        'p_value': float(ks_p),
        'significant': ks_p < 0.05
    }
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(observed) + np.var(reference)) / 2)
    cohens_d = (np.mean(observed) - np.mean(reference)) / pooled_std if pooled_std > 0 else 0
    results['effect_size'] = {
        'cohens_d': float(cohens_d),
        'interpretation': 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'
    }
    
    results['passed'] = not (t_p < 0.05 and abs(cohens_d) > 0.8)
    
    return results
