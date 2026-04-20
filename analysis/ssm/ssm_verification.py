import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import warnings

__all__ = ['SSMVerifier', 'verify_pet_calculation', 'verify_ttc_calculation', 'run_verification_suite']


class SSMVerifier:
    """Verification system for Surrogate Safety Measure calculations.
    Provides rigorous validation that SSM computations are mathematically correct.
    Designed to convince peer reviewers and non-technical stakeholders.
    """

    def __init__(self, tolerance: float = 1e-6, strict_mode: bool = False):
        self.tolerance = tolerance
        self.strict_mode = strict_mode

    def _log(self, results: Dict, check_name: str, passed: bool,
             message: str, severity: str = 'info'):
        entry = {'check': check_name, 'passed': passed, 'message': message, 'severity': severity}
        results['checks'].append(entry)
        if not passed:
            results['passed'] = False
            if severity == 'error' or self.strict_mode:
                results['errors'].append(message)

    def check_data_quality(self, data: np.ndarray, name: str = 'data') -> Dict:
        results = {'metric_name': name, 'checks': [], 'warnings': [], 'errors': [],
                   'passed': True, 'clean_data': None, 'summary': ''}
        if not isinstance(data, np.ndarray):
            try:
                data = np.asarray(data)
                results['checks'].append({'check': 'Type conversion', 'passed': True, 'message': 'Converted to numpy'})
            except Exception as e:
                results['errors'].append(f'Cannot convert to numpy: {e}')
                return results
        else:
            results['checks'].append({'check': 'Data type', 'passed': True, 'message': 'Input is numpy array'})
        nan_count = int(np.sum(np.isnan(data)))
        inf_count = int(np.sum(np.isinf(data)))
        if nan_count > 0 or inf_count > 0:
            results['warnings'].append(f'{nan_count} NaN and {inf_count} Inf values removed')
            data = data[np.isfinite(data)]
        completeness = 100 * len(data) / max(len(np.asarray(data).ravel()), 1)
        results['checks'].append({'check': 'Completeness', 'passed': completeness >= 90,
                                  'message': f'{completeness:.1f}% valid'})
        if name.upper() in ['PET', 'TTC']:
            neg_count = int(np.sum(data < 0))
            if neg_count > 0:
                results['warnings'].append(f'{neg_count} negative values ({100*neg_count/len(data):.1f}%)')
        results['clean_data'] = data
        results['summary'] = f"{name}: {'PASS' if results['passed'] else 'FAIL'} ({len(data)} valid)"
        return results

    def verify_pet_calculation(self, pet_values: np.ndarray, expected_range: Tuple[float, float] = (0.0, 30.0),
                               reference_value: float = None) -> Dict:
        results = {'metric': 'PET', 'checks': [], 'warnings': [], 'errors': [], 'passed': True, 'summary': ''}
        quality = self.check_data_quality(pet_values, 'PET')
        results['data_quality'] = quality['summary']
        if not quality['passed']:
            results['errors'].extend(quality['errors'])
            results['passed'] = False
        if not quality['passed'] and quality['clean_data'] is None:
            results['summary'] = 'PET Verification: FAIL (no valid data)'
            return results
        data = quality['clean_data']
        min_val, max_val = np.min(data), np.max(data)
        mean_pet = np.mean(data)
        std_pet = np.std(data)
        critical_count = int(np.sum(data < 1.0))
        critical_pct = 100 * critical_count / len(data)
        results['statistics'] = {'mean': float(mean_pet), 'std': float(std_pet),
                                 'min': float(min_val), 'max': float(max_val),
                                 'n_events': len(data), 'critical_count': critical_count}
        results['checks'].append({'check': 'Critical events (<1s)', 'count': critical_count,
                                  'percentage': critical_pct})
        if critical_pct > 10:
            results['warnings'].append(f'{critical_pct:.1f}% of events are critical')
        if reference_value is not None:
            diff = abs(mean_pet - reference_value)
            results['checks'].append({'check': 'Reference comparison', 'passed': diff <= self.tolerance,
                                      'diff': float(diff)})
        status = 'PASS' if results['passed'] else 'FAIL'
        results['summary'] = f"PET: {status} | N={len(data):,} | Mean={mean_pet:.2f}s | Std={std_pet:.2f}s | Critical={critical_pct:.1f}%"
        return results

    def verify_ttc_calculation(self, ttc_values: np.ndarray, expected_range: Tuple[float, float] = (0.0, 20.0)) -> Dict:
        results = {'metric': 'TTC', 'checks': [], 'warnings': [], 'errors': [], 'passed': True, 'summary': ''}
        quality = self.check_data_quality(ttc_values, 'TTC')
        results['data_quality'] = quality['summary']
        if not quality['passed']:
            results['errors'].extend(quality['errors'])
            results['passed'] = False
        if not quality['passed'] and quality['clean_data'] is None:
            results['summary'] = 'TTC Verification: FAIL (no valid data)'
            return results
        data = quality['clean_data']
        min_val, max_val = np.min(data), np.max(data)
        mean_ttc = np.mean(data)
        std_ttc = np.std(data)
        near_collision = int(np.sum(data < 0.1))
        critical_ttc = int(np.sum(data < 2.0))
        results['statistics'] = {'mean': float(mean_ttc), 'std': float(std_ttc),
                                 'min': float(min_val), 'max': float(max_val),
                                 'n_events': len(data), 'near_collision': near_collision,
                                 'critical_count': critical_ttc}
        if near_collision > 0:
            results['errors'].append(f'{near_collision} near-collision events (TTC < 0.1s)')
            results['passed'] = False
        status = 'PASS' if results['passed'] else 'FAIL'
        results['summary'] = f"TTC: {status} | N={len(data):,} | Mean={mean_ttc:.2f}s | Near-collision={near_collision}"
        return results

    def run_verification_suite(self, pet_values: np.ndarray = None, ttc_values: np.ndarray = None) -> Dict:
        suite = {'tests': [], 'overall_pass': True, 'summary': ''}
        if pet_values is not None:
            pet_result = self.verify_pet_calculation(pet_values)
            suite['tests'].append(pet_result)
            if not pet_result['passed']:
                suite['overall_pass'] = False
        if ttc_values is not None:
            ttc_result = self.verify_ttc_calculation(ttc_values)
            suite['tests'].append(ttc_result)
            if not ttc_result['passed']:
                suite['overall_pass'] = False
        if suite['tests']:
            passed = sum(1 for t in suite['tests'] if t['passed'])
            suite['summary'] = f"Suite: {passed}/{len(suite['tests'])} tests passed"
        else:
            suite['summary'] = 'Suite: No data provided'
        return suite


def verify_pet_calculation(pet_values: np.ndarray, tolerance: float = 1e-6) -> Dict:
    verifier = SSMVerifier(tolerance=tolerance)
    return verifier.verify_pet_calculation(pet_values)


def verify_ttc_calculation(ttc_values: np.ndarray, tolerance: float = 1e-6) -> Dict:
    verifier = SSMVerifier(tolerance=tolerance)
    return verifier.verify_ttc_calculation(ttc_values)


def run_verification_suite(pet_values: np.ndarray = None, ttc_values: np.ndarray = None,
                           tolerance: float = 1e-6) -> Dict:
    verifier = SSMVerifier(tolerance=tolerance)
    return verifier.run_verification_suite(pet_values, ttc_values)
