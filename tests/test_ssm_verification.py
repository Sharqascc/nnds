"""
Tests for SSM verification functions.
"""
import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.ssm.ssm_verification import (
    verify_pet_calculation,
    verify_ttc_calculation,
    verify_drac_calculation,
    compare_with_reference,
    run_verification_suite
)

def test_verify_pet_calculation_basic():
    """Test PET verification with basic values."""
    pet_values = np.array([0.5, 0.8, 1.2, 2.0])
    result = verify_pet_calculation(pet_values)
    assert result is not None
    assert isinstance(result, dict)

def test_verify_pet_calculation_empty():
    """Test PET verification with empty array."""
    result = verify_pet_calculation(np.array([]))
    assert result is not None

def test_verify_ttc_calculation_basic():
    """Test TTC verification with basic values."""
    ttc_values = np.array([2.0, 3.0, 4.0, 5.0])
    result = verify_ttc_calculation(ttc_values)
    assert result is not None

def test_verify_drac_calculation_basic():
    """Test DRAC verification with basic values."""
    drac_values = np.array([0.5, 1.0, 1.5, 2.0])
    result = verify_drac_calculation(drac_values)
    assert result is not None

def test_compare_with_reference():
    """Test comparison with reference."""
    observed = np.array([1.0, 2.0, 3.0, 4.0])
    reference = np.array([1.1, 2.1, 3.1, 4.1])
    result = compare_with_reference(observed, reference, metric_name='PET')
    assert result is not None

def test_run_verification_suite():
    """Test complete verification suite."""
    pet_values = np.array([0.5, 0.8, 1.2])
    result = run_verification_suite(pet_values=pet_values)
    assert result is not None

def test_verification_invalid_values():
    """Test verification with invalid values (NaN)."""
    pet_values = np.array([0.5, np.nan, 1.2])
    result = verify_pet_calculation(pet_values)
    assert result is not None  # Should handle NaN gracefully
