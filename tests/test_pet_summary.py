"""
Tests for PET summary module.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.pet_summary import PETEventAnalyzer


def test_pet_event_analyzer_import():
    """Test that PETEventAnalyzer can be imported."""
    assert PETEventAnalyzer is not None


def test_pet_event_analyzer_requires_csv_path():
    """Test PETEventAnalyzer requires csv_path."""
    with pytest.raises(Exception):
        PETEventAnalyzer()  # Missing csv_path should raise error


def test_pet_event_analyzer_initialization(pet_csv_path):
    """Test PETEventAnalyzer initializes with a CSV."""
    analyzer = PETEventAnalyzer(str(pet_csv_path))
    assert analyzer is not None
