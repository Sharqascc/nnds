"""
Tests for PET summary module.
"""
import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.pet_summary import PETEventAnalyzer

def test_pet_event_analyzer_import():
    """Test that PETEventAnalyzer can be imported."""
    assert PETEventAnalyzer is not None

def test_pet_event_analyzer_requires_csv_path():
    """Test PETEventAnalyzer requires csv_path."""
    with pytest.raises(Exception):
        PETEventAnalyzer()  # Missing csv_path should raise error

def test_pet_event_analyzer_initialization():
    """Test PETEventAnalyzer initializes with a CSV."""
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp())
    csv_path = tmp_dir / 'test.csv'
    csv_path.write_text('event_id,pet,conflict_type\n1,0.5,head_on\n2,1.2,rear_end\n')
    
    analyzer = PETEventAnalyzer(str(csv_path))
    assert analyzer is not None
