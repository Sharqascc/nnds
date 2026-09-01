"""
Tests for reproducibility audit functions.
"""
import numpy as np
import pytest
import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.logging.reproducibility_audit import (
    hash_file,
    audit_environment,
    generate_audit_report,
    verify_reproducibility,
    ReproducibilityAuditor
)

def test_hash_file():
    """Test file hash computation."""
    tmp_dir = Path(tempfile.mkdtemp())
    test_file = tmp_dir / 'test.txt'
    test_file.write_text('hello world')
    
    hash_value = hash_file(str(test_file))
    assert hash_value is not None
    assert isinstance(hash_value, str)
    assert len(hash_value) == 32 or len(hash_value) == 16

def test_hash_file_nonexistent():
    """Test file hash with nonexistent file (returns error string)."""
    tmp_dir = Path(tempfile.mkdtemp())
    test_file = tmp_dir / 'missing.txt'
    result = hash_file(str(test_file))
    assert result is not None

def test_audit_environment():
    """Test environment audit returns dict with expected keys."""
    result = audit_environment()
    assert result is not None
    assert 'title' in result
    assert 'version' in result
    assert 'session' in result
    assert 'system' in result['session']

def test_reproducibility_auditor_initialization():
    """Test ReproducibilityAuditor initialization."""
    auditor = ReproducibilityAuditor()
    assert auditor is not None

def test_reproducibility_auditor_methods():
    """Test ReproducibilityAuditor has expected methods."""
    auditor = ReproducibilityAuditor()
    assert hasattr(auditor, 'generate_report')
    assert hasattr(auditor, 'verify_reproducibility')
    assert hasattr(auditor, '_hash_file')

def test_verify_reproducibility_requires_path():
    """Test verify_reproducibility requires report_path."""
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp())
    report_path = tmp_dir / 'report.json'
    report_path.write_text('{}')
    result = verify_reproducibility(str(report_path))
    assert result is not None
