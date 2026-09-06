
import hashlib
import os
import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.analysis.logging.reproducibility_audit import ReproducibilityAuditor


@given(st.integers(min_value=0, max_value=10**12))
def test_format_bytes_non_negative(size_bytes):
    auditor = ReproducibilityAuditor(project_root=tempfile.mkdtemp())
    result = auditor._format_bytes(size_bytes)
    assert isinstance(result, str)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    assert any(unit in result for unit in units)
    # zero case
    if size_bytes == 0:
        assert result.startswith("0.00 B")

@given(st.binary(min_size=0, max_size=1000))
def test_hash_file_matches_manual(data):
    # Write data to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(data)
        temp_path = f.name

    try:
        auditor = ReproducibilityAuditor(project_root=tempfile.mkdtemp())
        result = auditor._hash_file(temp_path, algorithm="sha256")
        # Compute expected hash manually
        expected = hashlib.sha256(data).hexdigest()[:16]
        assert result == expected
    finally:
        os.unlink(temp_path)

def test_hash_file_nonexistent_returns_error():
    auditor = ReproducibilityAuditor(project_root=tempfile.mkdtemp())
    result = auditor._hash_file("/nonexistent/file/path", algorithm="sha256")
    assert isinstance(result, str)
    assert result.startswith("error:")
