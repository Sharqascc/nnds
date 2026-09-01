
import importlib
import sys
import types
import builtins
from unittest.mock import patch
import pytest

def test_analysis_init_import_success():
    import src.analysis as analysis
    assert hasattr(analysis, 'PETEventAnalyzer')
    assert analysis.check_installation()["pet_summary"] == True

