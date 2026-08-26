import subprocess, sys, os
from pathlib import Path
import pandas as pd

repo = Path(__file__).resolve().parents[1]

def test_sensitivity_analysis_script_exists():
    assert (repo / 'scripts' / 'sensitivity_pet_fragmentation.py').exists()

def test_mot_metrics_script_exists():
    assert (repo / 'scripts' / 'evaluate_mot_metrics.py').exists()
