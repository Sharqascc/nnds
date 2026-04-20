import os
import sys
import hashlib
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

__all__ = ['ReproducibilityAuditor', 'audit_environment', 'hash_file', 'generate_audit_report']


class ReproducibilityAuditor:
    """Systematic reproducibility auditing for research pipelines.
    Captures environment state for exact reproduction of SSM analysis results.
    """

    def __init__(self):
        self.session_start = None
        self.session_data = {}

    def start_session(self, config: Optional[Dict] = None):
        self.session_start = datetime.now().isoformat()
        self.session_data = {
            'session_start': self.session_start,
            'python_version': sys.version,
            'platform': sys.platform,
            'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
            'cpu_count': os.cpu_count(),
            'config': config or {},
            'inputs': {},
            'outputs': {},
            'environment': self._get_package_versions(),
            'git_info': self._get_git_info()
        }

    def _get_package_versions(self) -> Dict[str, str]:
        versions = {}
        for pkg in ['numpy', 'scipy', 'matplotlib', 'pandas', 'torch', 'tensorflow', 'statsmodels']:
            try:
                mod = __import__(pkg.replace('-', '_'))
                versions[pkg] = getattr(mod, '__version__', 'unknown')
            except ImportError:
                pass
        return versions

    def _get_git_info(self) -> Dict[str, str]:
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=5)
            commit = result.stdout.strip() if result.returncode == 0 else 'unknown'
            result = subprocess.run(['git', 'branch', '--show-current'], capture_output=True, text=True, timeout=5)
            branch = result.stdout.strip() if result.returncode == 0 else 'unknown'
            return {'commit': commit, 'branch': branch}
        except Exception:
            return {'commit': 'unknown', 'branch': 'unknown'}

    def log_input(self, name: str, path: str):
        checksum = self._hash_file(path)
        size = os.path.getsize(path) if os.path.exists(path) else 0
        self.session_data['inputs'][name] = {'path': path, 'checksum': checksum, 'size_bytes': size}

    def log_output(self, name: str, path: str):
        checksum = self._hash_file(path)
        size = os.path.getsize(path) if os.path.exists(path) else 0
        self.session_data['outputs'][name] = {'path': path, 'checksum': checksum, 'size_bytes': size}

    def _hash_file(self, path: str) -> str:
        try:
            with open(path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
        except Exception:
            return 'unavailable'

    def generate_report(self, save_path: Optional[str] = None) -> Dict:
        report = {'title': 'NNDS Pipeline Reproducibility Audit', 'generated_at': datetime.now().isoformat(),
                  'session': self.session_data}
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        return report

    def verify_reproducibility(self, saved_report_path: str) -> Dict:
        with open(saved_report_path) as f:
            saved = json.load(f)
        mismatches = []
        if saved['session']['python_version'] != sys.version:
            mismatches.append('Python version differs')
        saved_pkgs = saved['session'].get('environment', {})
        current_pkgs = self._get_package_versions()
        for pkg, ver in saved_pkgs.items():
            if pkg in current_pkgs and current_pkgs[pkg] != ver:
                mismatches.append(f"{pkg}: {ver} vs {current_pkgs[pkg]}")
        return {'reproducible': len(mismatches) == 0, 'mismatches': mismatches}


def audit_environment(save_path: Optional[str] = None) -> Dict:
    auditor = ReproducibilityAuditor()
    auditor.start_session()
    return auditor.generate_report(save_path)


def hash_file(path: str) -> str:
    try:
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        return 'unavailable'


def generate_audit_report(config: Optional[Dict] = None, save_path: Optional[str] = None) -> Dict:
    auditor = ReproducibilityAuditor()
    auditor.start_session(config)
    return auditor.generate_report(save_path)
