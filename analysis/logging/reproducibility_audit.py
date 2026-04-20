"""Reproducibility Audit Module.

Tracks environment state, package versions, and file hashes
to ensure experiment reproducibility.
"""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AuditReport:
    """Reproducibility audit report."""
    timestamp: str
    environment: Dict[str, Any]
    packages: Dict[str, str]
    file_hashes: Dict[str, str]
    git_info: Optional[Dict[str, str]] = None
    reproducible: bool = True
    mismatches: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ReproducibilityAuditor:
    """Auditor for experiment reproducibility.
    
    Captures environment state, package versions, file hashes,
    and git commit info for reproducibility verification.
    """
    
    def __init__(self):
        self._session_packages: Optional[Dict[str, str]] = None
        self._start_time: Optional[str] = None
    
    def start_session(self, config: Optional[Dict] = None) -> None:
        """Start a reproducibility audit session."""
        self._start_time = datetime.utcnow().isoformat()
        self._session_packages = self._get_package_versions()
        self._config = config
    
    def _get_package_versions(self) -> Dict[str, str]:
        """Get current package versions."""
        import importlib
        key_packages = ['numpy', 'torch', 'torchvision', 'opencv_python',
                        'pandas', 'matplotlib', 'scikit_learn', 'scipy']
        versions = {}
        for pkg in key_packages:
            try:
                mod = importlib.import_module(pkg)
                versions[pkg] = getattr(mod, '__version__', 'unknown')
            except ImportError:
                versions[pkg] = 'not_installed'
        return versions
    
    def _get_git_info(self, repo_path: str = '.') -> Optional[Dict[str, str]]:
        """Get git repository information."""
        try:
            result = subprocess.run(
                ['git', '-C', repo_path, 'rev-parse', 'HEAD'],
                capture_output=True, text=True, check=True
            )
            commit = result.stdout.strip()
            result = subprocess.run(
                ['git', '-C', repo_path, 'branch', '--show-current'],
                capture_output=True, text=True, check=True
            )
            branch = result.stdout.strip()
            return {'commit': commit, 'branch': branch}
        except (subprocess.SubprocessError, FileNotFoundError):
            return None
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """Get environment information."""
        return {
            'python_version': sys.version,
            'platform': sys.platform,
            'cpu_count': os.cpu_count(),
            'start_time': self._start_time,
            'config': self._config,
        }
    
    def generate_report(
        self,
        save_path: Optional[str] = None,
        files_to_hash: Optional[List[str]] = None,
    ) -> AuditReport:
        """Generate a full reproducibility audit report.
        
        Args:
            save_path: Optional path to save JSON report
            files_to_hash: Optional list of files to hash
            
        Returns:
            AuditReport with all reproducibility information
        """
        file_hashes = {}
        if files_to_hash:
            for fpath in files_to_hash:
                file_hashes[fpath] = hash_file(fpath)
        
        report = AuditReport(
            timestamp=self._start_time or datetime.utcnow().isoformat(),
            environment=self._get_environment_info(),
            packages=self._session_packages or {},
            file_hashes=file_hashes,
            git_info=self._get_git_info(),
        )
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
        
        return report
    
    def check_reproducibility(
        self,
        saved_packages: Dict[str, str],
    ) -> Dict[str, Any]:
        """Check if current environment matches a saved state.
        
        Args:
            saved_packages: Previously saved package versions
            
        Returns:
            Dict with 'reproducible' bool and 'mismatches' list
        """
        current = self._get_package_versions()
        mismatches = []
        for pkg, ver in saved_packages.items():
            if pkg in current and current[pkg] != ver:
                mismatches.append(f"{pkg}: {ver} vs {current[pkg]}")
        return {
            'reproducible': len(mismatches) == 0,
            'mismatches': mismatches,
        }


def hash_file(path: str) -> str:
    """Compute SHA256 hash of a file."""
    try:
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        return 'unavailable'


def audit_environment(save_path: Optional[str] = None) -> Dict:
    """Convenience function for quick environment audit."""
    auditor = ReproducibilityAuditor()
    auditor.start_session()
    report = auditor.generate_report(save_path)
    return asdict(report)


def generate_audit_report(
    config: Optional[Dict] = None,
    save_path: Optional[str] = None,
) -> AuditReport:
    """Generate an audit report with optional config."""
    auditor = ReproducibilityAuditor()
    auditor.start_session(config)
    return auditor.generate_report(save_path)
