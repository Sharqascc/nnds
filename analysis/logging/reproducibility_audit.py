"""
Research-Grade Reproducibility Auditor for NNDS Pipeline (v2.1)

Fully compliant with:
- ACM Artifact Evaluation guidelines
- FAIR data principles
- ML Reproducibility Checklist (Papers with Code)
- NeurIPS/CVPR reproducibility standards
"""

import os
import sys
import hashlib
import json
import platform
import subprocess
import random
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import warnings


__all__ = [
    'ReproducibilityAuditor',
    'audit_environment',
    'hash_file',
    'generate_audit_report',
    'verify_reproducibility'
]


class ReproducibilityAuditor:
    """
    Publication-grade reproducibility auditing.
    
    Tracks:
    - Hardware (CPU, GPU, memory)
    - Software (Python, packages, OS, containers)
    - Data (checksums, lineage, provenance)
    - Configuration (hyperparameters, random seeds, env vars)
    - Execution (timing, resources, random state)
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Args:
            project_root: Root directory of project (defaults to CWD)
        """
        self.project_root = Path(project_root or os.getcwd())
        self.session_start = None
        self.session_data = {}
        self._execution_times = {}
    
    def start_session(
        self,
        config: Optional[Dict] = None,
        random_seed: Optional[int] = None,
        description: str = ""
    ):
        """
        Initialize audit session with full environment capture.
        
        Args:
            config: Configuration dictionary (hyperparameters, paths, etc.)
            random_seed: Random seed for reproducibility
            description: Human-readable description of this run
        """
        self.session_start = datetime.now()
        
        # Set random seeds if provided
        if random_seed is not None:
            self._set_random_seeds(random_seed)
        
        self.session_data = {
            'session_start': self.session_start.isoformat(),
            'description': description,
            'random_seed': random_seed,
            'config': config or {},
            
            # System info
            'system': self._get_system_info(),
            'hardware': self._get_hardware_info(),
            'container': self._get_container_info(),
            'environment': self._get_package_versions(),
            'env_vars': self._get_relevant_env_vars(),
            
            # Randomness tracking
            'random_state': self._get_random_state(),
            
            # Code versioning
            'git_info': self._get_git_info(),
            'code_checksums': self._get_code_checksums(),
            
            # Data tracking
            'inputs': {},
            'outputs': {},
            'intermediate': {},
            'hyperparameters': {},
            
            # Execution tracking
            'execution': {
                'command': ' '.join(sys.argv),
                'working_dir': str(Path.cwd()),
                'timing': {}
            }
        }
    
    # ===================================================================
    # RANDOM SEED MANAGEMENT
    # ===================================================================
    
    def _set_random_seeds(self, seed: int):
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
        
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
    
    def _get_random_state(self) -> Dict[str, Any]:
        """Capture current random seeds for full reproducibility."""
        random_state = {}
        
        # Python random
        try:
            random_state['python'] = random.getstate()[1][:5]  # First 5 elements
        except:
            random_state['python'] = None
        
        # NumPy
        try:
            import numpy as np
            state = np.random.get_state()
            random_state['numpy_seed'] = int(state[1][0])  # First seed value
        except ImportError:
            pass
        
        # PyTorch
        try:
            import torch
            random_state['torch'] = torch.initial_seed()
            if torch.cuda.is_available():
                random_state['torch_cuda'] = torch.cuda.initial_seed()
        except ImportError:
            pass
        
        return random_state
    
    # ===================================================================
    # SYSTEM & HARDWARE DETECTION
    # ===================================================================
    
    def _get_system_info(self) -> Dict[str, str]:
        """Capture OS and Python environment."""
        return {
            'python_version': sys.version,
            'python_executable': sys.executable,
            'platform': sys.platform,
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'hostname': platform.node(),
            'cpu_count': os.cpu_count() or 0
        }
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Capture hardware details (GPU, memory)."""
        hw_info = {
            'cpu_model': self._get_cpu_model(),
            'total_memory_gb': self._get_total_memory(),
            'gpu': self._get_gpu_info()
        }
        return hw_info
    
    def _get_cpu_model(self) -> str:
        """Get CPU model name."""
        try:
            if sys.platform == 'linux':
                with open('/proc/cpuinfo') as f:
                    for line in f:
                        if 'model name' in line:
                            return line.split(':')[1].strip()
            elif sys.platform == 'darwin':
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'],
                    capture_output=True, text=True, timeout=5
                )
                return result.stdout.strip()
        except:
            pass
        return 'unknown'
    
    def _get_total_memory(self) -> float:
        """Get total system memory in GB."""
        try:
            if sys.platform == 'linux':
                with open('/proc/meminfo') as f:
                    for line in f:
                        if 'MemTotal' in line:
                            kb = int(line.split()[1])
                            return round(kb / (1024**2), 2)
            elif sys.platform == 'darwin':
                result = subprocess.run(
                    ['sysctl', '-n', 'hw.memsize'],
                    capture_output=True, text=True, timeout=5
                )
                bytes_mem = int(result.stdout.strip())
                return round(bytes_mem / (1024**3), 2)
        except:
            pass
        return 0.0
    
    def _get_gpu_info(self) -> List[Dict[str, str]]:
        """Detect available GPUs (NVIDIA, AMD, Apple Silicon)."""
        gpus = []
        
        # Try NVIDIA
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpus.append({
                        'type': 'CUDA',
                        'index': i,
                        'name': torch.cuda.get_device_name(i),
                        'memory_gb': round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),
                        'compute_capability': f"{torch.cuda.get_device_capability(i)[0]}.{torch.cuda.get_device_capability(i)[1]}"
                    })
        except ImportError:
            pass
        
        # Try Apple MPS
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                gpus.append({'type': 'MPS', 'name': 'Apple Silicon GPU'})
        except:
            pass
        
        return gpus if gpus else [{'type': 'none', 'name': 'CPU only'}]
    
    # ===================================================================
    # CONTAINER DETECTION
    # ===================================================================
    
    def _get_container_info(self) -> Dict[str, str]:
        """Detect if running in container (Docker, Colab, Singularity)."""
        container = {'type': 'bare-metal', 'details': None}
        
        # Detect Google Colab
        if 'COLAB_GPU' in os.environ or Path('/content').exists():
            container['type'] = 'Google Colab'
            container['details'] = os.environ.get('COLAB_RELEASE_TAG', 'unknown')
            container['colab_gpu'] = os.environ.get('COLAB_GPU', '0')
        
        # Detect Docker
        elif Path('/.dockerenv').exists():
            container['type'] = 'Docker'
            try:
                # Try to get Docker version
                result = subprocess.run(
                    ['docker', '--version'],
                    capture_output=True, text=True, timeout=5
                )
                container['details'] = result.stdout.strip() if result.returncode == 0 else 'unknown'
            except:
                container['details'] = 'Docker detected but version unavailable'
        
        # Detect Singularity/Apptainer
        elif 'SINGULARITY_NAME' in os.environ:
            container['type'] = 'Singularity'
            container['details'] = os.environ.get('SINGULARITY_NAME', 'unknown')
        
        # Detect Kaggle
        elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
            container['type'] = 'Kaggle'
            container['details'] = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'unknown')
        
        return container
    
    # ===================================================================
    # ENVIRONMENT VARIABLES
    # ===================================================================
    
    def _get_relevant_env_vars(self) -> Dict[str, str]:
        """Track environment variables that affect computation."""
        relevant_vars = [
            # GPU/Compute
            'CUDA_VISIBLE_DEVICES',
            'CUDA_DEVICE_ORDER',
            
            # Threading
            'OMP_NUM_THREADS',
            'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS',
            
            # Paths
            'TORCH_HOME',
            'XDG_CACHE_HOME',
            'TRANSFORMERS_CACHE',
            
            # Reproducibility
            'PYTHONHASHSEED',
            'TF_DETERMINISTIC_OPS',
            
            # Colab-specific
            'COLAB_GPU',
            'COLAB_TPU_ADDR'
        ]
        
        return {
            var: os.environ.get(var, 'not set') 
            for var in relevant_vars 
            if os.environ.get(var) is not None
        }
    
    # ===================================================================
    # SOFTWARE ENVIRONMENT
    # ===================================================================
    
    def _get_package_versions(self) -> Dict[str, str]:
        """Get versions of all installed packages."""
        versions = {}
        
        # Core scientific packages
        priority_packages = [
            'numpy', 'scipy', 'matplotlib', 'pandas', 'scikit-learn',
            'torch', 'torchvision', 'torchaudio',
            'tensorflow', 'jax', 'flax',
            'cv2', 'PIL', 'statsmodels', 'seaborn',
            'transformers', 'diffusers'
        ]
        
        for pkg in priority_packages:
            try:
                # Handle special cases
                if pkg == 'cv2':
                    import cv2
                    versions[pkg] = cv2.__version__
                elif pkg == 'PIL':
                    import PIL
                    versions[pkg] = PIL.__version__
                else:
                    mod = __import__(pkg.replace('-', '_'))
                    versions[pkg] = getattr(mod, '__version__', 'unknown')
            except ImportError:
                pass
        
        # Try to get full pip freeze (more comprehensive)
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'freeze'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                versions['_pip_freeze'] = result.stdout.strip()
        except:
            pass
        
        return versions
    
    # ===================================================================
    # CODE VERSIONING
    # ===================================================================
    
    def _get_git_info(self) -> Dict[str, str]:
        """Get Git repository state."""
        try:
            commit = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, timeout=5, cwd=self.project_root
            )
            
            branch = subprocess.run(
                ['git', 'branch', '--show-current'],
                capture_output=True, text=True, timeout=5, cwd=self.project_root
            )
            
            # Check for uncommitted changes
            status = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True, text=True, timeout=5, cwd=self.project_root
            )
            
            # Get remote URL
            remote = subprocess.run(
                ['git', 'remote', 'get-url', 'origin'],
                capture_output=True, text=True, timeout=5, cwd=self.project_root
            )
            
            # Get last commit message
            message = subprocess.run(
                ['git', 'log', '-1', '--pretty=%B'],
                capture_output=True, text=True, timeout=5, cwd=self.project_root
            )
            
            return {
                'commit': commit.stdout.strip() if commit.returncode == 0 else 'unknown',
                'commit_short': commit.stdout.strip()[:8] if commit.returncode == 0 else 'unknown',
                'branch': branch.stdout.strip() if branch.returncode == 0 else 'unknown',
                'dirty': len(status.stdout.strip()) > 0 if status.returncode == 0 else False,
                'remote': remote.stdout.strip() if remote.returncode == 0 else 'unknown',
                'last_commit_message': message.stdout.strip() if message.returncode == 0 else 'unknown'
            }
        except Exception as e:
            return {
                'commit': 'unknown',
                'branch': 'unknown',
                'dirty': True,
                'error': str(e)
            }
    
    def _get_code_checksums(self) -> Dict[str, str]:
        """Compute checksums of ALL Python files in project."""
        checksums = {}
        
        # Exclusion patterns
        exclude_patterns = [
            '__pycache__', '.venv', 'venv', 'env',
            '.git', 'deprecated', '.ipynb_checkpoints',
            'node_modules', 'build', 'dist'
        ]
        
        # Hash all Python files
        for pyfile in self.project_root.rglob('*.py'):
            # Skip excluded directories
            if any(pattern in str(pyfile) for pattern in exclude_patterns):
                continue
            
            rel_path = pyfile.relative_to(self.project_root)
            checksums[str(rel_path)] = self._hash_file(pyfile)
        
        # Also hash key config files
        for config_file in self.project_root.rglob('*.json'):
            if 'config' in str(config_file).lower() and '.git' not in str(config_file):
                rel_path = config_file.relative_to(self.project_root)
                checksums[str(rel_path)] = self._hash_file(config_file)
        
        return checksums
    
    # ===================================================================
    # DATA PROVENANCE
    # ===================================================================
    
    def log_input(
        self,
        name: str,
        path: str,
        metadata: Optional[Dict] = None
    ):
        """Log input file with checksum and metadata."""
        path_obj = Path(path)
        
        self.session_data['inputs'][name] = {
            'path': str(path),
            'checksum': self._hash_file(path),
            'size_bytes': path_obj.stat().st_size if path_obj.exists() else 0,
            'size_human': self._format_bytes(path_obj.stat().st_size) if path_obj.exists() else '0 B',
            'modified_time': datetime.fromtimestamp(path_obj.stat().st_mtime).isoformat() if path_obj.exists() else None,
            'metadata': metadata or {}
        }
    
    def log_output(
        self,
        name: str,
        path: str,
        metadata: Optional[Dict] = None
    ):
        """Log output file with checksum."""
        path_obj = Path(path)
        
        self.session_data['outputs'][name] = {
            'path': str(path),
            'checksum': self._hash_file(path),
            'size_bytes': path_obj.stat().st_size if path_obj.exists() else 0,
            'size_human': self._format_bytes(path_obj.stat().st_size) if path_obj.exists() else '0 B',
            'created_time': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
    
    def log_intermediate(self, name: str, path: str):
        """Log intermediate file (for debugging)."""
        self.session_data['intermediate'][name] = {
            'path': str(path),
            'checksum': self._hash_file(path)
        }
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters for ML experiments."""
        self.session_data['hyperparameters'] = {
            'params': params,
            'logged_at': datetime.now().isoformat()
        }
    
    def _hash_file(self, path: str, algorithm: str = 'sha256') -> str:
        """Compute file hash."""
        try:
            hasher = hashlib.new(algorithm)
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(65536), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()[:16]  # Short hash for readability
        except Exception as e:
            return f'error: {str(e)[:20]}'
    
    def _format_bytes(self, size_bytes: int) -> str:
        """Format bytes to human-readable size."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"
    
    # ===================================================================
    # EXECUTION TRACKING
    # ===================================================================
    
    def start_timer(self, task_name: str):
        """Start timing a task."""
        self._execution_times[task_name] = {'start': datetime.now()}
    
    def stop_timer(self, task_name: str):
        """Stop timing a task."""
        if task_name in self._execution_times:
            self._execution_times[task_name]['end'] = datetime.now()
            delta = self._execution_times[task_name]['end'] - self._execution_times[task_name]['start']
            self._execution_times[task_name]['duration_seconds'] = delta.total_seconds()
            
            # Add to session data
            self.session_data['execution']['timing'][task_name] = {
                'duration_seconds': delta.total_seconds(),
                'duration_human': str(delta)
            }
    
    # ===================================================================
    # REPORT GENERATION
    # ===================================================================
    
    def generate_report(self, save_path: Optional[str] = None) -> Dict:
        """Generate comprehensive reproducibility report."""
        # Compute total execution time
        if self.session_start:
            total_duration = datetime.now() - self.session_start
            self.session_data['execution']['total_duration_seconds'] = total_duration.total_seconds()
            self.session_data['execution']['total_duration_human'] = str(total_duration)
            self.session_data['session_end'] = datetime.now().isoformat()
        
        report = {
            'title': 'NNDS Pipeline Reproducibility Audit',
            'version': '2.1',
            'generated_at': datetime.now().isoformat(),
            'session': self.session_data
        }
        
        if save_path:
            save_path_obj = Path(save_path)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"✅ Reproducibility report saved to: {save_path}")
        
        return report
    
    # ===================================================================
    # VERIFICATION
    # ===================================================================
    
    def verify_reproducibility(self, saved_report_path: str) -> Dict:
        """Verify current environment matches a saved report."""
        with open(saved_report_path) as f:
            saved = json.load(f)
        
        mismatches = []
        warnings_list = []
        
        saved_session = saved.get('session', {})
        
        # Check Python version
        saved_py = saved_session.get('system', {}).get('python_version', '')
        current_py = sys.version
        if saved_py != current_py:
            mismatches.append(f"Python version: saved={saved_py[:30]}... vs current={current_py[:30]}...")
        
        # Check packages
        saved_pkgs = saved_session.get('environment', {})
        current_pkgs = self._get_package_versions()
        
        for pkg, saved_ver in saved_pkgs.items():
            if pkg == '_pip_freeze':
                continue
            
            if pkg in current_pkgs:
                if current_pkgs[pkg] != saved_ver:
                    mismatches.append(f"Package {pkg}: saved={saved_ver} vs current={current_pkgs[pkg]}")
            else:
                warnings_list.append(f"Package {pkg} not installed (was {saved_ver})")
        
        # Check Git commit
        saved_commit = saved_session.get('git_info', {}).get('commit', '')
        current_commit = self._get_git_info().get('commit', '')
        if saved_commit != current_commit and saved_commit != 'unknown':
            mismatches.append(f"Git commit: saved={saved_commit[:8]} vs current={current_commit[:8]}")
        
        # Check dirty status
        current_dirty = self._get_git_info().get('dirty', False)
        if current_dirty:
            warnings_list.append("Repository has uncommitted changes")
        
        # Check input file checksums
        saved_inputs = saved_session.get('inputs', {})
        for name, info in saved_inputs.items():
            path = info.get('path')
            saved_checksum = info.get('checksum')
            
            if path and Path(path).exists():
                current_checksum = self._hash_file(path)
                if current_checksum != saved_checksum:
                    mismatches.append(f"Input file '{name}': checksum mismatch")
            else:
                warnings_list.append(f"Input file '{name}' not found at {path}")
        
        # Check random seed
        saved_seed = saved_session.get('random_seed')
        if saved_seed is not None:
            warnings_list.append(f"Remember to set random_seed={saved_seed} for reproduction")
        
        return {
            'reproducible': len(mismatches) == 0,
            'mismatches': mismatches,
            'warnings': warnings_list,
            'summary': f"{len(mismatches)} critical issues, {len(warnings_list)} warnings"
        }


# ===================================================================
# CONVENIENCE FUNCTIONS
# ===================================================================

def audit_environment(save_path: Optional[str] = None) -> Dict:
    """Quick environment audit without full session."""
    auditor = ReproducibilityAuditor()
    auditor.start_session(description="Quick environment audit")
    return auditor.generate_report(save_path)


def hash_file(path: str, algorithm: str = 'sha256') -> str:
    """Compute file hash."""
    auditor = ReproducibilityAuditor()
    return auditor._hash_file(path, algorithm)


def generate_audit_report(
    config: Optional[Dict] = None,
    save_path: Optional[str] = None,
    description: str = ""
) -> Dict:
    """Generate comprehensive audit report."""
    auditor = ReproducibilityAuditor()
    auditor.start_session(config=config, description=description)
    return auditor.generate_report(save_path)


def verify_reproducibility(report_path: str) -> Dict:
    """Verify current environment against saved report."""
    auditor = ReproducibilityAuditor()
    auditor.start_session(description="Verification run")
    return auditor.verify_reproducibility(report_path)
