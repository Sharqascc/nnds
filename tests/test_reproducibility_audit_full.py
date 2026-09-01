
import json
import os
import sys
import platform
import subprocess
import tempfile
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.analysis.logging.reproducibility_audit import (
    ReproducibilityAuditor,
    audit_environment,
    generate_audit_report,
    hash_file,
    verify_reproducibility,
)


@pytest.fixture
def auditor(tmp_path):
    return ReproducibilityAuditor(project_root=tmp_path)


# ---------------- start_session and random seeds ----------------
def test_start_session_with_seed(auditor):
    auditor.start_session(config={"a": 1}, random_seed=123, description="test")
    assert auditor.session_start is not None
    assert auditor.session_data["random_seed"] == 123
    assert "system" in auditor.session_data
    assert "hardware" in auditor.session_data
    assert "git_info" in auditor.session_data


def test_set_random_seeds_torch_import_error(monkeypatch):
    auditor = ReproducibilityAuditor()
    # Simulate torch import failure via sys.modules
    monkeypatch.setitem(sys.modules, "torch", None)
    # numpy import should still succeed, torch branch should be skipped
    auditor._set_random_seeds(42)
    # No exception is enough


# ---------------- random state ----------------
def test_get_random_state(auditor):
    state = auditor._get_random_state()
    assert isinstance(state, dict)
    # Python random is always captured
    assert "python" in state
    # numpy may or may not exist but likely
    # (we don't assert strict keys)


# ---------------- hardware / system ----------------
def test_get_cpu_model_linux(monkeypatch, tmp_path):
    auditor = ReproducibilityAuditor(project_root=tmp_path)
    fake_cpuinfo = tmp_path / "cpuinfo"
    fake_cpuinfo.write_text("model name : Fake CPU Model\n")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("builtins.open", lambda *a, **k: fake_cpuinfo.open() if a and a[0] == "/proc/cpuinfo" else open(*a, **k))
    assert "Fake" in auditor._get_cpu_model()


def test_get_cpu_model_unknown():
    auditor = ReproducibilityAuditor()
    with patch("sys.platform", "unknown"):
        assert auditor._get_cpu_model() == "unknown"


def test_get_total_memory_linux(monkeypatch, tmp_path):
    auditor = ReproducibilityAuditor(project_root=tmp_path)
    fake_meminfo = tmp_path / "meminfo"
    fake_meminfo.write_text("MemTotal: 16384000 kB\n")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("builtins.open", lambda *a, **k: fake_meminfo.open() if a and a[0] == "/proc/meminfo" else open(*a, **k))
    mem = auditor._get_total_memory()
    assert mem == pytest.approx(16384000 / (1024**2), rel=0.01)


def test_get_total_memory_unknown():
    auditor = ReproducibilityAuditor()
    with patch("sys.platform", "unknown"):
        assert auditor._get_total_memory() == 0.0


# ---------------- GPU info ----------------
def test_get_gpu_info_cpu_only():
    auditor = ReproducibilityAuditor()
    # Simulate torch not available or cuda false
    import sys
    with patch.dict(sys.modules, {"torch": None}):
        gpus = auditor._get_gpu_info()
    assert gpus == [{"type": "none", "name": "CPU only"}]


def test_get_gpu_info_mps(monkeypatch):
    auditor = ReproducibilityAuditor()
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = True
    fake_torch.backends.mps = MagicMock()
    with patch.dict(sys.modules, {"torch": fake_torch}):
        gpus = auditor._get_gpu_info()
    assert any(g["type"] == "MPS" for g in gpus)


# ---------------- container detection ----------------
def test_container_info_colab(monkeypatch):
    auditor = ReproducibilityAuditor()
    monkeypatch.setenv("COLAB_GPU", "T4")
    monkeypatch.setenv("COLAB_RELEASE_TAG", "2024")
    info = auditor._get_container_info()
    assert info["type"] == "Google Colab"
    assert info["colab_gpu"] == "T4"


def test_container_info_docker(tmp_path, monkeypatch):
    auditor = ReproducibilityAuditor(project_root=tmp_path)
    # Clear Colab/Kaggle/Singularity env vars
    for var in ["COLAB_GPU", "COLAB_RELEASE_TAG", "COLAB_TPU_ADDR", "KAGGLE_KERNEL_RUN_TYPE", "SINGULARITY_NAME"]:
        monkeypatch.delenv(var, raising=False)
    # Patch Path.exists: /.dockerenv True, /content False
    monkeypatch.setattr(Path, "exists", lambda self: self == Path("/.dockerenv"))
    with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="Docker version 20.10.0\n", stderr="")):
        info = auditor._get_container_info()
    assert info["type"] == "Docker"


def test_container_info_baremetal(monkeypatch):
    auditor = ReproducibilityAuditor()
    # Ensure no container env vars
    for var in ["COLAB_GPU", "COLAB_TPU_ADDR", "KAGGLE_KERNEL_RUN_TYPE", "SINGULARITY_NAME"]:
        monkeypatch.delenv(var, raising=False)
    with patch.object(Path, "exists", return_value=False):
        info = auditor._get_container_info()
    assert info["type"] == "bare-metal"


# ---------------- package versions ----------------
def test_package_versions_pip_freeze_failure(monkeypatch):
    auditor = ReproducibilityAuditor()
    with patch("subprocess.run", side_effect=Exception("no pip")):
        versions = auditor._get_package_versions()
    assert isinstance(versions, dict)
    # Core packages should still be captured without pip freeze
    assert "numpy" in versions


# ---------------- git info ----------------
def test_get_git_info_failure(tmp_path, monkeypatch):
    auditor = ReproducibilityAuditor(project_root=tmp_path)
    with patch("subprocess.run", side_effect=Exception("git missing")):
        info = auditor._get_git_info()
    assert info["commit"] == "unknown"
    assert info["branch"] == "unknown"


# ---------------- code checksums ----------------
def test_get_code_checksums(tmp_path):
    auditor = ReproducibilityAuditor(project_root=tmp_path)
    (tmp_path / "script.py").write_text("print('hello')\n")
    (tmp_path / "config_data.json").write_text('{"key": "value"}')
    checksums = auditor._get_code_checksums()
    assert "script.py" in checksums
    assert "config_data.json" in checksums


# ---------------- logging methods ----------------
def test_log_input_output_intermediate_hyperparams(tmp_path):
    auditor = ReproducibilityAuditor(project_root=tmp_path)
    auditor.start_session(description="test")
    in_file = tmp_path / "input.txt"
    in_file.write_text("input data")
    out_file = tmp_path / "output.txt"
    out_file.write_text("output data")
    inter_file = tmp_path / "inter.txt"
    inter_file.write_text("inter")
    auditor.log_input("in", str(in_file))
    auditor.log_output("out", str(out_file))
    auditor.log_intermediate("inter", str(inter_file))
    auditor.log_hyperparameters({"lr": 0.001})
    assert "in" in auditor.session_data["inputs"]
    assert "out" in auditor.session_data["outputs"]
    assert "inter" in auditor.session_data["intermediate"]
    assert "lr" in auditor.session_data["hyperparameters"]["params"]


# ---------------- hash file error path ----------------
def test_hash_file_error(auditor):
    result = auditor._hash_file("/nonexistent/file.txt")
    assert result.startswith("error:")


# ---------------- format bytes ----------------
def test_format_bytes():
    auditor = ReproducibilityAuditor()
    assert auditor._format_bytes(0) == "0.00 B"
    assert auditor._format_bytes(1024) == "1.00 KB"
    assert auditor._format_bytes(1024*1024) == "1.00 MB"


# ---------------- timers ----------------
def test_timer_start_stop(auditor):
    auditor.start_session()
    auditor.start_timer("task1")
    auditor.stop_timer("task1")
    assert "task1" in auditor.session_data["execution"]["timing"]
    assert "duration_seconds" in auditor.session_data["execution"]["timing"]["task1"]


def test_timer_stop_without_start(auditor):
    auditor.start_session()
    auditor.stop_timer("nonexistent")
    # Should not raise


# ---------------- report generation and verification ----------------
def test_generate_report_save(tmp_path):
    auditor = ReproducibilityAuditor(project_root=tmp_path)
    auditor.start_session(description="test report")
    save_path = tmp_path / "report.json"
    report = auditor.generate_report(str(save_path))
    assert save_path.exists()
    assert report["title"] == "NNDS Pipeline Reproducibility Audit"


def test_verify_reproducibility_missing_file():
    auditor = ReproducibilityAuditor()
    # Should attempt to open and fail? The function will raise FileNotFoundError.
    with pytest.raises(FileNotFoundError):
        auditor.verify_reproducibility("/nonexistent/report.json")


def test_verify_reproducibility_with_mismatches(tmp_path):
    auditor = ReproducibilityAuditor(project_root=tmp_path)
    auditor.start_session(description="verify test")
    report_path = tmp_path / "report.json"
    # Create a saved report with mismatched python version
    saved = {
        "session": {
            "system": {"python_version": "1.0.0 (fake)"},
            "environment": {"nonexistent_pkg": "1.0"},
            "git_info": {"commit": "deadbeef", "dirty": False},
            "inputs": {},
            "random_seed": None,
        }
    }
    report_path.write_text(json.dumps(saved))
    with patch.object(auditor, '_get_package_versions', return_value={}),          patch.object(auditor, '_get_git_info', return_value={"commit": "cafebabe", "dirty": True}):
        result = auditor.verify_reproducibility(str(report_path))
    assert result["reproducible"] == False
    assert len(result["mismatches"]) > 0


# ---------------- additional uncovered branches ----------------

def test_set_random_seeds_cuda_path():
    auditor = ReproducibilityAuditor()
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.manual_seed_all = MagicMock()
    fake_torch.backends.cudnn.deterministic = None
    fake_torch.backends.cudnn.benchmark = None
    with patch.dict(sys.modules, {"torch": fake_torch}):
        auditor._set_random_seeds(42)
    assert fake_torch.cuda.manual_seed_all.called


def test_get_random_state_import_errors():
    auditor = ReproducibilityAuditor()
    # Simulate numpy import failure
    with patch.dict(sys.modules, {"numpy": None}):
        state = auditor._get_random_state()
    assert "python" in state


def test_get_cpu_model_darwin(monkeypatch):
    auditor = ReproducibilityAuditor()
    monkeypatch.setattr(sys, "platform", "darwin")
    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = "Apple M1 Pro\n"
    fake_result.stderr = ""
    with patch("subprocess.run", return_value=fake_result):
        assert "Apple M1 Pro" in auditor._get_cpu_model()


def test_get_total_memory_darwin(monkeypatch):
    auditor = ReproducibilityAuditor()
    monkeypatch.setattr(sys, "platform", "darwin")
    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = "17179869184\n"  # 16 GB
    fake_result.stderr = ""
    with patch("subprocess.run", return_value=fake_result):
        mem = auditor._get_total_memory()
    assert mem == 16.0


def test_get_gpu_info_with_cuda():
    auditor = ReproducibilityAuditor()
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.device_count.return_value = 1
    fake_torch.cuda.get_device_name.return_value = "Tesla T4"
    fake_torch.cuda.get_device_properties.return_value.total_memory = 16 * 1024**3
    fake_torch.cuda.get_device_capability.return_value = (7, 5)
    fake_torch.backends.mps.is_available.return_value = False
    with patch.dict(sys.modules, {"torch": fake_torch}):
        gpus = auditor._get_gpu_info()
    assert gpus[0]["type"] == "CUDA"
    assert gpus[0]["name"] == "Tesla T4"


def test_container_info_docker_exception(monkeypatch):
    auditor = ReproducibilityAuditor()
    for var in ["COLAB_GPU", "COLAB_RELEASE_TAG", "COLAB_TPU_ADDR", "KAGGLE_KERNEL_RUN_TYPE", "SINGULARITY_NAME"]:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(Path, "exists", lambda self: self == Path("/.dockerenv"))
    with patch("subprocess.run", side_effect=Exception("docker not found")):
        info = auditor._get_container_info()
    assert info["type"] == "Docker"
    assert "Docker detected" in info["details"]


def test_container_info_singularity(monkeypatch):
    auditor = ReproducibilityAuditor()
    monkeypatch.setenv("SINGULARITY_NAME", "mycontainer.sif")
    for var in ["COLAB_GPU", "KAGGLE_KERNEL_RUN_TYPE"]:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(Path, "exists", lambda self: False)
    info = auditor._get_container_info()
    assert info["type"] == "Singularity"


def test_container_info_kaggle(monkeypatch):
    auditor = ReproducibilityAuditor()
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "batch")
    for var in ["COLAB_GPU", "SINGULARITY_NAME"]:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(Path, "exists", lambda self: False)
    info = auditor._get_container_info()
    assert info["type"] == "Kaggle"


def test_format_bytes_pb():
    auditor = ReproducibilityAuditor()
    huge = 1024**5  # 1 PB
    assert auditor._format_bytes(huge) == "1.00 PB"


def test_verify_reproducibility_pkg_mismatch(tmp_path):
    auditor = ReproducibilityAuditor(project_root=tmp_path)
    auditor.start_session(description="verify mismatch")
    report_path = tmp_path / "report.json"
    saved = {
        "session": {
            "system": {"python_version": sys.version},
            "environment": {"fake_pkg": "1.0"},
            "git_info": {"commit": auditor._get_git_info()["commit"], "dirty": False},
            "inputs": {},
            "random_seed": None,
        }
    }
    report_path.write_text(json.dumps(saved))
    # Patch current package versions to empty, so fake_pkg missing -> warning
    with patch.object(auditor, '_get_package_versions', return_value={}),          patch.object(auditor, '_get_git_info', return_value={"commit": auditor._get_git_info()["commit"], "dirty": False}):
        result = auditor.verify_reproducibility(str(report_path))
    assert result["reproducible"] == True
    assert any("not installed" in w for w in result["warnings"])


def test_verify_reproducibility_input_checksum_mismatch(tmp_path):
    auditor = ReproducibilityAuditor(project_root=tmp_path)
    auditor.start_session(description="input mismatch")
    in_file = tmp_path / "input.txt"
    in_file.write_text("original content")
    report_path = tmp_path / "report.json"
    saved = {
        "session": {
            "system": {"python_version": sys.version},
            "environment": {},
            "git_info": auditor._get_git_info(),
            "inputs": {
                "in": {"path": str(in_file), "checksum": "deadbeef"}
            },
            "random_seed": 42,
        }
    }
    report_path.write_text(json.dumps(saved))
    with patch.object(auditor, '_get_git_info', return_value=auditor._get_git_info()):
        result = auditor.verify_reproducibility(str(report_path))
    assert result["reproducible"] == False
    assert any("checksum mismatch" in m for m in result["mismatches"])


def test_verify_reproducibility_input_missing(tmp_path):
    auditor = ReproducibilityAuditor(project_root=tmp_path)
    auditor.start_session(description="input missing")
    report_path = tmp_path / "report.json"
    saved = {
        "session": {
            "system": {"python_version": sys.version},
            "environment": {},
            "git_info": auditor._get_git_info(),
            "inputs": {
                "in": {"path": str(tmp_path / "nonexistent.txt"), "checksum": "abc"}
            },
            "random_seed": 7,
        }
    }
    report_path.write_text(json.dumps(saved))
    with patch.object(auditor, '_get_git_info', return_value=auditor._get_git_info()):
        result = auditor.verify_reproducibility(str(report_path))
    assert any("not found" in w for w in result["warnings"])


# ---------------- remaining uncovered branches ----------------

def test_set_random_seeds_numpy_import_error(monkeypatch):
    auditor = ReproducibilityAuditor()
    monkeypatch.setitem(sys.modules, "numpy", None)
    # numpy import will fail, but torch import might also fail or succeed; ensure no exception
    monkeypatch.setitem(sys.modules, "torch", None)
    auditor._set_random_seeds(42)
    # No exception is enough


def test_get_random_state_python_exception(monkeypatch):
    auditor = ReproducibilityAuditor()
    def bad_getstate():
        raise RuntimeError("random state unavailable")
    monkeypatch.setattr("random.getstate", bad_getstate)
    state = auditor._get_random_state()
    assert state["python"] is None


def test_get_random_state_torch_import_error(monkeypatch):
    auditor = ReproducibilityAuditor()
    monkeypatch.setitem(sys.modules, "torch", None)
    state = auditor._get_random_state()
    # Should not raise and should not contain torch keys
    assert "torch" not in state


def test_get_cpu_model_linux_open_failure(monkeypatch):
    auditor = ReproducibilityAuditor()
    monkeypatch.setattr(sys, "platform", "linux")
    def bad_open(path, *args, **kwargs):
        if path == "/proc/cpuinfo":
            raise OSError("cannot open")
        return open(path, *args, **kwargs)
    monkeypatch.setattr("builtins.open", bad_open)
    assert auditor._get_cpu_model() == "unknown"


def test_get_total_memory_linux_open_failure(monkeypatch):
    auditor = ReproducibilityAuditor()
    monkeypatch.setattr(sys, "platform", "linux")
    def bad_open(path, *args, **kwargs):
        if path == "/proc/meminfo":
            raise OSError("cannot open")
        return open(path, *args, **kwargs)
    monkeypatch.setattr("builtins.open", bad_open)
    assert auditor._get_total_memory() == 0.0


def test_generate_audit_report(tmp_path):
    save_path = tmp_path / "report.json"
    report = generate_audit_report(config={"a": 1}, save_path=str(save_path), description="test")
    assert save_path.exists()
    assert report["session"]["config"] == {"a": 1}


def test_verify_reproducibility_pip_freeze_skipped(tmp_path):
    auditor = ReproducibilityAuditor(project_root=tmp_path)
    auditor.start_session(description="pip freeze skip")
    report_path = tmp_path / "report.json"
    saved = {
        "session": {
            "system": {"python_version": sys.version},
            "environment": {"_pip_freeze": "long string"},
            "git_info": auditor._get_git_info(),
            "inputs": {},
            "random_seed": None,
        }
    }
    report_path.write_text(json.dumps(saved))
    with patch.object(auditor, '_get_git_info', return_value=auditor._get_git_info()):
        result = auditor.verify_reproducibility(str(report_path))
    assert result["reproducible"] == True


def test_verify_reproducibility_pkg_version_mismatch(tmp_path):
    auditor = ReproducibilityAuditor(project_root=tmp_path)
    auditor.start_session(description="pkg mismatch")
    report_path = tmp_path / "report.json"
    saved = {
        "session": {
            "system": {"python_version": sys.version},
            "environment": {"numpy": "1.99.0"},
            "git_info": auditor._get_git_info(),
            "inputs": {},
            "random_seed": None,
        }
    }
    report_path.write_text(json.dumps(saved))
    # Current packages has numpy version 1.0.0
    with patch.object(auditor, '_get_package_versions', return_value={"numpy": "1.0.0"}),          patch.object(auditor, '_get_git_info', return_value=auditor._get_git_info()):
        result = auditor.verify_reproducibility(str(report_path))
    assert any("Package numpy" in m for m in result["mismatches"])


def test_get_random_state_torch_cuda_available(monkeypatch):
    auditor = ReproducibilityAuditor()
    fake_torch = MagicMock()
    fake_torch.initial_seed.return_value = 12345
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.initial_seed.return_value = 67890
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    # Ensure numpy exists to avoid import error
    import numpy as np
    monkeypatch.setitem(sys.modules, "numpy", np)
    state = auditor._get_random_state()
    assert "torch" in state
    assert state["torch"] == 12345
    assert state["torch_cuda"] == 67890
