# NNDS Migration Guide

## For Users Upgrading to Cleaned Repository

If you're using NNDS after the cleanup, here are the import changes you need to make:

### Import Updates

#### Before (Old Structure)
```python
from traffic_analyzer import TrafficAnalyzer
from bev_mapper import BEVMapper
from pet_conflict_checker import ConflictDetector
from gate_counter import GateCounter
```

#### After (New Structure - Phase 2)
```python
from core import (
    TrafficAnalyzer,
    BEVMapper,
    ConflictDetector,
    GateCounter,
)
```

### File Locations

| Old Path | New Path | Status |
|----------|----------|--------|
| `traffic_analyzer.py` | `core/traffic_analyzer.py` | Phase 2 |
| `bev_mapper.py` | `core/bev_mapper.py` | Phase 2 |
| `pet_conflict_checker.py` | `core/pet_conflict_checker.py` | Phase 2 |
| `gate_counter.py` | `core/gate_counter.py` | Phase 2 |
| `giti_bev_calib.py` | `core/calibration.py` | Phase 2 |
| `dependency_graph.png` | `docs/images/dependency_graph.png` | Phase 2 |

### Shell Script Updates

#### Old Colab Setup
```bash
# Old: Used bootstrap_nnds_session.sh + colab_ready.py separately
!bash bootstrap_nnds_session.sh
!python colab_ready.py
```

#### New Colab Setup (Phase 2)
```bash
# New: Single consolidated script
!bash scripts/bootstrap.sh
```

### Makefile Changes

#### Before
```bash
make install
make grid
make test
```

#### After (Phase 2 - Enhanced)
```bash
make install    # Install dependencies
make grid       # Run video-to-PET pipeline
make test       # Run tests
make clean      # Remove __pycache__ and .pyc files
make format     # Format code with black/isort
make lint       # Run linting checks
make type-check # Run mypy type checking
```

### Configuration Files

No changes needed - configuration files remain in `configs/` directory:
```python
from pathlib import Path

CONFIG_DIR = Path(__file__).parent / "configs"
BEV_CONFIG = CONFIG_DIR / "bev_config.json"
GRID_CONFIG = CONFIG_DIR / "GITI_grid_config.json"
```

## Troubleshooting

### Import Error: ModuleNotFoundError

**Problem:**
```
ModuleNotFoundError: No module named 'traffic_analyzer'
```

**Solution:**
```python
# Update import to:
from core import TrafficAnalyzer

# Or ensure PYTHONPATH is set:
# export PYTHONPATH=.
```

### File Not Found Error

**Problem:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'traffic_analyzer.py'
```

**Solution:**
```bash
# Update file paths:
cd nnds
PYTHONPATH=. python -m core.traffic_analyzer --video videos/traffic_video.mp4
```

## Backward Compatibility

**Phase 1 (Current - .gitignore):**
- ✅ No breaking changes
- ✅ All existing imports still work
- ✅ No action required

**Phase 2 (Planned - Reorganization):**
- ⚠️ Will require import updates
- 📢 Deprecation warnings will be added
- 📋 This guide will be updated

**Phase 3 (Future - Code Quality):**
- ✅ Type hints added (backward compatible)
- ✅ Code formatting only (no functionality changes)

## Getting Help

If you encounter issues after upgrading:

1. Check this migration guide
2. Review updated README.md
3. Run diagnostic tests: `pytest tests/test_imports_smoke.py -v`
4. Check repository issues on GitHub

---

**Last Updated:** 2026-05-07  
**Current Phase:** 1 (Completed)  
**Next Phase:** 2 (Pending)
