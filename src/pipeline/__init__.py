"""Traffic-analysis pipeline package.

Public submodules (import the concrete module, not this package):

    src.pipeline.traffic_analyzer  -- CLI + run_pipeline entry point
    src.pipeline.custom_tracker    -- SORT-style tracker
    src.pipeline.preflight         -- --self-check / --dry-run helpers
    src.pipeline.reid_encoder      -- appearance embedding
    src.pipeline.rt_detr_detector  -- RT-DETR backend
    src.pipeline.tracking_state    -- per-track state

This module intentionally re-exports nothing. Eager imports here would pull
in cv2, torch, and openvino when callers only need a single helper. Import
the concrete submodule instead:

    from src.pipeline.traffic_analyzer import parse_args
"""

from __future__ import annotations

__all__: list[str] = []
