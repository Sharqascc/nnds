# Experimental Modules (Not Used in Paper Results)
## Diffusion (`src/diffusion/`)
- `train_trajectory_diffusion.py`
- `trajectory_diffusion.py`
- `transformer_diffusion.py`
- `train_position_ddpm.py`
- `complete_ddpm.py`
**Why not used:** Paper reports PET from deterministic geometric tracking, not generative diffusion.
## VLM (`src/vlm/`)
- `vlm_enhanced_pipeline.py`
- `gate_validator.py`
- `analyzer.py`
**Why not used:** Gate validation is geometric, not language-model-based.
## How to identify
If a module imports from `src.diffusion` or `src.vlm`, it is experimental.
