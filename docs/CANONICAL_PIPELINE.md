# Canonical PET Pipeline

The authoritative workflow for final PET event CSV and figures.

## Command

```bash
make reproduce-final
```

This runs:
- `python src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py --detector uvh-coco-fused`
- `python scripts/validate_outputs.py`
- `python scripts/generate_results_table.py`

## Coordinate System

Current PET uses a **pixel-space conflict zone** (20 px radius). This is a trajectory-based PET proxy, not a physically calibrated BEV PET. Document this in any manuscript.

## Outputs

Final outputs in `outputs/final/`:
- `giti_pet_events.csv`
- `mrc_pet_events.csv`
- `pet_summary_table.csv`
