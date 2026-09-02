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

Actual final outputs in `outputs/`:
- `giti_screened.csv`
- `mrc_screened.csv`
- `giti_screened_with_gates.csv`
- `mrc_screened_with_gates.csv`
- `giti_raw.csv`
- `mrc_raw.csv`
- `final_screened_summary.json`
- `final_dual_site_figure.png`
- `reproducibility_manifest.json`
- `petevents_bev.csv`

The `make reproduce-final` target is expected to generate these files.
