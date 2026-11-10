# NNDS entry points

## Main entry point (MEP)

Command:

```bash
nnds
```

Implementation:

```text
src/pipeline/traffic_analyzer.py:main
```

`nnds` is the primary user-facing CLI. It runs detection, tracking, BEV
projection, and PET analysis.

## Wrapper

Command:

```bash
nnds-run
```

Implementation:

```text
scripts/run_pipeline.py:main
```

`nnds-run` prepares models and dispatches to `nnds`. It is not the MEP.

## UVH example

```bash
nnds \
  --video path/to/video.mp4 \
  --detector uvh-coco-fused \
  --device cpu \
  --uvh-model data/models/uvh26.pt \
  --coco-person-model data/models/yolo11n.pt \
  --out-csv outputs/petevents_uvh.csv \
  --no-progress
```

## Demo

```bash
nnds --demo
```

`--demo` runs a lightweight self-contained demonstration. It is not the full
UVH video pipeline.
