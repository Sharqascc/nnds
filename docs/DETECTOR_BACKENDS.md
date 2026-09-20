# Detector backends

## Primary backend

Use `uvh-coco-fused` for production PET analysis. It produces the canonical structured PET output, including time-based PET, entry/exit timing, trajectory metadata, and source fields.

Example:

```bash
python -m src.pipeline.traffic_analyzer \
    --video data/sample_data/anonymized_traffic_video_50f.mp4 \
    --detector uvh-coco-fused \
    --out-csv outputs/pet_events.csv
```

## Legacy fallback

`yolo-cpu` is retained as an isolated legacy fallback for diagnostics and compatibility tests. It does not define the canonical PET output schema and should not be used for primary scientific analysis.

## Other backends

- `sam3`: supported alternative backend.
- `rtdetr`: experimental and currently not implemented for video PET.
