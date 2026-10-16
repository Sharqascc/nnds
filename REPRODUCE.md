# Reproduce the PET conflict results

Artifacts frozen at tag **`v1.8-paper`** (commit `bc2bebd`). Original results
were produced at commit `3c79dd2`; the pipeline was repaired and verified in
`#44`/`#45`.

## 1. Clone and install

    git clone https://github.com/Sharqascc/nnds.git
    cd nnds
    git checkout v1.8-paper

    git lfs install && git lfs pull        # fetch sample videos
    python -m pip install -U pip
    pip install -e ".[dev]" || pip install -r requirements-dev.txt
    pip install ultralytics                 # YOLO backbone used by UVH-COCO fusion

## 2. Fetch model weights

    mkdir -p data/models
    # UVH-26 YOLOv11-S (fine-tuned on the UVH-26 traffic dataset)
    wget -O data/models/uvh26.pt \
      "https://huggingface.co/iisc-aim/UVH-26/resolve/main/weights/YOLOv11-S/UVH-26-MV-YOLOv11-S.pt"
    # Person class
    python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" && mv yolo11n.pt data/models/

## 3. Run the pipeline

    python -m src.pipeline.traffic_analyzer \
      --video data/sample_data/GITI_traffic_video.mp4 \
      --out-csv outputs/petevents_bev.csv \
      --detector uvh-coco-fused \
      --uvh-model data/models/uvh26.pt \
      --yolo-weights data/models/yolo11n.pt \
      --device auto

For a fast sanity check use `data/sample_data/anonymized_traffic_video_50f.mp4`
with `--max-frames 50`. On this clip the pipeline emits 0 PET events (1 valid
track over 1.67 s — no pair can form); this is the correct outcome and
confirms the pipeline runs end-to-end.

## 4. Compare against frozen results

    python - <<'PY'
    import pandas as pd
    from pathlib import Path
    out = Path("outputs")
    for site, fname in [("GITI","giti_screened_with_gates.csv"),
                        ("MRC","mrc_screened_with_gates.csv")]:
        df = pd.read_csv(out / fname)
        print(site, len(df), round(df.pet.median(),4))
    PY

Expected: GITI 153 (median 1.5663), MRC 34 (median 1.5996); combined 187,
mean 1.5894, range 0.1333-2.9992; severity 57/32/98/0.

## 5. Verify environment

`outputs/paper_env.txt` records the CPU verification environment. The frozen
CSVs were produced at commit `3c79dd2`; the tag `v1.8-paper` includes the
pipeline fix (`bc2bebd`) and is the recommended revision for re-running.
