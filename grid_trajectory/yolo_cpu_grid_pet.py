from __future__ import annotations

from pathlib import Path
import time

import cv2
import pandas as pd

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


def run_yolo_cpu_grid_pet(
    video_path: str,
    weights_path: str,
    output_csv_path: str,
    max_frames: int | None = None,
    imgsz: int = 480,
    conf: float = 0.25,
    classes: list[int] | None = None,
):
    if YOLO is None:
        raise ModuleNotFoundError(
            "Ultralytics YOLO is unavailable in this environment."
        ) from _IMPORT_ERROR

    video_path = str(Path(video_path).resolve())
    weights_path = str(Path(weights_path).resolve())
    output_csv_path = str(Path(output_csv_path).resolve())

    model = YOLO(weights_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    rows = []
    frame_idx = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        if max_frames is not None and frame_idx > max_frames:
            break

        results = model.predict(
            source=frame,
            imgsz=imgsz,
            conf=conf,
            device="cpu",
            verbose=False,
            classes=classes,
        )

        if results:
            r = results[0]
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy()
                confs = boxes.conf.cpu().numpy()

                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i].tolist()
                    rows.append({
                        "frame": frame_idx,
                        "track_id": -1,
                        "class_id": int(cls[i]),
                        "confidence": float(confs[i]),
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "cx": float((x1 + x2) / 2.0),
                        "cy": float((y1 + y2) / 2.0),
                        "w": float(x2 - x1),
                        "h": float(y2 - y1),
                    })

        if frame_idx % 10 == 0:
            elapsed = time.time() - t0
            fps = frame_idx / elapsed if elapsed > 0 else 0.0
            print(f"[YOLO-CPU] frame={frame_idx} rows={len(rows)} fps={fps:.2f}")

    cap.release()

    df = pd.DataFrame(rows)
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)

    det_csv_path = str(Path(output_csv_path).with_name(Path(output_csv_path).stem + "_detections.csv"))
    df.to_csv(det_csv_path, index=False)
    print(f"[YOLO-CPU] detections_csv={det_csv_path} rows={len(df)}")

    return {
        "detections_csv": det_csv_path,
        "detections_csv": output_csv_path,
        "pet_events": [],
        "total_frames": frame_idx,
        "total_detections": len(df),
    }
