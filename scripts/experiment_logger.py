#!/usr/bin/env python3
"""
Generate a JSON experiment log from pipeline outputs.

Usage:
    python scripts/experiment_logger.py --detections outputs/petevents_detections.csv --pet outputs/petevents.csv --output outputs/experiment_log.json
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", default="outputs/petevents_bev_detections.csv")
    parser.add_argument("--pet", default="outputs/petevents_bev.csv")
    parser.add_argument("--output", default="outputs/experiment_log.json")
    args = parser.parse_args()

    det_path = Path(args.detections)
    pet_path = Path(args.pet)
    log = {
        "timestamp": time.time(),
        "detections": None,
        "pet": None,
    }

    if det_path.exists():
        det = pd.read_csv(det_path)
        log["detections"] = {
            "file": str(det_path),
            "rows": len(det),
            "columns": list(det.columns),
            "unique_tracks": int(det["track_id"].nunique()) if "track_id" in det.columns else None,
            "mean_conf": float(det["conf"].mean()) if "conf" in det.columns else None,
        }
    else:
        log["detections"] = {"file": str(det_path), "exists": False}

    if pet_path.exists():
        pet = pd.read_csv(pet_path)
        log["pet"] = {
            "file": str(pet_path),
            "rows": len(pet),
            "columns": list(pet.columns),
            "mean_pet": float(pet["pet"].mean())
            if "pet" in pet.columns and not pet.empty
            else None,
            "median_pet": float(pet["pet"].median())
            if "pet" in pet.columns and not pet.empty
            else None,
        }
    else:
        log["pet"] = {"file": str(pet_path), "exists": False}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(log, indent=2))
    print(f"✅ Experiment log saved to {output_path}")


if __name__ == "__main__":
    main()
