"""
Scan the repo for anything that could tell us about the camera/rig used to
produce the calibration frames: existing intrinsics/calibration files,
camera model/spec mentions in configs or docs, video source metadata,
and any leftover calibration matrices.

This does not require camera access -- it just checks what the repo
already knows, so we can decide whether checkerboard calibration is
even a live option before assuming either way.
"""
import json
import re
from pathlib import Path

ROOT = Path("/content/nnds_verify")

KEYWORDS = [
    "focal", "intrinsic", "camera_matrix", "cameraMatrix", "fx", "fy",
    "distortion", "checkerboard", "calibrat", "sensor_width", "sensor width",
    "lens", "fov", "field of view", "rtsp", "youtube", "stream_url",
    "video_source", "source_url", "camera_model", "make:", "model:",
]

TEXT_EXTS = {".py", ".json", ".yaml", ".yml", ".md", ".txt", ".cfg", ".ini", ".toml"}


def find_text_files(root):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_EXTS]


def scan_file_for_keywords(path):
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return []
    hits = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        low = line.lower()
        for kw in KEYWORDS:
            if kw.lower() in low:
                hits.append((line_no, kw, line.strip()[:160]))
                break
    return hits


def find_calibration_artifacts(root):
    candidates = []
    for p in root.rglob("*.npy"):
        candidates.append(p)
    for p in root.rglob("*.json"):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        text = json.dumps(data).lower()
        if any(k in text for k in ("camera_matrix", "intrinsic", "focal", "fx", "fy")):
            candidates.append(p)
    return sorted(set(candidates))


def main():
    print("=" * 80)
    print(f"Scanning {ROOT} for camera / calibration / source metadata")
    print("=" * 80)

    text_files = find_text_files(ROOT)
    print(f"\nScanning {len(text_files)} text-like files for keyword hits...\n")

    any_hits = False
    for p in text_files:
        hits = scan_file_for_keywords(p)
        if hits:
            any_hits = True
            rel = p.relative_to(ROOT)
            print(f"--- {rel} ---")
            for line_no, kw, line in hits[:10]:
                print(f"  L{line_no} [{kw}]: {line}")
            if len(hits) > 10:
                print(f"  ... and {len(hits) - 10} more matches")
            print()

    if not any_hits:
        print("No keyword hits found in any text/config/doc file.\n")

    print("=" * 80)
    print("Checking for existing calibration artifacts (.npy, camera-matrix-like .json)")
    print("=" * 80)
    artifacts = find_calibration_artifacts(ROOT)
    if not artifacts:
        print("None found beyond the known homography .npy/.json files.")
    else:
        for a in artifacts:
            print(f"  {a.relative_to(ROOT)}")

    print("\n" + "=" * 80)
    print("Checking image filenames/paths for camera or source hints")
    print("=" * 80)
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic")
    images = [p for p in ROOT.rglob("*") if p.suffix.lower() in exts]
    for p in images:
        print(f"  {p.relative_to(ROOT)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
