#!/usr/bin/env python3
"""Generate an annotation kit from a pipeline run.

Extracts sample frames as PNGs and writes editable CSV templates so a human
annotator can produce ground truth for the metric scripts.

The templates are BLANK on purpose: the annotator must fill in the correct
boxes and PET verdicts. Do not treat the pre-filled predictions as GT.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

SAMPLE_EVERY = 10  # extract every Nth frame


def _extract_frames(video_path: Path, out_dir: Path, max_frames: int, every: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    written = 0
    frame_idx = 0
    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % every == 0:
            out_path = out_dir / f"frame_{frame_idx:05d}.png"
            cv2.imwrite(str(out_path), frame)
            written += 1
        frame_idx += 1
    cap.release()
    return written


def _write_detection_template(pred_det: Path, out_csv: Path, sample_frames: set[int]) -> int:
    df = pd.read_csv(pred_det)
    df = df[df["frame"].isin(sample_frames)].copy()
    df["class_name"] = ""  # annotator must fill
    df["conf"] = 0.0  # placeholder; not used for GT
    out = df[["frame", "x1", "y1", "x2", "y2", "class_name", "conf"]]
    out.to_csv(out_csv, index=False)
    return len(out)


def _write_tracking_template(pred_trk: Path, out_csv: Path, sample_frames: set[int]) -> int:
    df = pd.read_csv(pred_trk)
    df = df[df["frame"].isin(sample_frames)].copy()
    df["track_id"] = -1  # annotator assigns correct IDs
    out = df[["frame", "track_id", "x", "y", "w", "h"]]
    out.to_csv(out_csv, index=False)
    return len(out)


def _write_trajectory_template(pred_traj: Path, out_csv: Path, sample_frames: set[int]) -> int:
    df = pd.read_csv(pred_traj)
    df = df[df["frame"].isin(sample_frames)].copy()
    df["track_id"] = -1
    out = df[["frame", "track_id", "x", "y"]]
    out.to_csv(out_csv, index=False)
    return len(out)


def _write_ssm_template(pred_ssm: Path, out_csv: Path) -> int:
    df = pd.read_csv(pred_ssm)
    df["verdict"] = ""  # annotator: "real" or "false"
    df["actual_pet"] = ""  # measured PET if verdict == "real"
    df["notes"] = ""
    out = df[["track_a", "track_b", "pet", "verdict", "actual_pet", "notes"]]
    out.to_csv(out_csv, index=False)
    return len(out)


def _write_readme(out_dir: Path, video_name: str, n_frames: int, sample_every: int) -> None:
    lines = [
        "# GITI annotation pack",
        "",
        f"Source video: `{video_name}` (first {n_frames} frames).",
        "",
        "## Contents",
        "",
        f"- `frames/frame_XXXXX.png` - one PNG per sampled frame (every {sample_every}th frame).",
        "- `gt_detection_template.csv` - predicted boxes for those frames. "
        "**EDIT the `class_name` column; replace wrong boxes; delete false "
        "detections; add missed objects.**",
        "- `gt_tracking_template.csv` - same boxes with a `track_id` column. "
        "**Assign consistent IDs across frames.**",
        "- `gt_trajectory_template.csv` - world coordinates. Fill after "
        "tracking GT exists (can be projected from your corrected detection GT).",
        "- `gt_ssm_template.csv` - every predicted PET event. Watch the video "
        "and mark each: `verdict = real` or `false`. If `real`, record the "
        "PET you measure (seconds, t_b_entry - t_a_exit).",
        "",
        "## Workflow",
        "",
        "1. Open the video at the same frame numbers used in the CSVs.",
        "2. For each sampled frame, correct the boxes and fill `class_name`.",
        "3. Assign `track_id` in `gt_tracking_template.csv`: same ID for the "
        "same physical object across frames.",
        "4. Review each PET event in `gt_ssm_template.csv` against the video.",
        "5. Save your edited files as `gt_detection.csv`, `gt_tracking.csv`, "
        "`gt_trajectory.csv`, `gt_ssm.csv`.",
        "",
        "## DO NOT",
        "",
        "- Do not leave the prediction values in place and call it GT. That "
        "biases every metric toward the model.",
        "- Do not mark a PET event `real` without watching it in the video. "
        "Predicted events can be tracking artifacts.",
        "",
        "## Next step",
        "",
        "    python scripts/validate_gt.py --detection gt_detection.csv "
        "--tracking gt_tracking.csv --trajectory gt_trajectory.csv "
        "--ssm gt_ssm.csv",
        "",
        "Then run the four evaluate_* scripts per `docs/ANNOTATION_GUIDE.md`.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--pred-det", required=True)
    parser.add_argument("--pred-trk", required=True)
    parser.add_argument("--pred-traj", required=True)
    parser.add_argument("--pred-ssm", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--sample-every", type=int, default=SAMPLE_EVERY)
    args = parser.parse_args()

    out = Path(args.out_dir)
    frames_dir = out / "frames"
    out.mkdir(parents=True, exist_ok=True)

    n_written = _extract_frames(Path(args.video), frames_dir, args.max_frames, args.sample_every)
    sample_frames = set(range(0, args.max_frames, args.sample_every))

    n_det = _write_detection_template(
        Path(args.pred_det), out / "gt_detection_template.csv", sample_frames
    )
    n_trk = _write_tracking_template(
        Path(args.pred_trk), out / "gt_tracking_template.csv", sample_frames
    )
    n_traj = _write_trajectory_template(
        Path(args.pred_traj), out / "gt_trajectory_template.csv", sample_frames
    )
    n_ssm = _write_ssm_template(Path(args.pred_ssm), out / "gt_ssm_template.csv")

    _write_readme(out, Path(args.video).name, args.max_frames, args.sample_every)

    print(f"Wrote {n_written} frames to {frames_dir}")
    print(f"gt_detection_template.csv: {n_det} rows")
    print(f"gt_tracking_template.csv: {n_trk} rows")
    print(f"gt_trajectory_template.csv: {n_traj} rows")
    print(f"gt_ssm_template.csv: {n_ssm} rows")
    print(f"README.md: {out / 'README.md'}")


if __name__ == "__main__":
    main()
