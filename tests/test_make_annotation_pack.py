import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def _make_video(path: Path, n_frames: int = 20, w: int = 64, h: int = 48) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), i * 8 % 255, dtype=np.uint8)
        vw.write(frame)
    vw.release()


def _mk_pred_csvs(d: Path) -> tuple[Path, Path, Path, Path]:
    det_rows = []
    for f in range(20):
        det_rows.append(
            {"frame": f, "x1": 5, "y1": 5, "x2": 20, "y2": 20, "class_name": "car", "conf": 0.9}
        )
    det = d / "pred_detection.csv"
    pd.DataFrame(det_rows).to_csv(det, index=False)

    trk_rows = [
        {"frame": f, "track_id": 1, "x": 5.0, "y": 5.0, "w": 15.0, "h": 15.0} for f in range(20)
    ]
    trk = d / "pred_tracking.csv"
    pd.DataFrame(trk_rows).to_csv(trk, index=False)

    traj_rows = [{"frame": r["frame"], "track_id": 1, "x": 1.0, "y": 2.0} for r in det_rows]
    traj = d / "pred_trajectory.csv"
    pd.DataFrame(traj_rows).to_csv(traj, index=False)

    ssm_rows = [{"track_a": 1, "track_b": 2, "pet": 0.5}, {"track_a": 1, "track_b": 3, "pet": 1.2}]
    ssm = d / "pred_ssm.csv"
    pd.DataFrame(ssm_rows).to_csv(ssm, index=False)

    return det, trk, traj, ssm


def test_annotation_pack_runs(tmp_path):
    video = tmp_path / "v.mp4"
    _make_video(video, n_frames=20)
    det, trk, traj, ssm = _mk_pred_csvs(tmp_path)

    out_dir = tmp_path / "pack"
    r = subprocess.run(
        [
            sys.executable,
            "scripts/make_annotation_pack.py",
            "--video",
            str(video),
            "--pred-det",
            str(det),
            "--pred-trk",
            str(trk),
            "--pred-traj",
            str(traj),
            "--pred-ssm",
            str(ssm),
            "--out-dir",
            str(out_dir),
            "--max-frames",
            "20",
            "--sample-every",
            "5",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr

    # frames extracted: 0,5,10,15 -> 4 PNGs
    frames = sorted((out_dir / "frames").glob("*.png"))
    assert len(frames) == 4

    # templates exist and are empty where expected
    det_t = pd.read_csv(out_dir / "gt_detection_template.csv")
    assert len(det_t) == 4
    assert (det_t["class_name"].fillna("") == "").all()

    trk_t = pd.read_csv(out_dir / "gt_tracking_template.csv")
    assert len(trk_t) == 4
    assert (trk_t["track_id"] == -1).all()

    traj_t = pd.read_csv(out_dir / "gt_trajectory_template.csv")
    assert len(traj_t) == 4

    ssm_t = pd.read_csv(out_dir / "gt_ssm_template.csv")
    assert len(ssm_t) == 2
    assert (ssm_t["verdict"].fillna("") == "").all()
    assert (ssm_t["actual_pet"].fillna("") == "").all()

    readme = (out_dir / "README.md").read_text()
    assert "GITI annotation pack" in readme
