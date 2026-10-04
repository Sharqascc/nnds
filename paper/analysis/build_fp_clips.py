"""Build overlay MP4 clips for the 73 false positives.

Run:  python -m paper.analysis.build_fp_clips

Reads:  paper/annotations/fp_overlay/ (must be empty or missing)
        data/reviews/ssm_review_114/fp_list.csv
        outputs/giti_raw.csv
        outputs/giti_screened_with_gates.csv
        data/sample_data/GITI_traffic_video.mp4
Writes: paper/annotations/fp_overlay/fp_*.mp4
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
ANN = REPO / "paper" / "annotations" / "fp_overlay"
WORK = Path("/tmp/fp_build")
FP_LIST = REPO / "data" / "reviews" / "ssm_review_114" / "fp_list.csv"
RAW_CSV = REPO / "outputs" / "giti_raw.csv"
SCREENED = REPO / "outputs" / "giti_screened_with_gates.csv"
VIDEO = REPO / "data" / "sample_data" / "GITI_traffic_video.mp4"
BEV_CFG = REPO / "configs" / "bev_config.json"

FPS_VIDEO = 30.0
TOTAL_FRAMES = 1837
PAD = 20
SCALE = 520 / 1600
R_WORLD = 1.5


def _main() -> Path:
    bev = json.loads(BEV_CFG.read_text())
    H = np.array(bev["H_pixel_to_world"])
    A_X, C_X = H[0, 0], H[0, 2]
    A_Y, C_Y = H[1, 1], H[1, 2]

    def px_to_world(px: float, py: float) -> tuple[float, float]:
        return A_X * px + C_X, A_Y * py + C_Y

    def world_to_px(wx: float, wy: float) -> tuple[float, float]:
        return (wx - C_X) / A_X, (wy - C_Y) / A_Y

    raw = pd.read_csv(RAW_CSV)
    track_trajs: dict[int, dict[int, tuple[float, float]]] = {}
    for _, r in raw.iterrows():
        for tc, jc in (("track_a", "traj_a_json"), ("track_b", "traj_b_json")):
            t, js = r.get(tc), r.get(jc)
            if pd.isna(t) or pd.isna(js):
                continue
            try:
                pts = json.loads(js)
            except json.JSONDecodeError:
                continue
            d = track_trajs.setdefault(int(t), {})
            for p in pts:
                d[int(p["frame"])] = (float(p["x_pixel"]), float(p["y_pixel"]))

    def conflict_point(tA, tB):
        best_d, best_xy = float("inf"), None
        for _, (xa, ya) in tA.items():
            wa = px_to_world(xa, ya)
            for _, (xb, yb) in tB.items():
                wb = px_to_world(xb, yb)
                d = float(np.hypot(wa[0] - wb[0], wa[1] - wb[1]))
                if d < best_d:
                    best_d, best_xy = d, ((wa[0] + wb[0]) / 2, (wa[1] + wb[1]) / 2)
        return (best_xy, best_d) if best_xy else None

    def nearest_pt(td, f, win=15):
        for off in range(win + 1):
            for ff in (f - off, f + off):
                if ff in td:
                    return td[ff]
        return None

    fps_df = pd.read_csv(FP_LIST)
    giti = pd.read_csv(SCREENED)

    shutil.rmtree(ANN, ignore_errors=True)
    ANN.mkdir(parents=True)
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir()

    built = 0
    for _, row in fps_df.iterrows():
        item = f"fp_{int(row.idx):03d}"
        g = giti.iloc[int(row.giti_idx)]
        ta, tb = int(row.track_a), int(row.track_b)
        tA, tB = track_trajs.get(ta, {}), track_trajs.get(tb, {})
        if not tA or not tB:
            continue
        tmin = max(0, int(min(g.track_a_entry_frame, g.track_b_entry_frame)) - PAD)
        tmax = min(TOTAL_FRAMES - 1, int(max(g.track_a_exit_frame, g.track_b_exit_frame)) + PAD)
        dur = (tmax - tmin + 1) / FPS_VIDEO
        clip = WORK / f"{item}.mp4"
        subprocess.run(
            f"ffmpeg -y -loglevel error -ss {tmin / FPS_VIDEO:.3f} -i {VIDEO} "
            f"-t {dur:.3f} -c:v libx264 -preset veryfast -crf 23 -an {clip}",
            shell=True,
            check=False,
        )
        if not clip.exists():
            continue
        cp = conflict_point(tA, tB)
        if cp is None:
            continue
        (wx_c, wy_c), dmin = cp
        px_c, py_c = world_to_px(wx_c, wy_c)
        r_px = max(12, int(R_WORLD / A_X * SCALE))

        fd = WORK / item
        fd.mkdir(exist_ok=True)
        subprocess.run(
            f"ffmpeg -y -loglevel error -i {clip} -vf scale=520:-1,fps=10 {fd}/f_%04d.png",
            shell=True,
        )
        fs = sorted(fd.glob("f_*.png"))
        if not fs:
            shutil.rmtree(fd, ignore_errors=True)
            continue
        out_dir = fd / "out"
        out_dir.mkdir(exist_ok=True)
        for j, fp in enumerate(fs):
            src_f = tmin + j * 3
            im = Image.open(fp).convert("RGB")
            dr = ImageDraw.Draw(im, "RGBA")
            cx, cy = px_c * SCALE, py_c * SCALE
            dr.ellipse(
                [cx - r_px, cy - r_px, cx + r_px, cy + r_px],
                fill=(255, 220, 0, 80),
                outline=(255, 170, 0, 255),
                width=3,
            )
            for t, col in ((tA, (0, 120, 255)), (tB, (255, 40, 40))):
                pt = nearest_pt(t, src_f)
                if pt:
                    x, y = pt[0] * SCALE, pt[1] * SCALE
                    dr.ellipse([x - 12, y - 12, x + 12, y + 12], outline=(*col, 255), width=4)
            dr.rectangle([0, 0, 210, 24], fill=(0, 0, 0, 180))
            dr.text((8, 6), f"dmin={dmin:.2f}m  pet={row.pet:.2f}s", fill=(255, 255, 255, 255))
            im.save(out_dir / f"o_{j:04d}.png")
        mp4 = ANN / f"{item}.mp4"
        subprocess.run(
            f"ffmpeg -y -loglevel error -framerate 10 -i {out_dir}/o_%04d.png "
            f"-c:v libx264 -preset veryfast -crf 26 -pix_fmt yuv420p {mp4}",
            shell=True,
            check=False,
        )
        shutil.rmtree(fd, ignore_errors=True)
        built += 1
    return ANN


def main() -> Path:
    p = _main()
    n = len(list(p.glob("fp_*.mp4")))
    print(f"wrote {n} clips to {p}")
    return p


if __name__ == "__main__":
    main()
