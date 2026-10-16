"""Blind recall sampling — 40 windows NOT selected by the algorithm.

Run:  python -m paper.analysis.blind_windows

Reads:  outputs/giti_screened_with_gates.csv
        data/sample_data/GITI_traffic_video.mp4
Writes: paper/annotations/blind_windows/blind_*.mp4
        paper/annotations/blind_windows/manifest.json
        paper/annotations/blind_review.html

Purpose: measure recall. Recall cannot be estimated from events the
algorithm flagged; it requires sampling windows independently and having
a human identify real conflicts in those windows. This module samples
40 two-second windows from the region where the pipeline flagged nothing
and builds a review UI.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SCREENED = REPO / "outputs" / "giti_screened_with_gates.csv"
VIDEO = REPO / "data" / "sample_data" / "GITI_traffic_video.mp4"
OUT_DIR = REPO / "paper" / "annotations" / "blind_windows"
HTML_OUT = REPO / "paper" / "annotations" / "blind_review.html"

FPS = 30.0
TOTAL_FRAMES = 1837
WINDOW_FRAMES = 60
N_WINDOWS = 40
EVENT_PAD_FRAMES = 30
SEED = 42


def _forbidden_frames(screened: pd.DataFrame) -> set:
    forbidden: set = set()
    for _, r in screened.iterrows():
        lo = int(min(r.track_a_entry_frame, r.track_b_entry_frame)) - EVENT_PAD_FRAMES
        hi = int(max(r.track_a_exit_frame, r.track_b_exit_frame)) + EVENT_PAD_FRAMES
        lo = max(0, lo)
        hi = min(TOTAL_FRAMES - 1, hi)
        forbidden.update(range(lo, hi + 1))
    return forbidden


def _sample_windows(forbidden: set) -> list:
    rng = np.random.default_rng(SEED)
    allowed = [
        f
        for f in range(0, TOTAL_FRAMES - WINDOW_FRAMES)
        if f not in forbidden and (f + WINDOW_FRAMES) not in forbidden
    ]
    if len(allowed) < N_WINDOWS:
        raise RuntimeError(f"not enough allowed frames: {len(allowed)}")
    rng.shuffle(allowed)
    taken: set = set()
    starts: list = []
    for s in allowed:
        if len(starts) >= N_WINDOWS:
            break
        window = set(range(s, s + WINDOW_FRAMES))
        if window & taken:
            continue
        starts.append(int(s))
        taken |= window
    if len(starts) < N_WINDOWS:
        extra = rng.choice(
            [a for a in allowed if a not in starts], size=N_WINDOWS - len(starts), replace=False
        )
        starts.extend(int(x) for x in extra)
    return sorted(starts)


def _build_clip(start: int, idx: int) -> Path:
    dur = WINDOW_FRAMES / FPS
    out = OUT_DIR / f"blind_{idx:03d}.mp4"
    subprocess.run(
        f"ffmpeg -y -loglevel error -ss {start / FPS:.3f} -i {VIDEO} "
        f"-t {dur:.3f} -c:v libx264 -preset veryfast -crf 23 -an {out}",
        shell=True,  # nosec B602 - ffmpeg args are constructed from constants
        check=False,
    )
    return out


HTML_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<title>Blind recall review</title>
<style>
body {{ font-family:-apple-system,sans-serif; margin:20px; background:#111; color:#eee; }}
.item {{ border:1px solid #333; padding:12px; margin:12px 0; border-radius:8px; }}
video {{ width:520px; display:block; }}
label {{ margin-right:14px; }}
input[type=number] {{ width:80px; margin-left:8px; }}
input[type=text] {{ margin-left:8px; width:300px; background:#222; color:#eee;
                    border:1px solid #444; padding:4px; }}
#dl {{ position:fixed; bottom:20px; right:20px; background:#2c7; color:#000;
       border:none; padding:12px 20px; font-size:16px; cursor:pointer;
       border-radius:6px; font-weight:bold; z-index:99; }}
#ctr {{ position:fixed; bottom:20px; left:20px; background:#333; color:#eee;
        padding:12px 20px; border-radius:6px; font-weight:bold; z-index:99; }}
</style></head><body>
<h2>Blind recall review</h2>
<p>Watch each clip. Did two distinct road users pass through the same zone
within 3 seconds of each other? If yes, estimate the PET in seconds.
These are random windows sampled independently of algorithm output.</p>
{rows}
<div id="ctr">0 / {n}</div>
<button id="dl" onclick="saveCSV()">Download CSV</button>
<script>
function up() {{
  const items = document.querySelectorAll('.item');
  let n = 0;
  items.forEach(it => {{
    const id = it.dataset.id;
    if (document.querySelector(`input[name="${{id}}"]:checked`)) n++;
  }});
  document.getElementById('ctr').textContent = `${{n}} / ${{items.length}}`;
}}
document.addEventListener('change', up);
function saveCSV() {{
  const items = document.querySelectorAll('.item');
  let csv = 'clip_id,start_frame,conflict,petyp_s,notes\n';
  items.forEach(it => {{
    const id = it.dataset.id;
    const start = it.dataset.start;
    const v = document.querySelector(`input[name="${{id}}"]:checked`);
    const pet = it.querySelector(`input[name="${{id}}_pet"]`).value;
    const nt = it.querySelector(`input[name="${{id}}_notes"]`).value.replace(/,/g,';');
    csv += `${{id}},${{start}},${{v?v.value:''}},${{pet}},${{nt}}\n`;
  }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv], {{type:'text/csv'}}));
  a.download = 'blind_recall.csv';
  a.click();
}}
</script></body></html>"""


def _make_html(rows: list) -> str:
    parts = []
    for r in rows:
        cid = r["clip_id"]
        parts.append(
            f'<div class="item" data-id="{cid}" data-start="{r["start_frame"]}">'
            f"<div><b>{cid}</b>  (start frame {r['start_frame']})</div>"
            f'<video src="blind_windows/{cid}.mp4" autoplay loop muted playsinline></video>'
            f"<div>"
            f'<label><input type="radio" name="{cid}" value="yes"> yes</label>'
            f'<label><input type="radio" name="{cid}" value="no"> no</label>'
            f'<label><input type="radio" name="{cid}" value="unsure"> unsure</label>'
            f'PET (s): <input type="number" name="{cid}_pet" min="0" max="5" step="0.1">'
            f"</div>"
            f'<div>notes: <input type="text" name="{cid}_notes" placeholder="optional"></div>'
            f"</div>"
        )
    return HTML_TEMPLATE.format(rows="\n".join(parts), n=len(rows))


def main() -> Path:
    if not VIDEO.exists():
        raise FileNotFoundError(f"video missing: {VIDEO}")
    screened = pd.read_csv(SCREENED)
    forbidden = _forbidden_frames(screened)
    starts = _sample_windows(forbidden)
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, s in enumerate(starts):
        clip = _build_clip(s, i)
        if clip.exists():
            rows.append({"clip_id": f"blind_{i:03d}", "start_frame": s})
    HTML_OUT.write_text(_make_html(rows))
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "n_windows": len(rows),
                "window_frames": WINDOW_FRAMES,
                "fps": FPS,
                "seed": SEED,
                "event_pad_frames": EVENT_PAD_FRAMES,
                "clips": rows,
            },
            indent=2,
        )
    )
    return HTML_OUT


if __name__ == "__main__":
    p = main()
    print(f"wrote {p}  ({p.stat().st_size} bytes)")
