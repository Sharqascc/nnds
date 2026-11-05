"""Generate the FP taxonomy review HTML from fp_overlay/ clips.

Run:
    python -m paper.analysis.generate_fp_taxonomy

Reads:  paper/annotations/fp_overlay/fp_*.mp4
Writes: paper/annotations/fp_taxonomy.html
"""

from __future__ import annotations

from pathlib import Path

ANN = Path(__file__).resolve().parents[1] / "annotations"
FP_DIR = ANN / "fp_overlay"

CATEGORIES = [
    "parallel_flow",
    "non_interacting_crossing",
    "tracking_artifact",
    "sub_resolution_noise",
    "other",
]


def _clip_rows(clips: list[Path]) -> str:
    rows = []
    for c in clips:
        item = c.stem
        opts = "".join(
            f'<label><input type="radio" name="{item}" value="{cat}"> {cat}</label>'
            for cat in CATEGORIES
        )
        rows.append(
            f'<div class="item"><div class="meta"><strong>{item}</strong></div>'
            f'<video src="fp_overlay/{item}.mp4" autoplay loop muted playsinline '
            f'style="width:520px;display:block"></video>'
            f'<div class="judge">{opts}</div>'
            f'<input type="text" name="{item}_notes" placeholder="notes"></div>'
        )
    return "\n".join(rows)


HTML_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<title>FP taxonomy</title>
<style>
body {{ font-family:-apple-system,sans-serif; margin:20px; background:#111; color:#eee; }}
.item {{ border:1px solid #333; padding:10px; margin:10px 0; border-radius:8px; }}
.meta {{ font-size:15px; margin-bottom:6px; }}
label {{ display:inline-block; margin-right:14px; font-size:13px; }}
input[type=text] {{ margin-left:8px; width:300px; background:#222; color:#eee;
                    border:1px solid #444; padding:4px; }}
#dl {{ position:fixed; bottom:20px; right:20px; background:#2c7; color:#000;
       border:none; padding:12px 20px; font-size:16px; cursor:pointer;
       border-radius:6px; font-weight:bold; z-index:99; }}
#ctr {{ position:fixed; bottom:20px; left:20px; background:#333; color:#eee;
        padding:12px 20px; border-radius:6px; font-weight:bold; z-index:99; }}
</style></head><body>
<h2>FP failure taxonomy — {n} clips</h2>
<p>Choose the dominant failure mode per clip: parallel_flow,
non_interacting_crossing, tracking_artifact, sub_resolution_noise, other.</p>
{rows}
<div id="ctr">0 / {n}</div>
<button id="dl" onclick="saveCSV()">Download CSV</button>
<script>
function up() {{
  const items = document.querySelectorAll('.item');
  let n = 0;
  items.forEach(it => {{
    const id = it.querySelector('strong').textContent.trim();
    if (document.querySelector(`input[name="${{id}}"]:checked`)) n++;
  }});
  document.getElementById('ctr').textContent = `${{n}} / ${{items.length}}`;
}}
document.addEventListener('change', up);
function saveCSV() {{
  const items = document.querySelectorAll('.item');
  let csv = 'item_id,category,notes\n';
  items.forEach(it => {{
    const id = it.querySelector('strong').textContent.trim();
    const v = document.querySelector(`input[name="${{id}}"]:checked`);
    const nt = it.querySelector(`input[name="${{id}}_notes"]`).value.replace(/,/g,';');
    csv += `${{id}},${{v?v.value:''}},${{nt}}\n`;
  }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv], {{type:'text/csv'}}));
  a.download = 'fp_taxonomy.csv';
  a.click();
}}
</script></body></html>"""


def main() -> Path:
    clips = sorted(FP_DIR.glob("fp_*.mp4"))
    html = HTML_TEMPLATE.format(rows=_clip_rows(clips), n=len(clips))
    out = ANN / "fp_taxonomy.html"
    out.write_text(html)
    return out


if __name__ == "__main__":
    p = main()
    print(f"wrote {p}  ({p.stat().st_size / 1024:.1f} KB)")
