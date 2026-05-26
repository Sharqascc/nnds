from pathlib import Path

repo = Path(__file__).resolve().parents[1]
readme = repo / "README.md"

start = "<!-- AUTO-README-START -->"
end = "<!-- AUTO-README-END -->"

generated = """## Auto-generated structure

- `experimental/contact_point_pipeline.py` — contact-point projection into world coordinates.
- `traffic_analyzer.py` — main video-to-PET entry point.
- `analysis/research_run.py` — orchestrated research workflow.
"""

text = readme.read_text(encoding="utf-8")

block = f"{start}\n{generated}\n{end}"

if start in text and end in text:
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    new_text = before.rstrip() + "\n\n" + block + after
else:
    new_text = text.rstrip() + "\n\n" + block + "\n"

readme.write_text(new_text, encoding="utf-8")
print("README updated")
