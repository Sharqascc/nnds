from pathlib import Path

p = Path("/content/nnds/results/README.md")
p.parent.mkdir(parents=True, exist_ok=True)

p.write_text("""# Results archive

This directory stores versioned, reproducible result artifacts that are important to keep in Git.

## Layout

- `YYYY-MM-DD/` — dated snapshot of outputs worth preserving.
- `outputs/` remains for runtime scratch and regenerated files; many files there may be ignored by Git.

## Current entries

- `2026-06-20/safety_eval_diffusion.csv` — archived diffusion safety-evaluation output generated after fixing model loading and future-horizon sampling.

## Suggested practice

- Put only compact, decision-relevant artifacts here.
- Keep large binaries, checkpoints, and temporary files out of this directory.
- Add a short note in each dated folder when the artifact corresponds to a specific code fix or experiment.
""", encoding="utf-8")

print("Wrote", p)
print()
print(p.read_text(encoding="utf-8"))
