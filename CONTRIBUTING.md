# Contributing

## Code organization

- Put reusable diffusion logic inside `traffic_diffusion/`.
- Put reusable grid and PET logic inside `Grid_&_trajectory/`.
- Put experiment or evaluation runners inside `analysis/`.
- Treat `code/Grid_&_trajectory/` as legacy unless you are explicitly migrating code.

## Outputs

- Write generated CSV, plots, and artifacts into `outputs/`.
- Use stable, descriptive filenames so downstream scripts can reference them reliably.

## Research workflow

1. Generate PET events from the traffic analysis pipeline.
2. Train or evaluate the diffusion model on those events.
3. Save reproducible artifacts in `outputs/`.
4. Update `README.md` when adding new canonical entry points or workflows.

## Commit style

Use clear, scoped commit messages, for example:

- `Add diffusion sampling utility`
- `Document notebook safety pipeline`
- `Refactor PET event preprocessing`
