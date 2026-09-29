# Calibration

This folder contains the calibration workflow for the NNDS repository.

## Overview

The calibration package is organized for research reproducibility and publication review. It separates source inputs, derived outputs, reports, and executable scripts.

## Contents

| Path | Purpose |
|---|---|
| `raw/` | Source calibration inputs and immutable reference material. |
| `derived/` | Intermediate or computed calibration artifacts. |
| `reports/` | Final figures, tables, and publication-ready outputs. |
| `scripts/` | Canonical scripts used to reproduce calibration results. |
| `MANIFEST.json` | File inventory with sizes and SHA-256 hashes. |
| `PROVENANCE.md` | Notes on origin and transformation of calibration inputs. |
| `REPRODUCIBILITY.md` | Step-by-step instructions to regenerate calibration results. |

## Reproducibility

1. Start from a clean clone of the repository.
2. Place calibration inputs in `raw/`.
3. Run the scripts in `scripts/`.
4. Write outputs to `derived/` and final publication artifacts to `reports/`.
5. Regenerate `MANIFEST.json` after any file changes.

## Provenance

Calibration provenance is documented in `PROVENANCE.md`. For each artifact, record the source inputs, script version, parameters, and output path.

## Notes

- The scripts in `scripts/` are the canonical runnable versions.
- Regenerate the manifest after updating any calibration file.
- Keep raw inputs immutable once publication results are finalized.
