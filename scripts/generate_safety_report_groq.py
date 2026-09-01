#!/usr/bin/env python3
"""
Generate a structured safety report using Groq LLM (optional).

Requires GROQ_API_KEY environment variable.
Uses only the deterministic metrics from the pipeline; does NOT alter core results.

Usage:
    GROQ_API_KEY=xxx python scripts/generate_safety_report_groq.py --pet-csv outputs/e2e_validation_pet.csv --output outputs/safety_report_groq.md
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd


def build_metrics(pet_csv):
    pet = pd.read_csv(pet_csv)
    metrics = {
        "total_events": len(pet),
        "median_pet": float(pet['pet'].median()) if not pet.empty else None,
        "mean_pet": float(pet['pet'].mean()) if not pet.empty else None,
        "std_pet": float(pet['pet'].std()) if not pet.empty else None,
        "conflict_types": pet['conflict_type'].value_counts().to_dict() if 'conflict_type' in pet.columns else {},
        "grid_cells": pet['grid_cell'].value_counts().to_dict() if 'grid_cell' in pet.columns else {},
    }
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pet-csv", required=True)
    parser.add_argument("--output", default="outputs/safety_report_groq.md")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not set. Exiting without generating report.")
        return

    from groq import Groq
    client = Groq(api_key=api_key)

    metrics = build_metrics(args.pet_csv)
    prompt = f"""You are a traffic safety expert. Given these quantitative metrics:
{json.dumps(metrics, indent=2)}

Write a 3-paragraph objective report describing the findings, including implications for intersection safety. Do not invent data or make claims beyond the supplied numbers. Clearly separate results, interpretation, and limitations."""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1000,
    )
    report = response.choices[0].message.content

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"✅ Saved Groq report to {out_path}")

if __name__ == "__main__":
    main()
