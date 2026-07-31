#!/usr/bin/env python3
from pathlib import Path
import ast
import importlib.util
import json
import traceback
import sys

ROOT = Path(__file__).resolve().parent

EXPECTED_MODULES = [
    "analysis/research_run.py",
    "gate_counter.py",
    "pet_conflict_checker.py",
    "traffic_analyzer.py",
    "bev_mapper.py",
]

EXPECTED_DIRS = [
    "analysis",
    "tests",
    "configs",
    "core",
    "traffic_diffusion",
]

def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def check_syntax(pyfile: Path):
    try:
        ast.parse(pyfile.read_text(encoding="utf-8"))
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

def check_import(pyfile: Path):
    try:
        load_module(pyfile)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

def main():
    report = {
        "root": str(ROOT),
        "dirs": {},
        "modules": {},
        "syntax": {},
        "imports": {},
    }

    for d in EXPECTED_DIRS:
        p = ROOT / d
        report["dirs"][d] = p.exists()

    for rel in EXPECTED_MODULES:
        p = ROOT / rel
        exists = p.exists()
        report["modules"][rel] = exists
        if exists:
            report["syntax"][rel] = check_syntax(p)
            report["imports"][rel] = check_import(p)

    print(json.dumps(report, indent=2))
    failed = any(not v for v in report["dirs"].values()) or any(not v for v in report["modules"].values())
    failed = failed or any(not v["ok"] for v in report["syntax"].values()) or any(not v["ok"] for v in report["imports"].values())
    raise SystemExit(1 if failed else 0)

if __name__ == "__main__":
    main()
