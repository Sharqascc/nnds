#!/usr/bin/env python3
from pathlib import Path
import ast
import importlib.util
import json
import traceback
import sys

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "audit" / "audit_config.json").read_text())

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
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}

def check_contract(pyfile: Path, expected):
    try:
        mod = load_module(pyfile)
        missing = [name for name in expected if not hasattr(mod, name)]
        return {"ok": not missing, "missing": missing}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

def main():
    report = {"root": str(ROOT), "dirs": {}, "modules": {}, "syntax": {}, "imports": {}, "contracts": {}}

    for d in CONFIG.get("expected_dirs", []):
        p = ROOT / d
        report["dirs"][d] = p.exists()

    for rel in CONFIG.get("modules", []):
        p = ROOT / rel
        exists = p.exists()
        report["modules"][rel] = exists
        if exists:
            report["syntax"][rel] = check_syntax(p)
            report["imports"][rel] = check_import(p)
            expected = CONFIG.get("contracts", {}).get(rel, {}).get("must_have", [])
            report["contracts"][rel] = check_contract(p, expected)

    out = ROOT / "outputs" / "repo_audit_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    failed = (
        any(not v for v in report["dirs"].values())
        or any(not v for v in report["modules"].values())
        or any(not v["ok"] for v in report["syntax"].values())
        or any(not v["ok"] for v in report["imports"].values())
        or any(not v["ok"] for v in report["contracts"].values())
    )
    raise SystemExit(1 if failed else 0)

if __name__ == "__main__":
    main()
