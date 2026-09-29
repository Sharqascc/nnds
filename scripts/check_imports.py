#!/usr/bin/env python3
"""Import-layering guardrail.

Enforces two rules:
  1. Canonical pipeline modules (pipeline, grid_trajectory, bev, top-level
     analysis) must not import from experimental modules
     (src.diffusion, src.vlm).
  2. Leaf layers (src.core, src.utils) must not import anything else in src/.

Scans both static imports (AST) and dynamic imports (__import__ /
importlib.import_module via regex). Exit 0 if clean, 1 on violation.

Pattern syntax for applies_to / allowlist:
  **/  = zero or more path segments (so src/bev/**/*.py matches both
         src/bev/bev_mapper.py and src/bev/calibration/foo.py)
  *    = any run of non-slash characters
  ?    = a single non-slash character
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

RULES = [
    {
        "name": "canonical-pipeline must not import experimental modules",
        "applies_to": [
            "src/pipeline/**/*.py",
            "src/analysis/grid_trajectory/**/*.py",
            "src/analysis/*.py",
            "src/bev/**/*.py",
        ],
        "forbids": ["src.diffusion", "src.vlm"],
        "allowlist": [
            "src/analysis/safety_eval_diffusion.py",
            "src/analysis/safety_eval_diffusion_notebook.py",
            "src/analysis/pet_diffusion_analysis.py",
        ],
    },
    {
        "name": "leaf layers must not import higher layers",
        "applies_to": [
            "src/core/**/*.py",
            "src/utils/**/*.py",
        ],
        "forbids": [
            "src.analysis",
            "src.bev",
            "src.diffusion",
            "src.pipeline",
            "src.vlm",
        ],
        "allowlist": [],
    },
]


def _norm(p) -> str:
    return str(p).replace(chr(92), "/")


_PATTERN_CACHE: dict[str, re.Pattern] = {}


def _pattern_to_regex(pattern: str) -> re.Pattern:
    """Translate a glob to a regex where **/ means zero-or-more path segments.

    fnmatch treats **/ as 'one or more', which silently misses top-level
    files in a directory (src/bev/**/*.py would not match
    src/bev/bev_mapper.py). The translation below treats **/ as
    '(?:[^/]+/)*' so both shallow and nested files match.
    """
    cached = _PATTERN_CACHE.get(pattern)
    if cached is not None:
        return cached
    out: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern[i : i + 3] == "**/":
            out.append(r"(?:[^/]+/)*")
            i += 3
        elif pattern[i : i + 2] == "**":
            out.append(r".*")
            i += 2
        elif pattern[i] == "*":
            out.append(r"[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append(r"[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    rx = re.compile("^" + "".join(out) + "$")
    _PATTERN_CACHE[pattern] = rx
    return rx


def _matches(path: str, pattern: str) -> bool:
    return _pattern_to_regex(pattern).match(path) is not None


def _in_module(text: str, mod: str) -> bool:
    return text == mod or text.startswith(mod + ".")


def _static_imports(tree: ast.AST) -> list[str]:
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                out.append(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                out.append(node.module)
    return out


_DYN_IMPORT_RE = re.compile(
    r"__import__\s*\(\s*['\"]([\w.]+)['\"]"
    r"|importlib\.import_module\s*\(\s*['\"]([\w.]+)['\"]"
)


def _dynamic_imports(src: str) -> list[str]:
    out: list[str] = []
    for m in _DYN_IMPORT_RE.finditer(src):
        out.append(m.group(1) or m.group(2))
    return out


def _relative(path: Path, root: Path) -> str:
    try:
        return _norm(path.resolve().relative_to(root.resolve()))
    except ValueError:
        try:
            return _norm(path.relative_to(root))
        except ValueError:
            return _norm(path)


def check_file(path: Path, root: Path) -> list[tuple[str, str, str]]:
    rel = _relative(path, root)
    try:
        src = path.read_text()
        tree = ast.parse(src)
    except (SyntaxError, OSError):
        return []

    imports = set(_static_imports(tree)) | set(_dynamic_imports(src))
    hits: list[tuple[str, str, str]] = []
    for rule in RULES:
        if not any(_matches(rel, pat) for pat in rule["applies_to"]):
            continue
        if any(_matches(rel, pat) for pat in rule["allowlist"]):
            continue
        for mod in imports:
            for bad in rule["forbids"]:
                if _in_module(mod, bad):
                    hits.append((rule["name"], mod, rel))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="Files; if empty, walks src/")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    root = Path.cwd()
    if args.paths:
        targets = [Path(p) for p in args.paths if Path(p).suffix == ".py"]
    else:
        targets = sorted(Path("src").rglob("*.py"))

    violations: list[tuple[str, str, str]] = []
    for t in targets:
        violations.extend(check_file(t, root))

    if not violations:
        if not args.quiet:
            print(f"import guardrail: clean ({len(targets)} files)")
        return 0

    print(f"import guardrail: {len(violations)} violation(s):", file=sys.stderr)
    for rule, mod, rel in violations:
        print(f"  {rel}: imports {mod!r} (rule: {rule})", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
