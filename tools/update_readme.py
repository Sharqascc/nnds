#!/usr/bin/env python3
"""Auto-generate README.md sections from the NNDS codebase."""

from __future__ import annotations

import ast
import logging
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

# ---------- config ----------

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


@dataclass(frozen=True)
class ReadmeConfig:
    help_commands: Tuple[Tuple[str, Tuple[str, ...]], ...]
    scan_dirs: Tuple[str, ...]
    exclude_parts: Tuple[str, ...]
    docstring_max_lines: int


CONFIG = ReadmeConfig(
    help_commands=(
        ("traffic_analyzer", ("python", "core/traffic_analyzer.py", "--help")),
        ("pet_summary", ("python", "analysis/pet_summary.py", "--help")),
        ("research_run", ("python", "analysis/research_run.py", "--help")),
    ),
    scan_dirs=("analysis", "grid_trajectory", "traffic_diffusion", "tests"),
    exclude_parts=(".git", "__pycache__", ".pytest_cache"),
    docstring_max_lines=1,
)

MARKERS = ("COMMANDS", "TREE", "SCRIPTS")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("update_readme")


# ---------- helpers ----------

def run_help(cmd: Sequence[str]) -> str:
    """Execute a help command and return output."""
    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "(Help generation timed out)"
    except Exception as e:  # noqa: BLE001
        return f"(Could not generate help: {e})"


def extract_docstring(pyfile: Path, max_lines: int) -> str:
    """Extract summary from module docstring."""
    try:
        mod = ast.parse(pyfile.read_text(encoding="utf-8"))
        doc = ast.get_docstring(mod)
        if not doc:
            return ""
        lines = [ln.strip() for ln in doc.strip().splitlines() if ln.strip()]
        if not lines:
            return ""
        if max_lines <= 0 or len(lines) <= max_lines:
            return " ".join(lines)
        return " ".join(lines[:max_lines])
    except SyntaxError as e:
        return f"(Syntax error: {e})"
    except Exception:  # noqa: BLE001
        return ""


def replace_block(text: str, key: str, content: str) -> str:
    """Replace content between markers for a given key."""
    pattern = rf"(<!-- AUTO:{key}:START -->)(.*?)(<!-- AUTO:{key}:END -->)"
    repl = rf"\1\n{content}\n\3"
    return re.sub(pattern, repl, text, flags=re.DOTALL)


def has_marker(text: str, key: str) -> bool:
    return f"<!-- AUTO:{key}:START -->" in text and f"<!-- AUTO:{key}:END -->" in text


def iter_python_files(dirs: Iterable[str]) -> Iterable[Path]:
    for folder in dirs:
        root_dir = ROOT / folder
        if not root_dir.exists():
            continue
        for path in root_dir.rglob("*.py"):
            if any(part in path.parts for part in CONFIG.exclude_parts):
                continue
            yield path


# ---------- section builders ----------

def build_commands_section() -> str:
    parts: List[str] = []
    for name, cmd in CONFIG.help_commands:
        cmd_list = list(cmd)
        help_text = run_help(cmd_list)
        parts.append(
            f"### {name}\n\n"
            f"```bash\n{' '.join(cmd_list)}\n```\n\n"
            f"```text\n{help_text}\n```"
        )
    return "\n\n".join(parts)


def build_tree_section() -> str:
    """Simple ascii tree for key directories, without external deps."""
    lines: List[str] = ["```text", "nnds/"]

    def should_exclude(p: Path) -> bool:
        return any(part in p.parts for part in CONFIG.exclude_parts)

    for folder in CONFIG.scan_dirs:
        base = ROOT / folder
        if not base.exists():
            continue
        lines.append(f"├── {folder}/")
        # Collect entries relative to base
        entries = [
            p for p in base.rglob("*")
            if not should_exclude(p)
        ]
        rels = sorted(p.relative_to(ROOT) for p in entries)

        # Build a simple indent-based tree
        for rel in rels:
            parts = list(rel.parts)
            # first element is folder (already printed), skip it
            if not parts:
                continue
            depth = len(parts) - 1
            name = parts[-1] + ("/" if (ROOT / rel).is_dir() else "")
            indent = "    " * depth
            lines.append(f"{indent}├── {name}")

    lines.append("```")
    return "\n".join(lines)


def build_scripts_section() -> str:
    rows: List[str] = []
    for pyfile in sorted(iter_python_files(CONFIG.scan_dirs)):
        rel = pyfile.relative_to(ROOT)
        summary = extract_docstring(pyfile, CONFIG.docstring_max_lines)
        if not summary:
            summary = "No module docstring yet."
        rows.append(f"- `{rel}` — {summary}")
    return "\n".join(rows)


# ---------- main ----------

def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    dry_run = "--dry-run" in argv

    start = time.time()
    logger.info("📚 Building README documentation...")

    if not README.exists():
        logger.error(f"README.md not found at {README}")
        return 1

    original = README.read_text(encoding="utf-8")

    # Marker validation (warn only)
    for key in MARKERS:
        if not has_marker(original, key):
            logger.warning(f"⚠️ Missing AUTO:{key} markers in README")

    text = original
    text = replace_block(text, "COMMANDS", build_commands_section())
    text = replace_block(text, "TREE", build_tree_section())
    text = replace_block(text, "SCRIPTS", build_scripts_section())

    if text == original:
        logger.info("📝 No changes needed")
    elif dry_run:
        logger.info("🔍 DRY RUN – changes would be written, showing first 400 chars:")
        logger.info(text[:400] + ("..." if len(text) > 400 else ""))
    else:
        README.write_text(text, encoding="utf-8")
        logger.info(f"✅ Updated {README}")

    elapsed = time.time() - start
    logger.info(f"✨ Completed in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
