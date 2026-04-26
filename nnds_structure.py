#!/usr/bin/env python
# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 NNDS PIPELINE STRUCTURE VISUALIZER (ENHANCED, PRODUCTION-READY)
# - Auto-detect root
# - Safe file size handling
# - Icons per file type
# - Optional tqdm progress bar (used internally in helpers if needed)
# - Search, latest files, health checks
# - Export to TXT / JSON, baseline diff
# - CLI arguments
# - Jupyter/Colab-safe argparse handling
# ═══════════════════════════════════════════════════════════════════════════════

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Optional tqdm (fallback if not installed)
try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover
    TQDM_AVAILABLE = False

    def tqdm(iterable, **kwargs):
        return iterable  # no-op fallback


# ───────────────────────────────────────────────────────────────────────────────
# ROOT DETECTION
# ───────────────────────────────────────────────────────────────────────────────

def find_root() -> Path:
    """
    Auto-detect NNDS root directory.

    Order:
    - ./nnds
    - ../nnds
    - /content/nnds      (Colab)
    - /workspace/nnds    (Docker / devcontainer)
    - fallback: cwd
    """
    candidates = [
        Path.cwd() / "nnds",
        Path.cwd().parent / "nnds",
        Path("/content/nnds"),
        Path("/workspace/nnds"),
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p.resolve()
    return Path.cwd().resolve()


# ───────────────────────────────────────────────────────────────────────────────
# UTILS: SIZE, ICONS, SAFE STAT
# ───────────────────────────────────────────────────────────────────────────────

def format_size(size_bytes: int) -> str:
    """Format file size into human-readable units."""
    try:
        size = float(size_bytes)
    except (TypeError, ValueError):
        return "0 B"

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


ICONS: Dict[str, str] = {
    ".py": "🐍",
    ".ipynb": "📓",
    ".json": "📋",
    ".yaml": "⚙️",
    ".yml": "⚙️",
    ".csv": "📊",
    ".npy": "🔢",
    ".npz": "🔢",
    ".mp4": "🎥",
    ".avi": "🎥",
    ".mov": "🎥",
    ".pt": "🧠",
    ".pth": "🧠",
    ".ckpt": "🧠",
    ".png": "🖼️",
    ".jpg": "🖼️",
    ".jpeg": "🖼️",
    ".pdf": "📕",
    ".sh": "⚡",
    ".md": "📝",
    ".txt": "📄",
    ".log": "📋",
    ".parquet": "📊",
    ".feather": "📊",
}


def get_icon(path: Path) -> str:
    """Return an icon based on file extension (or generic)."""
    if path.is_dir():
        return "📁"
    return ICONS.get(path.suffix.lower(), "📄")


def get_file_size_safe(path: Path) -> int:
    """
    Safe file size retrieval.

    - Skips symlinks
    - Catches PermissionError, OSError, FileNotFoundError
    """
    try:
        if path.is_symlink():
            return 0
        return path.stat().st_size
    except (PermissionError, OSError, FileNotFoundError):
        return 0


# ───────────────────────────────────────────────────────────────────────────────
# CORE: DIRECTORY TREE (TEXT + JSON)
# ───────────────────────────────────────────────────────────────────────────────

IGNORE_PATTERNS = [
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".ipynb_checkpoints",
    "node_modules",
]


def should_ignore(path: Path) -> bool:
    """Return True if this path should be ignored based on IGNORE_PATTERNS."""
    s = str(path)
    return any(p in s for p in IGNORE_PATTERNS)


def build_tree_lines(
    root: Path,
    max_depth: int = 10,
) -> Tuple[List[str], Dict[str, int]]:
    """
    Build an ASCII tree (text lines) and collect stats.

    Returns:
        lines: list of strings ready to print
        stats: dict with keys 'dirs', 'files', 'size'
    """
    lines: List[str] = []
    stats = {"dirs": 0, "files": 0, "size": 0}

    def _walk(path: Path, prefix: str = "", depth: int = 0):
        try:
            children = sorted(
                [p for p in path.iterdir() if not should_ignore(p)],
                key=lambda x: (not x.is_dir(), x.name.lower()),
            )
        except (PermissionError, OSError):
            lines.append(f"{prefix}└── 🔒 Locked: {path.name}/")
            return

        count = len(children)
        for idx, child in enumerate(children):
            is_last = idx == count - 1
            branch = "└── " if is_last else "├── "
            next_prefix = prefix + ("    " if is_last else "│   ")

            if child.is_dir():
                stats["dirs"] += 1
                lines.append(f"{prefix}{branch}{get_icon(child)} {child.name}/")
                if depth < max_depth:
                    _walk(child, next_prefix, depth + 1)
            else:
                size = get_file_size_safe(child)
                stats["files"] += 1
                stats["size"] += size
                icon = get_icon(child)
                lines.append(
                    f"{prefix}{branch}{icon} {child.name} ({format_size(size)})"
                )

    if not root.exists():
        lines.append(f"❌ Root not found: {root}")
        return lines, stats

    lines.append(f"{get_icon(root)} {root.name}/")
    _walk(root, "", 0)
    return lines, stats


def build_tree_json(root: Path, max_depth: int = 10) -> Dict[str, Any]:
    """
    Build a JSON-serializable representation of the directory tree.

    Node format:
    {
        "name": "file_or_dir_name",
        "path": "relative/from/root",
        "type": "file" | "dir",
        "size": int,
        "children": [...]
    }
    """
    def _node(path: Path, rel: Path, depth: int) -> Dict[str, Any]:
        node: Dict[str, Any] = {
            "name": path.name,
            "path": str(rel),
            "type": "dir" if path.is_dir() else "file",
            "size": 0,
            "children": [],
        }

        if path.is_file():
            node["size"] = get_file_size_safe(path)
            return node

        if depth >= max_depth:
            return node

        try:
            children = sorted(
                [p for p in path.iterdir() if not should_ignore(p)],
                key=lambda x: (not x.is_dir(), x.name.lower()),
            )
        except (PermissionError, OSError):
            return node

        for child in children:
            child_rel = rel / child.name
            child_node = _node(child, child_rel, depth + 1)
            node["children"].append(child_node)
            node["size"] += child_node.get("size", 0)

        return node

    if not root.exists():
        return {"error": f"Root not found: {str(root)}"}

    return _node(root, Path("."), 0)


# ───────────────────────────────────────────────────────────────────────────────
# SEARCH + LATEST FILES
# ───────────────────────────────────────────────────────────────────────────────

def search_files(root: Path, pattern: str, case_sensitive: bool = False) -> List[Path]:
    """
    Search for files whose names contain 'pattern'.

    Returns list of Paths.
    """
    if not pattern:
        return []
    results: List[Path] = []
    patt = pattern if case_sensitive else pattern.lower()

    for p in root.rglob("*"):
        if should_ignore(p) or not p.is_file():
            continue
        name = p.name if case_sensitive else p.name.lower()
        if patt in name:
            results.append(p)
    return results


def latest_files(root: Path, n: int = 5) -> List[Tuple[Path, float]]:
    """
    Return the n most recently modified files under root.

    Returns list of (Path, mtime).
    """
    files: List[Tuple[Path, float]] = []
    for p in root.rglob("*"):
        if should_ignore(p) or not p.is_file():
            continue
        try:
            mtime = p.stat().st_mtime
        except (PermissionError, OSError, FileNotFoundError):
            continue
        files.append((p, mtime))

    files.sort(key=lambda x: x[1], reverse=True)
    return files[:n]


# ───────────────────────────────────────────────────────────────────────────────
# HEALTH CHECKS
# ───────────────────────────────────────────────────────────────────────────────

def check_pipeline_health(root: Path) -> List[str]:
    """
    Verify that critical pipeline files exist and are non-empty.

    Returns list of issue strings (empty if healthy).
    """
    critical = {
        "configs/bev_config.json": "BEV configuration",
        "configs/GITI_grid_config.json": "Grid configuration",
        "configs/giti_calibration_points.json": "Calibration points",
        "sam3.pt": "SAM3 model weights (3.45GB)",
        "videos/traffic_video.mp4": "Demo traffic video",
        "traffic_analyzer.py": "Main pipeline entry",
        "analysis/pet_diffusion_analysis.py": "Diffusion analysis module",
        "analysis/visualization/__init__.py": "Visualization package",
        "outputs": "Outputs directory (may be empty)",
    }

    issues: List[str] = []
    for rel, desc in critical.items():
        full = root / rel
        if not full.exists():
            issues.append(f"❌ Missing: {desc} ({rel})")
        else:
            if full.is_file():
                size = get_file_size_safe(full)
                if size == 0:
                    issues.append(f"⚠️ Empty file: {desc} ({rel})")
    return issues


# ───────────────────────────────────────────────────────────────────────────────
# BASELINE COMPARISON
# ───────────────────────────────────────────────────────────────────────────────

def snapshot_structure(root: Path, max_depth: int = 10) -> Dict[str, Any]:
    """
    Create a flat snapshot of the structure for baseline comparison.

    Returns:
        {
          "generated_at": "...",
          "root": "path",
          "files": {
              "relative/path.ext": {
                  "size": int,
                  "mtime": float
              },
              ...
          }
        }
    """
    files: Dict[str, Dict[str, Any]] = {}
    for p in root.rglob("*"):
        if should_ignore(p) or not p.is_file():
            continue
        rel = p.relative_to(root)
        try:
            st = p.stat()
        except (PermissionError, OSError, FileNotFoundError):
            continue
        files[str(rel)] = {
            "size": st.st_size,
            "mtime": st.st_mtime,
        }

    return {
        "generated_at": datetime.now().isoformat(),
        "root": str(root),
        "files": files,
    }


def compare_with_baseline(
    current_snapshot: Dict[str, Any],
    baseline_path: Path,
) -> Dict[str, Any]:
    """
    Compare current snapshot against a baseline JSON file.

    Returns dict with:
        {
          "new_files": [...],
          "deleted_files": [...],
          "changed_size": [...],
        }
    """
    if not baseline_path.exists():
        return {
            "error": f"Baseline file not found: {baseline_path}",
            "new_files": [],
            "deleted_files": [],
            "changed_size": [],
        }

    with baseline_path.open("r", encoding="utf-8") as f:
        baseline = json.load(f)

    old_files: Dict[str, Any] = baseline.get("files", {})
    new_files_dict: Dict[str, Any] = current_snapshot.get("files", {})

    old_set = set(old_files.keys())
    new_set = set(new_files_dict.keys())

    new_files = sorted(list(new_set - old_set))
    deleted_files = sorted(list(old_set - new_set))

    changed_size: List[Tuple[str, int, int]] = []
    for rel in sorted(list(old_set & new_set)):
        old_size = int(old_files[rel].get("size", 0))
        new_size = int(new_files_dict[rel].get("size", 0))
        if old_size != new_size:
            changed_size.append((rel, old_size, new_size))

    return {
        "new_files": new_files,
        "deleted_files": deleted_files,
        "changed_size": changed_size,
    }


# ───────────────────────────────────────────────────────────────────────────────
# ENHANCED ANALYSIS PRINTER
# ───────────────────────────────────────────────────────────────────────────────

def enhanced_analysis(root: Path):
    """Run additional analysis: latest files, health check, quick hints."""
    print("\n" + "=" * 80)
    print("🔍 ENHANCED ANALYSIS")
    print("=" * 80)

    # Latest modified files
    print("\n📅 Latest modified files:")
    latest = latest_files(root, n=5)
    if not latest:
        print("  (No files found)")
    else:
        for p, mtime in latest:
            ts = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            rel = p.relative_to(root)
            print(f"  {ts} │ {rel}")

    # Health check
    print("\n🩺 Pipeline health check:")
    issues = check_pipeline_health(root)
    if issues:
        for msg in issues:
            print(f"  {msg}")
    else:
        print("  ✅ All critical files present and non-empty")

    # Suggestions / quick commands
    print("\n💡 Quick commands (shell):")
    print("   - Count Python files:  find . -name '*.py' | wc -l")
    print("   - Total Python LOC:    find . -name '*.py' -exec cat {} \\; | wc -l")
    print("   - Grep imports:        rg 'import ' .")


# ───────────────────────────────────────────────────────────────────────────────
# EXPORTERS
# ───────────────────────────────────────────────────────────────────────────────

def export_to_file(lines: List[str], output_path: Path):
    """Save the textual tree to a file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"\n✅ Exported directory tree to: {output_path}")


def export_json(tree_json: Dict[str, Any], output_path: Path):
    """Save JSON representation to a file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(tree_json, f, indent=2)
    print(f"✅ Exported JSON structure to: {output_path}")


# ───────────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────────

def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NNDS Pipeline Directory Visualizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=str,
        help="Root directory of the NNDS repo (auto-detected if omitted)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=10,
        help="Max directory depth to traverse",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Export text tree to this file (e.g. nnds_tree.txt)",
    )
    parser.add_argument(
        "--json",
        type=str,
        help="Export JSON structure to this file (e.g. nnds_tree.json)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        help="Compare against baseline JSON snapshot file",
    )
    parser.add_argument(
        "--save-baseline",
        type=str,
        help="Path to save current baseline snapshot JSON",
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Search pattern for file names",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Case-sensitive search",
    )
    parser.add_argument(
        "--no-enhanced",
        action="store_true",
        help="Disable enhanced analysis section",
    )
    return parser.parse_args(argv)


def _clean_argv_for_notebook(args: List[str]) -> List[str]:
    """
    Remove Jupyter/Colab-injected arguments like:
        -f /root/.local/share/jupyter/runtime/kernel-XXXX.json
        or '-f=/root/...'
    """
    cleaned: List[str] = []
    skip_next = False
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if a == "-f":
            skip_next = True
            continue
        if a.startswith("-f="):
            continue
        cleaned.append(a)
    return cleaned


def main(argv: List[str] = None):
    # If called without explicit argv (e.g., `main()` from a notebook),
    # clean sys.argv to strip Jupyter's '-f kernel.json' so argparse doesn't fail.
    if argv is None:
        raw = sys.argv[1:]
        argv = _clean_argv_for_notebook(raw)

    args = parse_args(argv)

    root = Path(args.root).resolve() if args.root else find_root()

    print("=" * 80)
    print("🚀 NNDS PIPELINE - COMPLETE DIRECTORY STRUCTURE")
    print("=" * 80)
    print(f"📍 Root: {root}")
    print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Build tree (text)
    lines, stats = build_tree_lines(root, max_depth=args.depth)

    print("\n📦 Pipeline tree:\n")
    for line in lines:
        print(line)

    print("\n" + "=" * 80)
    print("📊 PIPELINE STATISTICS")
    print("=" * 80)
    print(f"📁 Total Directories: {stats['dirs']}")
    print(f"📄 Total Files:       {stats['files']}")
    print(f"💾 Total Size:        {format_size(stats['size'])}")

    # JSON export / snapshot
    tree_json = build_tree_json(root, max_depth=args.depth)

    if args.output:
        export_to_file(lines, Path(args.output))

    if args.json:
        export_json(tree_json, Path(args.json))

    # Baseline snapshot + comparison
    snapshot = snapshot_structure(root, max_depth=args.depth)

    if args.save_baseline:
        export_json(snapshot, Path(args.save_baseline))

    if args.baseline:
        diff = compare_with_baseline(snapshot, Path(args.baseline))
        print("\n" + "=" * 80)
        print("🔁 CHANGES SINCE BASELINE")
        print("=" * 80)

        if "error" in diff:
            print(diff["error"])
        else:
            if diff["new_files"]:
                print("\n🆕 New files:")
                for rel in diff["new_files"]:
                    print(f"  + {rel}")
            else:
                print("\n🆕 New files: (none)")

            if diff["deleted_files"]:
                print("\n🗑️ Deleted files:")
                for rel in diff["deleted_files"]:
                    print(f"  - {rel}")
            else:
                print("\n🗑️ Deleted files: (none)")

            if diff["changed_size"]:
                print("\n📏 Size changes:")
                for rel, old, new in diff["changed_size"]:
                    print(
                        f"  ~ {rel}: {format_size(old)} → {format_size(new)}"
                    )
            else:
                print("\n📏 Size changes: (none)")

    # Search
    if args.search:
        print("\n" + "=" * 80)
        print(f"🔎 SEARCH RESULTS for pattern: '{args.search}'")
        print("=" * 80)
        results = search_files(root, args.search, case_sensitive=args.case_sensitive)
        if not results:
            print("No matching files found.")
        else:
            for p in results:
                rel = p.relative_to(root)
                size = get_file_size_safe(p)
                print(f"  {rel} ({format_size(size)})")
            print(f"\nTotal matches: {len(results)}")

    # Enhanced analysis (latest files + health + hints)
    if not args.no_enhanced:
        enhanced_analysis(root)

    print("\n" + "=" * 80)
    print("✅ PIPELINE STRUCTURE COMPLETE")
    print("=" * 80)
    print("\n🚀 To run the main pipeline:")
    print("   PYTHONPATH=. python traffic_analyzer.py --video videos/traffic_video.mp4")
    print()


if __name__ == "__main__":
    main()
