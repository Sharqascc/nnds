#!/usr/bin/env python
# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 NNDS PIPELINE STRUCTURE VISUALIZER (GIT BRANCH-AWARE, IMPROVED)
# - Shows directory structure of a Git branch (default: main) using git ls-tree
# - Falls back to filesystem tree if --branch is omitted or Git is unavailable
# - Icons per file type, stats, search, health checks, exports
# - Safer argv cleaning for notebooks
# - Git mode includes file sizes and supports search
# - Smarter ignore patterns and basic caching for filesystem walks
# ═══════════════════════════════════════════════════════════════════════════════

import os
import sys
import json
import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from functools import lru_cache


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


def get_icon_from_name(name: str, is_dir: bool) -> str:
    """Icon from a simple name + flag (for virtual Git tree)."""
    if is_dir:
        return "📁"
    suffix = Path(name).suffix.lower()
    return ICONS.get(suffix, "📄")


def get_file_size_safe(path: Path) -> int:
    """
    Safe file size retrieval.

    Returns 0 on error (permission, missing, etc.), which is treated
    as "unknown or empty" in the UI.
    """
    try:
        if path.is_symlink():
            return 0
        return path.stat().st_size
    except (PermissionError, OSError, FileNotFoundError):
        return 0


# ───────────────────────────────────────────────────────────────────────────────
# IGNORE PATTERNS
# ───────────────────────────────────────────────────────────────────────────────

IGNORE_PATTERNS = [
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".ipynb_checkpoints",
    "node_modules",
]


def should_ignore(path: Path) -> bool:
    """
    Check if path should be ignored based on path components.

    This avoids false positives like matching '.git' inside '.gitignore'.
    """
    parts = path.parts
    return any(ignore in parts for ignore in IGNORE_PATTERNS)


def should_ignore_str(path_str: str) -> bool:
    """
    Ignore helper for plain string paths (Git ls-tree output).

    Uses component-based split so '.gitignore' is not treated as '.git'.
    """
    parts = path_str.split("/")
    return any(ignore in parts for ignore in IGNORE_PATTERNS)


# ───────────────────────────────────────────────────────────────────────────────
# FILESYSTEM CACHING
# ───────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=4)
def get_all_files(root_str: str) -> List[Path]:
    """
    Cache all files under root (excluding ignored) to avoid repeated rglob.

    Keyed by resolved root string.
    """
    root = Path(root_str)
    return [
        p for p in root.rglob("*")
        if p.is_file() and not should_ignore(p)
    ]


# ───────────────────────────────────────────────────────────────────────────────
# CORE: DIRECTORY TREE (FILESYSTEM MODE)
# ───────────────────────────────────────────────────────────────────────────────

def build_tree_lines(
    root: Path,
    max_depth: int = 10,
) -> Tuple[List[str], Dict[str, int]]:
    """
    Build an ASCII tree (text lines) and collect stats from filesystem.

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
    Build a JSON-serializable representation of the directory tree (filesystem).
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
# GIT HELPERS (REPO CHECK + PATHS + SIZES)
# ───────────────────────────────────────────────────────────────────────────────

def is_git_repo(path: Path = None) -> bool:
    """Check if current directory is in a Git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=path or Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def stream_paths_from_branch(branch: str):
    """
    Generator that yields (path, size) from git ls-tree without loading all lines in memory.

    Uses:
        git ls-tree -r --long <branch>
    """
    if not is_git_repo():
        print("❌ Not a Git repository")
        return

    try:
        proc = subprocess.Popen(
            ["git", "ls-tree", "-r", "--long", branch],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        print("❌ git executable not found")
        return

    # Read stdout line by line
    for line in proc.stdout or []:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        try:
            metadata, path = line.split("\t", 1)
        except ValueError:
            continue
        meta_parts = metadata.split()
        if len(meta_parts) >= 4 and meta_parts[1] == "blob":
            try:
                size = int(meta_parts[3])
            except (ValueError, IndexError):
                size = 0
            if not should_ignore_str(path):
                yield path, size

    proc.wait()
    if proc.returncode != 0:
        err = (proc.stderr.read() if proc.stderr else "").strip()
        if err:
            print(f"❌ git ls-tree failed for branch '{branch}': {err}")


def list_paths_with_sizes_from_branch(branch: str, show_progress: bool = True) -> List[Tuple[str, int]]:
    """
    Get paths and their sizes from Git blob objects for a given branch.

    Uses stream_paths_from_branch, with optional tqdm progress for large repos.
    """
    # We don't know count upfront without loading all lines, so we just
    # wrap generator in tqdm if many items accumulate.
    paths: List[Tuple[str, int]] = []

    if not is_git_repo():
        print("❌ Not a Git repository")
        return []

    # First, collect into list, optionally with progress
    raw_iter = list(stream_paths_from_branch(branch))
    total = len(raw_iter)
    if show_progress and TQDM_AVAILABLE and total > 1000:
        print(f"📊 Processing {total} files from Git...")
        for path, size in tqdm(raw_iter, desc="Parsing Git tree", unit="files"):
            paths.append((path, size))
    else:
        paths = raw_iter

    return paths


# ───────────────────────────────────────────────────────────────────────────────
# GIT-BRANCH MODE: BUILD VIRTUAL TREE
# ───────────────────────────────────────────────────────────────────────────────

def build_virtual_tree(paths_with_sizes: List[Tuple[str, int]]) -> Dict[str, Any]:
    """
    Build a nested dict representing directories/files from a list of (path, size).

    Root format:
      {
        "name": ".",
        "type": "dir",
        "children": { "dir": {...}, "file.py": {...}, ... },
        "size": int
      }
    """
    root = {"name": ".", "type": "dir", "children": {}, "size": 0}

    for path, size in paths_with_sizes:
        parts = path.split("/")
        node = root
        node["size"] += size  # accumulate total size at root

        for i, part in enumerate(parts):
            is_last = (i == len(parts) - 1)
            children = node.setdefault("children", {})

            # Conflict: existing node type mismatch
            if part in children:
                expected_type = "file" if is_last else "dir"
                if children[part]["type"] != expected_type:
                    print(f"⚠️ Conflict in Git tree: {path} (file/dir name collision)")
                    # Skip this path; cannot represent both cleanly
                    node = None
                    break
            else:
                children[part] = {
                    "name": part,
                    "type": "file" if is_last else "dir",
                    "children": {} if not is_last else None,
                    "size": 0,
                }

            node = children[part]
            node["size"] += size  # propagate size up dir tree

        # If conflict occurred, node will be None and we skip
        if node is None:
            continue

    return root


def render_virtual_tree(
    tree: Dict[str, Any],
    prefix: str = "",
    is_last: bool = True,
    lines: List[str] = None,
) -> List[str]:
    """
    Render the virtual tree (from Git paths) into ASCII lines, including sizes.
    """
    if lines is None:
        lines = []

    name = tree["name"]
    is_dir = tree["type"] == "dir"
    icon = get_icon_from_name(name, is_dir)

    # Root ('.') special-case: don't print the dot
    if name != ".":
        branch_symbol = "└── " if is_last else "├── "
        if is_dir:
            lines.append(f"{prefix}{branch_symbol}{icon} {name}/ ({format_size(tree['size'])})")
        else:
            lines.append(f"{prefix}{branch_symbol}{icon} {name} ({format_size(tree['size'])})")

        prefix = prefix + ("    " if is_last else "│   ")

    if is_dir and tree.get("children"):
        items = sorted(
            tree["children"].values(),
            key=lambda x: (x["type"] != "dir", x["name"].lower()),
        )
        for idx, child in enumerate(items):
            render_virtual_tree(
                child,
                prefix=prefix,
                is_last=(idx == len(items) - 1),
                lines=lines,
            )
    return lines


def virtual_tree_stats(tree: Dict[str, Any]) -> Dict[str, int]:
    """
    Stats (dirs, files, size) from the virtual tree.
    """
    dirs = 0
    files = 0

    def _walk(node: Dict[str, Any]):
        nonlocal dirs, files
        if node["type"] == "dir":
            if node["name"] != ".":
                dirs += 1
            for child in (node.get("children") or {}).values():
                _walk(child)
        else:
            files += 1

    _walk(tree)
    return {"dirs": dirs, "files": files, "size": tree.get("size", 0)}


def build_virtual_tree_json(tree: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert virtual tree dict (with children as dicts) into JSON-serializable form.
    """
    def _convert(node: Dict[str, Any], current_path: List[str]) -> Dict[str, Any]:
        name = node["name"]
        is_dir = node["type"] == "dir"
        path_str = "." if not current_path else "/".join(current_path)
        result = {
            "name": name,
            "path": path_str,
            "type": "dir" if is_dir else "file",
            "size": node.get("size", 0),
            "children": [],
        }
        if is_dir and node.get("children"):
            for child in sorted(
                node["children"].values(),
                key=lambda x: (x["type"] != "dir", x["name"].lower()),
            ):
                child_path = current_path + [child["name"]] if name != "." else [child["name"]]
                result["children"].append(_convert(child, child_path))
        return result

    return _convert(tree, [])


def search_git_tree(tree: Dict[str, Any], pattern: str, case_sensitive: bool = False) -> List[str]:
    """
    Search for files in virtual Git tree.

    Returns list of paths (relative to repo root in the branch).
    """
    results: List[str] = []
    if not pattern:
        return results

    patt = pattern if case_sensitive else pattern.lower()

    def _search(node: Dict[str, Any], path: str = ""):
        name = node["name"]
        if node["type"] == "file":
            full_name = f"{path}/{name}" if path else name
            compare_name = full_name if case_sensitive else full_name.lower()
            if patt in compare_name:
                results.append(full_name)
        elif node.get("children"):
            new_path = f"{path}/{name}" if path and name != "." else (name if name != "." else "")
            for child in node["children"].values():
                _search(child, new_path)

    _search(tree)
    return results


# ───────────────────────────────────────────────────────────────────────────────
# SEARCH + LATEST FILES (FILESYSTEM MODE ONLY)
# ───────────────────────────────────────────────────────────────────────────────

def search_files(root: Path, pattern: str, case_sensitive: bool = False) -> List[Path]:
    """
    Search for files whose names contain 'pattern' on filesystem.

    Returns list of Paths.
    """
    if not pattern:
        return []
    patt = pattern if case_sensitive else pattern.lower()
    files = get_all_files(str(root.resolve()))
    results: List[Path] = []

    for p in files:
        name = p.name if case_sensitive else p.name.lower()
        if patt in name:
            results.append(p)
    return results


def latest_files(root: Path, n: int = 5) -> List[Tuple[Path, float]]:
    """
    Return the n most recently modified files under root (filesystem).
    """
    files = get_all_files(str(root.resolve()))
    items: List[Tuple[Path, float]] = []
    for p in files:
        try:
            mtime = p.stat().st_mtime
        except (PermissionError, OSError, FileNotFoundError):
            continue
        items.append((p, mtime))

    items.sort(key=lambda x: x[1], reverse=True)
    return items[:n]


# ───────────────────────────────────────────────────────────────────────────────
# HEALTH CHECKS (WORKING TREE)
# ───────────────────────────────────────────────────────────────────────────────

def check_pipeline_health(root: Path) -> List[str]:
    """
    Verify that critical pipeline files exist and are non-empty in current working tree.

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
                    issues.append(f"⚠️ Empty or unreadable file: {desc} ({rel})")
    return issues


# ───────────────────────────────────────────────────────────────────────────────
# BASELINE COMPARISON (FILESYSTEM SNAPSHOT)
# ───────────────────────────────────────────────────────────────────────────────

def snapshot_structure(root: Path, max_depth: int = 10) -> Dict[str, Any]:
    """
    Create a flat snapshot of the structure for baseline comparison (filesystem).
    """
    files: Dict[str, Dict[str, Any]] = {}
    for p in root.rglob("*"):
        if should_ignore(p) or not p.is_file():
            continue
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue
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
        "root": str(root.resolve()),
        "files": files,
    }


def compare_with_baseline(
    current_snapshot: Dict[str, Any],
    baseline_path: Path,
) -> Dict[str, Any]:
    """
    Compare current snapshot against a baseline JSON file.
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
# ENHANCED ANALYSIS (FILESYSTEM MODE)
# ───────────────────────────────────────────────────────────────────────────────

def enhanced_analysis(root: Path):
    """Run additional analysis: latest files, health check, quick hints."""
    print("\n" + "=" * 80)
    print("🔍 ENHANCED ANALYSIS (current working tree)")
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


# ────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────

def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NNDS Pipeline Directory Visualizer (Git branch-aware)",
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
        help="Max directory depth to traverse (filesystem mode)",
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
        help="Compare against baseline JSON snapshot file (filesystem mode)",
    )
    parser.add_argument(
        "--save-baseline",
        type=str,
        help="Path to save current baseline snapshot JSON (filesystem mode)",
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
        help="Disable enhanced analysis section (filesystem mode)",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="main",
        help="Git branch to visualize (uses git ls-tree; default: main). "
             "Set empty string ('') or use --no-git to use filesystem mode only.",
    )
    parser.add_argument(
        "--no-git",
        action="store_true",
        help="Force filesystem mode, ignore --branch",
    )
    return parser.parse_args(argv)


def _clean_argv_for_notebook(args: List[str]) -> List[str]:
    """
    Remove known Jupyter/Colab-injected arguments like:
        -f /path/to/kernel-XXXX.json
        --log-level, --debug, etc.
    Without blindly stripping arbitrary flags.
    """
    jupyter_flags = {"-f", "--log-level", "--debug"}
    cleaned: List[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        # Skip -f <value> or -f=<value>
        if arg in jupyter_flags:
            if arg == "-f" and i + 1 < len(args) and not args[i + 1].startswith("-"):
                i += 2
            else:
                i += 1
            continue
        if arg.startswith("-f="):
            i += 1
            continue
        cleaned.append(arg)
        i += 1
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
    if args.no_git or not args.branch:
        print("🌿 Git branch: (disabled, using filesystem)")
    else:
        print(f"🌿 Git branch (tree source): {args.branch}")
    print("=" * 80)

    use_branch_mode = bool(args.branch) and not args.no_git

    # ── BRANCH MODE: show structure from Git `main` (or given branch)
    if use_branch_mode:
        paths_with_sizes = list_paths_with_sizes_from_branch(args.branch, show_progress=True)
        if not paths_with_sizes:
            print(f"❌ Could not read tree for branch '{args.branch}'. "
                  f"Falling back to filesystem view.\n")
            use_branch_mode = False
        else:
            vtree = build_virtual_tree(paths_with_sizes)
            lines = [f"📁 {args.branch}/ ({format_size(vtree.get('size', 0))})"]
            lines = render_virtual_tree(vtree, prefix="", is_last=True, lines=lines)
            stats = virtual_tree_stats(vtree)
            tree_json = build_virtual_tree_json(vtree)

    # ── FILESYSTEM MODE (fallback or explicit)
    if not use_branch_mode:
        # Clear cache to ensure consistency if root changes between runs
        get_all_files.cache_clear()
        # Build tree (text)
        lines, stats = build_tree_lines(root, max_depth=args.depth)
        tree_json = build_tree_json(root, max_depth=args.depth)

    print("\n📦 Pipeline tree:\n")
    for line in lines:
        print(line)

    print("\n" + "=" * 80)
    print("📊 PIPELINE STATISTICS")
    print("=" * 80)
    print(f"📁 Total Directories: {stats['dirs']}")
    print(f"📄 Total Files:       {stats['files']}")
    print(f"💾 Total Size:        {format_size(stats['size'])}")

    # JSON export
    if args.json:
        export_json(tree_json, Path(args.json))

    # SEARCH (both modes)
    if args.search:
        print("\n" + "=" * 80)
        print(f"🔎 SEARCH RESULTS for pattern: '{args.search}'")
        print("=" * 80)
        if use_branch_mode:
            results = search_git_tree(vtree, args.search, case_sensitive=args.case_sensitive)
            if not results:
                print("No matching files found in Git tree.")
            else:
                for rel in results:
                    print(f"  {rel}")
                print(f"\nTotal matches: {len(results)}")
        else:
            results = search_files(root, args.search, case_sensitive=args.case_sensitive)
            if not results:
                print("No matching files found.")
            else:
                for p in results:
                    rel = p.relative_to(root)
                    size = get_file_size_safe(p)
                    print(f"  {rel} ({format_size(size)})")
                print(f"\nTotal matches: {len(results)}")

    # FILESYSTEM-ONLY FEATURES
    if not use_branch_mode:
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

        # Enhanced analysis (latest files + health + hints)
        if not args.no_enhanced:
            enhanced_analysis(root)

    print("\n" + "=" * 80)
    print("✅ PIPELINE STRUCTURE COMPLETE")
    print("=" * 80)
    print("\n🚀 To run the main pipeline (from current working tree):")
    print("   PYTHONPATH=. python traffic_analyzer.py --video videos/traffic_video.mp4")
    print()


if __name__ == "__main__":
    main()
