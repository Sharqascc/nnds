"""Guard: no two test functions may share a name within the same file.

Python silently keeps only the last definition of a duplicated function
name. In the past this caused several real assertions in
test_pet_verification_visualizer.py to be shadowed and never run. This
meta-test AST-parses every test_*.py in tests/ and fails if a file
defines the same top-level function name more than once.
"""

import ast
from collections import Counter
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent


def _top_level_function_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    return [
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def test_no_duplicate_test_function_names():
    offenders: dict[str, list[str]] = {}
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        names = _top_level_function_names(path)
        dupes = sorted(n for n, c in Counter(names).items() if c > 1)
        if dupes:
            offenders[path.name] = dupes
    assert not offenders, (
        "Duplicate test function names found (Python keeps only the last "
        f"definition, silently skipping the earlier ones): {offenders}"
    )
