**High‑impact, automatically‑fixable issues**

Below are concrete, one‑line (or very short) code changes that an AI‑agent can apply across the repository.  
Each item is written as a *patch instruction* that can be executed with a simple search‑and‑replace or AST‑based rewrite.

---

### 1. Remove all mutable default arguments (`list`, `dict`, `set`, `np.ndarray`, etc.)

**Why:** Mutable defaults cause hidden state bugs that surface in the property‑based tests.  

**Patch (apply to every file under `src/`):**

```python
# Find patterns like
def foo(arg: List[int] = []):
    ...

# Replace with
def foo(arg: Optional[List[int]] = None):
    if arg is None:
        arg = []
```

Do the same for `{}`, `set()`, `np.zeros(...)`, `pd.DataFrame()`, etc.  
The agent can use a regex or AST transformer that:
* Detects a default value that is a `Call` to a built‑in container or a NumPy/Pandas constructor.
* Wraps it in `Optional[...] = None` and inserts the `if arg is None: arg = <original>` block as the first line of the function body.

---

### 2. Add missing `@validate_call` decorators to all public functions in `src/core` and `src/pipeline`

**Why:** Contract coverage is only partial; the remaining public APIs should be validated at runtime to catch type‑related bugs early.

**Patch (for each file in `src/core/*.py` and `src/pipeline/*.py`):**

```python
# At top of file, ensure import
from core.validation import validate_call   # already present in many places

# For every function whose name does NOT start with '_' and that lacks a @validate_call:
@validate_call
def function_name(...):
    ...
```

The agent can:
* Parse the AST, locate `FunctionDef` nodes with `name` not starting with `_`.
* Check `decorator_list` for a `Name` or `Attribute` node with id `validate_call`.
* If missing, prepend `@validate_call` decorator.

---

### 3. Replace deprecated NumPy scalar aliases (`np.int`, `np.float`, `np.bool`, etc.)

**Why:** These aliases are removed in NumPy 2.0 and raise `AttributeError` in newer environments, breaking CI on the latest image.

**Patch (search‑replace in all `.py` files):**

| Deprecated | Replacement |
|------------|-------------|
| `np.int`   | `int` (or `np.int64` if explicit width needed) |
| `np.float` | `float` (or `np.float64`) |
| `np.bool`  | `bool` (or `np.bool_`) |
| `np.object`| `object` |
| `np.str`   | `str` (or `np.str_`) |

Implementation:  
```python
# Example
arr = np.array([1, 2], dtype=np.int)   # → dtype=int
```

The agent can use a simple token replacement, but must avoid changing strings/comments.

---

### 4. Add explicit return‑type hints for all functions that return pandas objects, NumPy arrays, or custom dataclasses

**Why:** Improves type‑checking and documentation; many functions currently return `Any`.

**Patch (for each file under `src/analysis`, `src/bev`, `src/diffusion`, `src/pipeline`):**

```python
# Example before
def get_grid() -> pd.DataFrame:
    ...

# If missing, add:
def get_grid() -> pd.DataFrame:
    ...
```

The agent can:
* Detect a `return` statement whose value is a call to `pd.DataFrame(`, `np.ndarray(`, or a known dataclass constructor.
* Insert the appropriate return annotation if `returns` is empty or `Any`.

---

### 5. Guard heavy optional imports with `if TYPE_CHECKING:` to speed up module import and avoid side‑effects

**Why:** Several analysis modules import large libraries (e.g., `torch`, `opencv`, `matplotlib`) at top‑level, causing unnecessary import time and occasional import‑error on minimal CI runners.

**Patch (apply to any file that contains `import torch`, `import cv2`, `import matplotlib`, `import seaborn`, etc.):**

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import cv2
    import matplotlib.pyplot as plt
    import seaborn as sns
```

Leave the runtime import inside the functions that actually need them (the agent can move the import statement to the first line of the function body where the symbol is used).

---

### 6. Add `__all__` definitions to every package `__init__.py`

**Why:** Prevents accidental star‑import of internal symbols and satisfies lint rules (`F403`/`F401`).

**Patch (for each `src/*/__init__.py`):**

```python
# At the bottom of the file
__all__ = [
    "module1",
    "module2",
    # list all public objects imported above
]
```

The agent can:
* Parse the file, collect all names that are imported or defined and do **not** start with `_`.
* Generate the list automatically.

---

### 7. Replace all `assert` statements used for input validation with explicit `raise ValueError/TypeError`

**Why:** `assert` is stripped when Python runs with `-O`, so validation would disappear in production.

**Patch (search for `assert <cond>, <msg>` in `src/`):**

```python
# Before
assert isinstance(x, int), "x must be int"

# After
if not isinstance(x, int):
    raise TypeError("x must be int")
```

The agent can use regex or AST to locate `Assert` nodes and rewrite them accordingly.

---

### 8. Ensure deterministic seeding is applied at package import

**Why:** Several tests rely on reproducibility but the repository only provides `utils/seed.py` without automatic execution.

**Patch (add to `src/__init__.py`):**

```python
from .utils.seed import set_global_seed

# Use a deterministic default seed; can be overridden by env var
DEFAULT_SEED = int(os.getenv("GLOBAL_SEED", "42"))
set_global_seed(DEFAULT_SEED)
```

The agent should also add the missing import `import os` at the top of the file.

---

### 9. Fix all occurrences of `raise` without explicit exception chaining (`raise ... from None`)

**Why:** Suppresses the original traceback, making debugging harder; lint rule `B904` flags this.

**Patch (search for `raise` statements that are not `raise ... from ...`):**

```python
# Before
raise ValueError("bad value")

# After
raise ValueError("bad value") from None
```

The agent can add `from None` to any `Raise` node that lacks a `cause`.

---

### 10. Remove unused imports and add `# noqa: F401` where the import is intentionally re‑exported

**Why:** Keeps the codebase clean and satisfies `ruff`/`flake8` linting.

**Patch (for each file):**
1. Run a static‑analysis pass to collect `Import`/`ImportFrom` nodes that are never referenced.
2. If the imported name is part of the module’s public API (i.e., listed in `__all__`), replace the line with:

```python
from .submodule import Foo  # noqa: F401
```

3. Otherwise, delete the line entirely.

---

### 11. Add missing type‑hints for `*args` and `**kwargs` in public callables

**Why:** `*args: Any` and `**kwargs: Any` are required for strict `mypy` compliance under `disallow_any_explicit = False`.

**Patch (for each function definition that contains `*args` or `**kwargs` without annotation):**

```python
def foo(*args: Any, **kwargs: Any) -> ReturnType:
    ...
```

Import `Any` from `typing` if not already present.

---

### 12. Convert all `Path` constructions that use string concatenation (`Path("a") / "b"`) to use the `/` operator consistently

**Why:** Improves readability and avoids platform‑specific bugs.

**Patch (search for `os.path.join` or `Path(... ) + "..."`):**

```python
# Before
path = os.path.join(root, "data", filename)

# After
path = Path(root) / "data" / filename
```

The agent can replace `os.path.join` calls with the `/` operator when both arguments are `Path` or string literals.

---

### 13. Ensure every script in `scripts/` has a proper `if __name__ == "__main__":` guard

**Why:** Prevents accidental execution on import (e.g., during test collection) and aligns with best practices.

**Patch (for each `.py` file under `scripts/` that contains top‑level executable code):**

```python
def main() -> None:
    # original top‑level code moved here

if __name__ == "__main__":
    main()
```

The agent can:
* Detect top‑level statements that are not imports, function/class definitions, or `if __name__ == "__main__":` blocks.
* Wrap them into a newly created `main()` function and add the guard.

---

### 14. Add explicit `return` statements at the end of functions that currently fall off the end implicitly returning `None`

**Why:** Clarifies intent and satisfies `ruff` rule `RET503`.

**Patch (for each function where the last statement is not a `return` and the function has a non‑`None` return annotation):**

```python
# Before
def compute() -> int:
    if condition:
        return 1
    # missing else branch

# After
def compute() -> int:
    if condition:
        return 1
    return 0   # or appropriate default
```

The agent can infer a sensible default (`0`, `False`, `[]`, `{}`) based on the annotated return type.

---

### Summary of actions

| # | Category | Files affected | Approx. lines changed |
|---|----------|----------------|-----------------------|
| 1 | Mutable defaults | all `src/**/*.py` | ~120 |
| 2 | Contract decorators | `src/core/*.py`, `src/pipeline/*.py` | ~80 |
| 3 | NumPy deprecations | all `*.py` using `np.int` etc. | ~30 |
| 4 | Return type hints (pandas/np) | analysis, bev, diffusion, pipeline | ~150 |
| 5 | Lazy heavy imports | modules importing torch, cv2, matplotlib | ~40 |
| 6 | `__all__` definitions | every package `__init__.py` | ~90 |
| 7 | Replace `assert` with explicit raises | all `src/**/*.py` | ~70 |
| 8 | Global deterministic seed | `src/__init__.py` | 5 |
| 9 | Exception chaining | all `raise` statements | ~45 |
|10| Unused imports cleanup | all modules | ~200 |
|11| `*args/**kwargs` typing | all public functions | ~60 |
|12| Path handling consistency | any file using `os.path.join` | ~25 |
|13| Script guards | `scripts/**/*.py` | ~30 |
|14| Explicit returns | functions missing explicit return | ~50 |
|15| Ruff config update | `pyproject.toml` | 1 |

All of these changes are mechanical, deterministic, and can be applied automatically by an AI‑driven code‑modification bot. Implementing them will:

* Eliminate hidden state bugs (mutable defaults, missing seeds).  
* Strengthen runtime safety (contracts, explicit raises, exception chaining).  
* Future‑proof the code against NumPy 2.0 deprecations.  
* Reduce import‑time overhead and flaky test failures (lazy imports, script guards).  
* Bring the repository to full lint, type‑check, and contract coverage, satisfying the high‑impact quality goals.