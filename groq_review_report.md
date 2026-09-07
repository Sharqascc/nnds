

## src/analysis/__init__.py (part 1/1)
Here is the review of `src/analysis/__init__.py`.

### 1. Bugs & Logic Errors

*   **Inconsistent Lazy Import Logic (`__getattr__`)**:
    The logic `attr = getattr(module, name) if name == "PETEventAnalyzer" else module` is fragile and inconsistent with the `_LAZY_IMPORTS` map.
    *   If you add another class to `_LAZY_IMPORTS` (e.g., `"SSMCalculator": "src.analysis.ssm"`), this code will incorrectly return the *module* instead of the *class*, because `name != "PETEventAnalyzer"`.
    *   **Fix**: Store the target attribute name explicitly in the map or use a consistent rule (e.g., always try `getattr(module, name)` and fall back to `module` if it fails, or define a tuple `(module_path, attr_name)`).

*   **Redundant/Inefficient Availability Checks**:
    `_viz_available` and `_pet_summary_available` are computed at import time using `importlib.util.find_spec`. However, `check_installation` performs actual imports (`importlib.import_module`) to determine status.
    *   `find_spec` does not execute the module, so it may report `True` even if the module has a syntax error or missing dependency at runtime.
    *   These flags are defined but **never used** in the provided code. If they are not used elsewhere, remove them to avoid confusion. If they are used, they are unreliable indicators of *runtime* availability.

*   **`__dir__` Implementation**:
    `sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))` may include private attributes (e.g., `_LAZY_IMPORTS`, `_viz_available`) in the directory listing.
    *   **Fix**: Filter out keys starting with `_`.

### 2. Missing Contracts & Robustness

*   **No `__all__` Definition**:
    The module does not define `__all__`. This makes `from src.analysis import *` behavior unpredictable and less explicit.
    *   **Fix**: Define `__all__ = ["visualization", "PETEventAnalyzer", "check_installation", "__version__"]` (or similar) to explicitly control public API.

*   **Error Handling in `check_installation`**:



## src/analysis/conflict_classifier.py (part 1/1)
Here is the review of `src/analysis/conflict_classifier.py`.

### Critical Bugs & Logic Errors

1.  **Missing Lateral Offset Check for Rear-End vs. Side-Swipe**:
    *   **Issue**: The logic for `angle < 30.0` classifies based *only* on speed difference. A vehicle following directly behind another (rear-end) and a vehicle driving parallel in an adjacent lane (side-swipe) can have identical velocity vectors. Without checking the **lateral position** (perpendicular distance) at the conflict frame, this classifier will misclassify rear-ends as side-swipes if speeds are similar, or side-swipes as rear-ends if speeds differ slightly.
    *   **Fix**: Calculate the relative position vector at `conflict_frame`. Project this onto the axis perpendicular to the velocity vector. If the lateral offset is small (e.g., < 1.5m), it’s a rear-end. If large, it’s a side-swipe.

2.  **Zero-Velocity Handling**:
    *   **Issue**: `if v_a == (0, 0) or v_b == (0, 0): return "other"`.
    *   **Fix**: This is acceptable for "no motion," but ensure that `_get_velocity_vector` doesn’t return `(0,0)` due to *missing data* vs. *actual stop*. Currently, it returns `(0,0)` for both. Consider returning `None` for invalid data and handling it separately to distinguish "stationary vehicle" from "bad data."

3.  **Angle Gap (30°–60° and 120°–150°)**:
    *   **Issue**: Angles between 30°–60° and 120°–150° fall into `else: return "other"`.
    *   **Fix**: These are valid conflict geometries (e.g., merging, cutting in). Consider expanding thresholds or adding a "merging" category. If "other" is intended, document why these ranges are excluded.

### Missing Contracts & Robustness

4.  **No Validation of `conflict_frame`**:
    *   **Issue**: If `conflict_frame` is outside the range of available frames in `traj_a_json` or `traj

## src/analysis/gate_counter.py (part 1/5)
Here is the review for `src/analysis/gate_counter.py` (Part 1/5).

### **Critical Bugs**

1.  **`VirtualGate.check_crossing`: Mutates state before validation**
    *   **Issue:** `self.history[track_id] = frame_idx` is updated *before* the direction logic determines if it's an entry or exit. If the logic changes or if you add a "ignore" condition later, the history is already polluted. More importantly, if `entry_side` is invalid (see below), the count doesn't increment, but the history *does*, potentially blocking future valid crossings for `min_frames_between_crossings` frames.
    *   **Fix:** Move `self.history[track_id] = frame_idx` to the very end, only after a valid "entry" or "exit" is confirmed and counted.

2.  **`VirtualGate.check_crossing`: Invalid `entry_side` handling**
    *   **Issue:** The `else` block assumes `entry_side` is "right". If `entry_side` is "up", "down", or a typo, it silently treats it as "right".
    *   **Fix:** Validate `entry_side` in `__post_init__` or raise a `ValueError` in `check_crossing` if it's not "left" or "right".

3.  **`RobustTracker.update`: Logic Incomplete/Broken**
    *   **Issue:** The code snippet ends abruptly at `if missed <= self.max_missing:`. The logic for handling missed tracks (incrementing `missed`, updating `last_seen_frame`, or dropping tracks) is missing. As it stands, this method will crash or behave unpredictably if a track is missed.
    *   **Fix:** Complete the loop to handle missed tracks correctly.

4.  **`RobustTracker.update`: Mutates Input Detections**
    *   **Issue:** `det["track_id"] = tid` and `det["prev_centroid"] = ...` modify the input `detections` list in-place. This is a side effect that can cause bugs if the caller reuses the detection list or expects it to be immutable.
    *   **Fix:** Create a copy of the detection dict before modifying it, or return new objects.

### **Missing Contracts & Robustness**

## src/analysis/gate_counter.py (part 2/5)
Here is the review for `src/analysis/gate_counter.py` (part 2/5).

### **Critical Bugs**

1.  **Truncated Method Definition (Syntax Error)**
    The file ends abruptly inside `_normalize_class_name`:
    ```python
    return str(cls_name).strip().l
    ```
    This is a syntax error. It likely intended to be `.lower()`.
    **Fix:** Complete the method:
    ```python
    return str(cls_name).strip().lower()
    ```

2.  **Inconsistent Class Normalization**
    In `__post_init__`, you build `_class_whitelist` using `_normalize_class_name`. However, the default `classes_of_interest` list contains both `"auto rickshaw"` and `"auto-rickshaw"`.
    If `_normalize_class_name` only does `.strip().lower()`, these remain distinct entries in the set. If your detector returns `"auto rickshaw"` but you check against a normalized set that might have been built differently elsewhere (or if you intend to treat them as identical), this logic is fragile.
    **Recommendation:** Ensure `_normalize_class_name` handles hyphens/spaces consistently if you want to treat `auto-rickshaw` and `auto rickshaw` as the same class. Otherwise, document that they are distinct.

3.  **Potential `IndexError` in `load_gates`**
    ```python
    start = tuple(g.get("start", [0, 0]))
    end = tuple(g.get("end", [0, 0]))
    # ...
    p1=(int(start[0]), int(start[1])),
    ```
    If the YAML provides `start: [100]` (a list with one element), `start[1]` will raise an `IndexError`.
    **Fix:** Validate the length of `start` and `end` tuples before accessing indices.

### **Logic & Design Issues**

4.  **Silent Track Dropping**
    In the first snippet:
    ```python
    # else: drop track silently
    ```
    Dropping tracks without logging or metrics can make debugging difficult. If a track is dropped due to low confidence or missed detections, consider logging at `DEBUG` level or incrementing a `dropped_tracks

## src/analysis/gate_counter.py (part 3/5)
Here is the review for `src/analysis/gate_counter.py` (Part 3/5).

### Bugs & Logic Errors

1.  **`_draw_stats_panel`: Incorrect `cv2.getTextSize` unpacking**
    *   **Issue:** `cv2.getTextSize` returns a tuple `(size, baseline)`. The code does `(tw, _), _ = ...`. This is correct *if* `tw` is the width. However, `cv2.getTextSize` returns `(width, height)` as the first element. So `tw` is width. This part is actually okay.
    *   **Wait, let's look closer:** `cv2.getTextSize` returns `((width, height), baseline)`.
    *   Code: `(tw, _), _ = cv2.getTextSize(...)`
    *   This unpacks `tw` as width and `_` as height. This is **correct**.
    *   **However**, `max(text_widths, default=220)` is used. If `lines` contains empty strings, they are skipped in the loop, so `text_widths` might be empty if all lines are empty (unlikely, but possible). The `default` handles this.
    *   **Real Bug:** `panel_w = min(max(text_widths, default=220) + 26, w - 20)`. If `w` is small (e.g., < 20), `w - 20` is negative. `min` will pick the negative value, causing `x1` to be greater than `x2`, leading to invalid rectangle coordinates or errors in `cv2.rectangle`.
    *   **Fix:** Ensure `panel_w` is at least a minimum positive value, or clamp `x1` to be less than `x2`.

2.  **`_draw_stats_panel`: Potential `IndexError` or Logic Error in Gate Color Lookup**
    *   **Issue:** `gate_name = text.split(":")[0]`. If the gate name itself contains a colon (e.g., `Gate:1`), this splits incorrectly.
    *   **Fix:** Use a more robust parsing method or store the gate name in a separate data structure during the loop construction.

3.  **`_draw_tracks`: No bounds checking for `x,

## src/analysis/gate_counter.py (part 4/5)
Here is the review for `src/analysis/gate_counter.py` (Part 4/5).

### 1. Bugs & Logic Errors

*   **Critical: `plt.pause` in Headless/Non-Interactive Environments**
    *   **Issue:** `plt.pause(0.01)` blocks execution and requires a GUI backend. If this code runs in a headless server, CI pipeline, or non-interactive Jupyter kernel, it will hang or crash.
    *   **Fix:** Guard the preview logic with `if preview_interval is not None and plt.get_backend().startswith('TkAgg')` or similar, or better yet, use `matplotlib.use('Agg')` and skip plotting entirely if no display is available. Alternatively, use `cv2.imshow` if the goal is simple debugging, but `plt` is heavy for per-frame updates.
*   **Logic: `max_frames` Check Placement**
    *   **Issue:** The check `if max_frames is not None and frame_idx > max_frames: break` happens *after* `frame_idx += 1` but *before* processing the current frame.
    *   **Impact:** If `max_frames=10`, the loop reads frame 0, increments to 1, processes frame 0. ... Reads frame 9, increments to 10, processes frame 9. Reads frame 10, increments to 11, breaks. This processes 10 frames (0-9). This is likely correct, but the logic is fragile.
    *   **Improvement:** It is cleaner to check `if frame_idx >= max_frames: break` at the *start* of the loop or use a `for` loop with `range(max_frames)` if `max_frames` is known.
*   **Potential Bug: `total_frames` Calculation**
    *   **Issue:** `total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))` can be 0 or -1 for some video codecs/containers.
    *   **Impact:** `tqdm(total=0)` will behave unexpectedly or show 0% progress indefinitely if the video has frames but metadata is missing.
    *   **Fix:** If `total_frames <= 0`, set `total_frames = None` or a large number to disable the progress bar, or log a warning

## src/analysis/gate_counter.py (part 5/5)
Here is the review for `src/analysis/gate_counter.py` (Part 5/5).

### 1. Bugs & Logic Errors

*   **Truncated Code / Syntax Error**: The provided snippet starts mid-line (`es", {}).items():`). This suggests the input was truncated. Assuming the full line is `for name, stats in ... .items():`, the logic is likely correct, but **verify the source dictionary structure**. If `stats` is not a dict, `.get()` will fail.
*   **Missing Error Handling for `to_csv`**: `df.to_csv()` can raise `PermissionError` or `OSError` if the path is invalid or read-only. This should be handled or explicitly documented as a hard failure.

### 2. Missing Contracts & Robustness

*   **No Validation of `path`**: `path` is assumed to be a `Path` object. If a string is passed, `path.parent` will fail. Add a type check or convert: `path = Path(path)`.
*   **Empty DataFrame Handling**: If `rows` is empty, `pd.DataFrame(rows)` creates an empty DataFrame. `to_csv` will write only the header. This is usually acceptable, but ensure downstream consumers handle empty CSVs correctly.
*   **Missing Return Value**: The function likely returns `None`. Consider returning the `path` or the `DataFrame` for chaining or verification.

### 3. Style & Best Practices

*   **Magic Strings**: `"gate"`, `"entries"`, `"exits"` are hardcoded. Define constants or use a dataclass/TypedDict for the row structure to improve maintainability.
*   **Redundant `.get()` Defaults**: If the source data is guaranteed to have these keys, `.get()` with defaults hides bugs. If not, ensure the defaults are appropriate (e.g., `0` for counts).
*   **Logging**: Add logging for the file write operation (e.g., `logger.info(f"Wrote gate stats to {path}")`).

### 4. Potential Improvements

*   **Use `pathlib` Consistently**: Ensure `path` is a `Path` object at the start of the function.
*   **Atomic Write**: For production systems, consider writing to a temporary file and then renaming to avoid partial writes if the process crashes.
*   **Type

## src/analysis/grid_trajectory/__init__.py (part 1/1)
Here is the review of `src/analysis/grid_trajectory/__init__.py`:

### 1. Critical Bug: Incorrect `__all__` Construction
The current logic for building `__all__` is flawed and fragile.
```python
__all__.extend([n for n in globals() if not n.startswith("_")])
```
*   **Problem:** This captures **all** global names, including `logging`, `annotations`, `__name__`, `__doc__`, `__file__`, `__package__`, `__loader__`, `__spec__`, and the `try/except` variables (`exc`). It does not just capture the imported symbols.
*   **Consequence:** `from .grid_trajectory import *` will pollute the namespace with internal module attributes and logging objects.
*   **Fix:** Explicitly define `__all__` in the submodules (`pet_grid.py`, `spatial_grid.py`) and import them directly.

### 2. Style & Best Practice: Avoid `import *`
*   **Problem:** Using `from .module import *` is discouraged in PEP 8 because it makes it unclear where names come from and can cause namespace collisions.
*   **Fix:** Import specific names or import the submodules directly.

### 3. Missing Contract: No Explicit `__all__` Definition
*   **Problem:** The package does not explicitly declare its public API. Relying on dynamic `globals()` inspection is brittle.
*   **Fix:** Define `__all__` explicitly at the top of the file.

### 4. Minor: Redundant Deduplication
*   **Problem:** `dict.fromkeys(__all__)` is used to remove duplicates. If you fix the `__all__` construction (see above), duplicates are unlikely unless submodules share names. If they do, it’s better to handle conflicts explicitly.

### 5. Logging: `exc` Variable Unused
*   **Problem:** In the `except ImportError as exc` blocks, `exc` is assigned but never used. `_logger.exception()` automatically includes the traceback.
*   **Fix:** Remove `as exc` or use `exc` in the log message if needed.

---

### Recommended Refactor

```python
from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)

## src/analysis/grid_trajectory/pet_grid.py (part 1/3)
Here is the review for `src/analysis/grid_trajectory/pet_grid.py` (Part 1/3).

### 1. Bugs & Logic Errors

*   **Incomplete `build_intervals` Logic (Critical):**
    The method `build_intervals` is cut off at `prev_cell: Any = None`. It lacks the loop to iterate through samples, detect cell changes, and construct `Interval` objects. As it stands, it returns an empty list.
*   **Type Mismatch in `build_intervals`:**
    The method signature returns `list[IntervalType]`. If `CORE_TYPES_AVAILABLE` is `True`, `IntervalType` is `CoreInterval`. However, the local `Interval` dataclass is defined with specific fields (`obj_id`, `cell_id`, `t_enter`, `t_exit`, `world_samples`). If `CoreInterval` has a different structure or constructor signature, instantiating it inside `build_intervals` (once implemented) will likely fail or behave unexpectedly. The code does not show how local data is mapped to core types.
*   **Missing Time Calculation:**
    `Interval` requires `t_enter` and `t_exit` (floats). The `log` method stores `frame_idx` (int). The conversion `t = frame_idx / fps` is not visible in the provided snippet. Ensure this conversion happens in `build_intervals` and that `fps` is accessible there.

### 2. Missing Contracts & Robustness

*   **No Validation for `track_id` or `frame_idx`:**
    `log()` casts `track_id` and `frame_idx` to `int` but does not check for negative values. Negative frame indices are invalid in this context.
*   **`cell_id` Type Ambiguity:**
    `cell_id` is typed as `Any`. This makes it impossible to validate consistency (e.g., ensuring it's a string or int) and complicates downstream logic. Consider defining a `CellID` type alias or using a specific type.
*   **No Check for Duplicate Logs:**
    If `log()` is called twice for the same `(track_id, frame_idx)`, it appends a duplicate. This may lead to incorrect interval boundaries or double-counting. Consider raising an error or ignoring duplicates.
*   **`PETEvent.severity` Validation:**
    The `severity` field

## src/analysis/grid_trajectory/pet_grid.py (part 2/3)
Here is the review for `src/analysis/grid_trajectory/pet_grid.py` (Part 2/3).

### Critical Bugs

1.  **Inconsistent Downsampling Logic (State Desync)**
    *   **Location:** Inside the `for frame_idx, cell_id, wx, wy in samples:` loop.
    *   **Issue:** When `cell_id == prev_cell`, you increment `sample_counter` *only* if `wx` and `wy` are not `None`. However, when starting a *new* interval (`else` branch), you reset `sample_counter = 0` and then immediately increment it if `wx/wy` exist.
    *   **Consequence:** The downsampling logic (`sample_counter % self.downsample_every == 0`) behaves inconsistently between the first sample of a new interval and subsequent samples. More critically, if `wx` is `None` for a sample in the *same* cell, `sample_counter` is **not** incremented. This means the "every Nth sample" logic is actually "every Nth *valid* sample," which might be intended, but the reset logic in the `else` branch forces the first sample of a new interval to always be captured (since counter starts at 0, `0 % N == 0`). This creates a bias where interval boundaries are always sampled, while interior samples are downsampled. If this is unintended, it skews trajectory density.
    *   **Fix:** Clarify intent. If you want strict temporal downsampling, increment `sample_counter` for *every* frame, regardless of `wx/wy` validity. If you want valid-sample downsampling, ensure the reset logic aligns with the accumulation logic.

2.  **Potential `ZeroDivisionError` in `get_stats`**
    *   **Location:** `get_stats` method.
    *   **Issue:** `total_samples / max(len(self.tracks), 1)` is safe, but if `self.tracks` is empty, `total_samples` is 0. The division is safe due to `max(..., 1)`. However, the logic is slightly redundant.
    *   **Fix:** No bug, but see Style below.

3.  **Truncated Code / Syntax Error**
    *   **Location:** End of `summarize_pet`.
    *   **

## src/analysis/grid_trajectory/pet_grid.py (part 3/3)
Here is the review for `src/analysis/grid_trajectory/pet_grid.py` (Part 3/3).

### 1. Critical Logic Bug: Dead Code / Impossible Condition
**Issue:** The `elif` branch (Case 2) is logically unreachable given the sorting order.
*   **Analysis:** `sorted_intervals` is sorted by `t_enter`. Therefore, for any `i < j`, `A.t_enter <= B.t_enter`.
*   **Condition:** `B.t_exit <= A.t_enter`.
*   **Contradiction:** Since `B.t_exit >= B.t_enter` (assuming valid intervals where exit >= enter), and `B.t_enter >= A.t_enter`, it is impossible for `B.t_exit` to be less than or equal to `A.t_enter` unless the interval is zero-length or negative, which is typically invalid. Even if `B.t_enter == A.t_enter`, `B.t_exit` must be `>= A.t_enter`.
*   **Impact:** The `# pragma: no cover` suggests you already suspect this, but it indicates a misunderstanding of the "both directions" requirement.
*   **Fix:** You cannot detect "B exits before A enters" by just looking at `t_enter` sorted pairs where `i < j`.
    *   If you want to detect B->A transitions, you need to check if `B.t_exit < A.t_enter`. But since `A.t_enter <= B.t_enter`, this implies `B.t_exit < A.t_enter <= B.t_enter`, which means `B.t_exit < B.t_enter` (invalid interval).
    *   **Conclusion:** In a sorted-by-enter-time list, you can **only** detect A->B transitions (A exits before B enters). To detect B->A, you would need to iterate differently or realize that if `B.t_exit < A.t_enter`, then B must have entered *before* A, contradicting the sort order `i < j` (where A enters first).
    *   **Action:** Remove Case 2 entirely. It is dead code. If you truly need bidirectional detection, your sorting strategy or pair iteration logic is fundamentally flawed for this specific metric definition.

### 2. Missing Input Validation
**Issue:** No validation for `critical_threshold` or `moderate_threshold`.
*   **Risk:** If `critical_threshold >

## src/analysis/grid_trajectory/sam3_grid_pet.py (part 1/3)
Here is the review for `src/analysis/grid_trajectory/sam3_grid_pet.py` (Part 1/3).

### Critical Issues & Bugs

1.  **Import Order & Side Effects**:
    *   `import numpy as np` appears *after* the `overlay_pet_info` function definition. While Python allows this, it is highly non-standard and confusing. Move all imports to the top of the file.
    *   `cv2` is imported at the top but used in `overlay_pet_info`. Ensure `cv2` is available before the function is called (it is, but the structure is messy).

2.  **Silent Failure of SAM3 Import**:
    ```python
    try:
        from ultralytics.models.sam import SAM3VideoSemanticPredictor
    except Exception:
        SAM3VideoSemanticPredictor = None
    ```
    *   **Bug**: If `SAM3VideoSemanticPredictor` is `None`, the code will fail later with a confusing `TypeError` or `AttributeError` when trying to instantiate it.
    *   **Fix**: Raise a clear `ImportError` or `RuntimeError` at the start of `run_sam3_grid_pet` if `SAM3VideoSemanticPredictor is None`.

3.  **Missing Type Hints for `frame`**:
    *   `overlay_pet_info(frame, ...)` lacks a type hint for `frame`. It should be `np.ndarray`.

4.  **Unused Imports**:
    *   `json`, `time`, `logging` (if not used in this part), `Union` (if `X | Y` is preferred), `Any` (used in `LoggerType` but `LoggerType` is redundant).
    *   `LoggerType = Union[logging.Logger, Any]` is effectively `Any`. Remove this alias and just use `logging.Logger | None`.

### Style & Contract Issues

5.  **Redundant Return in `overlay_pet_info`**:
    *   `cv2.putText` modifies the image in-place. The `return frame` is unnecessary unless you intend to chain calls, but even then, it’s misleading. Remove it or document that it returns the same object.

6.  **Magic Numbers in `overlay_pet_info`**:
    *   `y = 30`, `y +=

## src/analysis/grid_trajectory/sam3_grid_pet.py (part 2/3)
Here is a concise review of the provided code snippet.

### 1. Bugs & Critical Issues

*   **Resource Leak on Early Exit:**
    If `cap.read()` fails or an exception occurs before the `try` block, `cap` is released, but if an exception occurs *inside* the `try` block (e.g., during `predictor` initialization or the loop), `cap` and `writer` are **never released**.
    *   *Fix:* Wrap the entire processing logic in a `try...finally` block to ensure `cap.release()` and `writer.release()` (if not None) are called.

*   **Inconsistent Frame Counting with Stride:**
    `frame_count` is incremented *after* the stride check. This means `frame_count` represents "processed frames," not "video frames." However, `frame_idx` from `enumerate` is the raw video index. If `max_frames` is intended to limit *processed* frames, the logic `if max_frames is not None and frame_idx >= max_frames` is incorrect because `frame_idx` grows by `frame_stride` effectively.
    *   *Fix:* If `max_frames` limits processed frames, compare against `frame_count`. If it limits video frames, the current logic is okay but confusing. Clarify intent.

*   **Potential `None` Type Error in `overlay_pet_info`:**
    In the `continue` branch where `boxes` is `None`, `frame` is used. If `res.orig_img` was `None` (unlikely but possible in error states), `.copy()` would fail earlier. However, if `overlay_pet_info` expects specific metadata that isn't available in this "skip" path, it might crash. Ensure `overlay_pet_info` handles minimal data.

*   **Truncated Code:**
    The code ends abruptly at `frame = overlay_pe`. This is likely a copy-paste error in the snippet, but in the actual file, ensure this line is complete (e.g., `overlay_pet_info(...)`).

### 2. Missing Contracts & Validation

*   **`concepts` Validation:**
    If `concepts` is provided, it should be validated to be a non-empty list of strings. Currently, if an empty list is passed, SAM3 might behave unexpectedly or silently do nothing.
    *   *Fix

## src/analysis/grid_trajectory/sam3_grid_pet.py (part 3/3)
Here is the review for `src/analysis/grid_trajectory/sam3_grid_pet.py` (Part 3/3).

### 1. Bugs & Logic Errors

*   **`cv2.putText` Positioning Bug**:
    ```python
    (x1, max(y1 - 5, 0))
    ```
    `cv2.putText` expects the **bottom-left** corner of the text bounding box. If `y1` is near the top of the frame (e.g., `y1=10`), `y1-5` is `5`. The text will be drawn *above* the box, which is correct. However, if `y1` is small (e.g., `y1=2`), `max(2-5, 0)` is `0`. The text baseline is at `y=0`, meaning the text is cut off at the top of the frame.
    **Fix**: Ensure the text is fully visible. A safer approach is to check if `y1 < 20` and place the text *inside* the box or below it, or simply clamp the y-coordinate to ensure the text height fits.
    ```python
    text_y = max(y1 - 5, 15) # Ensure at least 15px from top for text height
    ```

*   **`world_xy` Handling Inconsistency**:
    ```python
    world_xy = bev_mapper.pixel_to_world((cx, cy))
    if world_xy is not None:
        wx, wy = world_xy
    else:
        wx, wy = None, None
    ```
    If `bev_mapper.pixel_to_world` returns a tuple of `None` (e.g., `(None, None)`), this logic fails. It assumes `None` only if the entire return is `None`.
    **Fix**: Verify the contract of `pixel_to_world`. If it can return `(None, None)`, handle it explicitly:
    ```python
    world_xy = bev_mapper.pixel_to_world((cx, cy))
    if world_xy is None or world_xy[0] is None:
        wx, wy = None, None
    else:
        wx, wy = world_xy
    ```

*   **`track_ids` Type Mismatch Risk**:
    ```python


## src/analysis/grid_trajectory/spatial_grid.py (part 1/2)
Here is the review for `src/analysis/grid_trajectory/spatial_grid.py` (Part 1).

### Bugs & Logic Errors

1.  **`get_cell_from_pixels`: Integer Division Truncation vs. Rounding**
    *   **Issue:** Using `//` (floor division) on floats can lead to off-by-one errors if `px_x` or `px_y` are slightly negative relative to the origin due to floating-point precision, or if the grid origin isn't perfectly aligned. More critically, if `px_x` is exactly `x_max`, `col_idx` might calculate to the number of columns, which is out of bounds for a 0-indexed array if the grid width isn't a multiple of `cell_size`.
    *   **Fix:** Ensure `col_idx` and `row_idx` are clamped to the valid range `[0, max_cols-1]` and `[0, max_rows-1]` before formatting. Currently, you only clamp to `0`.
    *   **Example:** If grid width is 100px, `cell_size` is 40px. Valid cols are 0, 1, 2. If `px_x = 100`, `(100-0)//40 = 2`. This is valid. But if `px_x = 100.0000001` (due to float error) and `x_max` is strictly checked, it returns `OUT_OF_BOUNDS`. If `x_max` is inclusive, it might map to col 2. Ensure the boundary logic matches the physical grid definition.

2.  **`get_cell_center`: Incomplete Implementation (Truncated Code)**
    *   **Issue:** The code cuts off at `parts = cell_id.split("_")`. The logic to parse `col_letter` and `row_num` back into indices and calculate pixel centers is missing.
    *   **Risk:** If this is part 1/2, ensure the parsing logic handles the `naming_style` dynamically. Hardcoding `split("_")` assumes the prefix is always one character and separators are underscores. If `naming_style` is `"G_{col}_{row}"`, this works. If it were `"Cell_{col}_{row}"`, `split("_")` would yield `["Cell", "A", "3"]

## src/analysis/grid_trajectory/spatial_grid.py (part 2/2)
Here is the review for `src/analysis/grid_trajectory/spatial_grid.py` (Part 2/2).

### 1. Bugs & Logic Errors

*   **Column Label Overflow (>26 columns):**
    In `draw_overlay`, `label = chr(65 + (i % 26))` causes columns 27+ to repeat labels (A, B, ... Z, A, B...). This creates ambiguity in the grid visualization.
    *   *Fix:* Implement a helper to generate Excel-style column headers (A, B, ..., Z, AA, AB...) or limit the grid size.
*   **Row Label Misalignment:**
    In `draw_overlay`, the row label position is `pos = (max(5, self.x_min - 45), y + 35)`. The `+35` offset is arbitrary and likely misaligns the text vertically with the grid line, especially if `cell_size` varies.
    *   *Fix:* Use `y + (self.cell_size // 2)` or `y + 10` (depending on font size) to center the text vertically on the line.
*   **Integer Division in Center Calculation:**
    In `get_cell_center`, `self.cell_size // 2` is used for the offset. If `cell_size` is odd, this truncates the center. While minor, using `self.cell_size / 2` and casting to `int` at the end is more precise for pixel alignment.
    *   *Note:* The current code does `int(x)` at the end, which is fine, but the intermediate `//` is inconsistent with floating-point geometry expectations.
*   **Missing Bounds Check in `get_cell_bounds`:**
    `get_cell_bounds` relies on `get_cell_center`. If `cell_id` is valid but the resulting bounds exceed the image frame (e.g., due to `x_min`/`y_min` being negative or `x_max`/`y_max` exceeding frame size), `cv2.rectangle` in `draw_overlay` will clip or fail silently.
    *   *Fix:* Ensure `get_cell_bounds` clamps coordinates to the frame dimensions if the grid extends beyond the image, or document that the grid must fit within the frame.

### 2. Missing Contracts & Type Safety

*   **

## src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py (part 1/7)
Here is the review for Part 1/7 of `src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py`.

### Critical Issues

1.  **Truncated Code / Syntax Error**
    The file ends abruptly inside the `_box_intersection` function:
    ```python
    x2 = min(ax2,
    ```
    This will cause an immediate `SyntaxError` upon import. Ensure the file is complete.

2.  **Bare `except` in `_compute_histogram`**
    ```python
    except Exception:
        return None
    ```
    Catching `Exception` silently swallows all errors (including `MemoryError`, `KeyboardInterrupt` if not handled properly, or logic bugs).
    *   **Fix:** Catch specific exceptions (e.g., `cv2.error`, `ValueError`) or log the exception before returning `None`. Silent failures in data processing pipelines make debugging extremely difficult.

3.  **Inconsistent Class ID Mapping**
    `CLASS_NAME_TO_ID` maps `"bike"` to `3` and `"motorcycle"` to `3`. However, `UVH_DISPLAY_MAP` maps `"Two-wheeler"` to `"bike"`.
    *   **Risk:** If the model outputs `"motorcycle"` but the display logic expects `"bike"`, or if downstream logic relies on strict ID uniqueness for distinct physical classes, this aliasing might cause confusion. Ensure this is intentional (i.e., bikes and motorcycles are treated as the same class for analysis).

### Logic & Robustness

4.  **`_segment_intersection` Edge Cases**
    *   **Collinear Overlap:** The function returns `None` if segments are collinear and overlapping (`abs(rxs) < 1e-9 and abs(qpxr) < 1e-9`). For trajectory conflict detection, overlapping collinear segments *are* a conflict. Returning `None` here may miss valid conflicts.
    *   **Floating Point Precision:** Using `1e-9` for epsilon is generally fine, but ensure `t` and `u` checks (`0.0 <= t <= 1.0`) account for floating-point drift. Consider using `np.isclose` or a small epsilon in the bounds check.

5.  **`_compute_pet_from_windows` Logic**
    *   The function assumes

## src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py (part 2/7)
Here is the review for `src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py` (Part 2/7).

### **Bugs & Logic Errors**

1.  **`_get_entry_gate`: Missing Zero-Crossing Handling**
    *   **Issue:** The condition `if side_prev * side_curr < 0` fails if a point lies exactly on the line (`side == 0`). If `side_prev == 0`, the product is `0`, and the crossing is missed.
    *   **Fix:** Use `if (side_prev < 0 and side_curr > 0) or (side_prev > 0 and side_curr < 0):` or handle zero explicitly.

2.  **`_load_gates`: Silent Failure & Type Safety**
    *   **Issue:** `g["start"]` and `g["end"]` are assumed to be lists/tuples. If they are strings or malformed, `tuple()` will create a tuple of characters, leading to cryptic errors later in `_line_side`.
    *   **Fix:** Validate that `g["start"]` and `g["end"]` are sequences of length 2 containing numbers.

3.  **`run_uvh_coco_fused_grid_pet`: Redundant/Confusing Device Logic**
    *   **Issue:**
        ```python
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # ... later ...
        if backend == "auto":
            if torch.cuda.is_available():
                device = "cuda:0" if device == "auto" else device # device is no longer "auto" here
        ```
        The second check `device == "auto"` is dead code because `device` was already resolved at the top. This is confusing and suggests the logic was refactored incompletely.
    *   **Fix:** Resolve `device` once at the top. Remove the redundant check inside the `backend` block.

4.  **`run_uvh_coco_fused_grid_pet`: Swallowing Exceptions**
    *   **Issue:** `except Exception: gates = []` and `except Exception: spatial_grid = None` hide configuration errors. If the YAML is malformed, the user gets no feedback,

## src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py (part 3/7)
Here is the review for `src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py` (Part 3/7).

### Critical Bugs & Logic Errors

1.  **Shadowing Built-in `del`**:
    *   **Issue**: `del = YOLO(...)` shadows the Python built-in `del` statement. While syntactically valid in this scope, it is dangerous and confusing. If you later try to delete a variable in this scope, it will fail or behave unexpectedly.
    *   **Fix**: Rename to `uvh_model`.

2.  **Inconsistent Model Initialization Logic**:
    *   **Issue**: The `if/else` block initializes models differently based on a condition (likely `use_openvino`), but the variable names `uvh_model` and `coco_model` are reused. However, the `else` branch assigns to `uvh_model` and `coco_model`, while the `if` branch assigns to `del` and `coco_model`. This creates a mismatch where `uvh_model` might be undefined if the `if` branch is taken, or `del` is undefined if the `else` branch is taken.
    *   **Fix**: Ensure consistent variable naming. Use `uvh_model` in both branches.

3.  **Potential `None` Access on `orig_img`**:
    *   **Issue**: `uvh_r.orig_img` and `coco_r.orig_img` are accessed directly. If the model fails to load the image or returns a result without `orig_img` (e.g., due to an error in the stream), this will raise an `AttributeError`.
    *   **Fix**: Add a check: `if uvh_r.orig_img is not None:` before computing histograms.

4.  **Race Condition / State Management in Tracker**:
    *   **Issue**: The code snippet ends with `matched = custom_tra...` (truncated). If `CustomTracker` maintains internal state (like track IDs), ensure that the tracker is reset or properly initialized for each video run. If this function is called multiple times, the tracker might retain stale tracks from previous runs.
    *   **Fix**: Ensure `CustomTracker` is instantiated fresh for each call to this function (which it appears to be, but verify no global state is used inside `Custom

## src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py (part 4/7)
Here is the review for `src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py` (Part 4/7).

### 🚨 Critical Bugs & Logic Errors

1.  **Broken Track ID Counting Logic**
    *   **Location:** `print(f"[UVH-COCO] Tracking splitter: original={len(set(tid // 1000 for tid in tracks if tid >= 1000) | {tid for tid in tracks if tid < 1000})}, ...")`
    *   **Issue:** This logic is nonsensical. It mixes integer division (`tid // 1000`) with raw IDs for `tid < 1000`. If `tid=1500`, it adds `1` to the set. If `tid=500`, it adds `500`. This does not count unique original tracks; it counts a mix of "object groups" and "individual low IDs".
    *   **Fix:** If the goal is to count unique *original* track IDs before splitting, you need a separate set maintained during the splitting process or a mapping. If the goal is just to show the number of tracks *before* splitting, you should have stored `len(tracks)` before calling `_split_tracks_by_gaps`.
    *   **Recommendation:** Store `original_track_count = len(tracks)` before the split call. Use that in the log.

2.  **Inefficient Interactive Pause (Re-opening Video)**
    *   **Location:** `cap = cv2.VideoCapture(video_path)` inside the loop.
    *   **Issue:** Opening and closing a `VideoCapture` object every 20 frames is extremely expensive and can cause race conditions or file handle exhaustion on some systems. It also ignores the current position of the main video reader if one exists.
    *   **Fix:** If you have access to the main video reader, use it. If not, consider using `cv2.VideoCapture` once outside the loop and seeking, or better yet, save the frame to a temporary file or display it from memory if the frame is already loaded. *Note: The code snippet shows `frame_img` is available earlier, but the interactive block re-reads from disk.*
    *   **Better Approach:** Use the `frame_img` variable that is already in scope

## src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py (part 5/7)
Here is the review for `src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py` (Part 5/7).

### 1. Bugs & Logic Errors

*   **Inconsistent ID Mapping in `conflict_type` and Gates:**
    The logic `pts_a if first_id == track_a_id else pts_b` is fragile and likely incorrect.
    *   `first_id` is derived from `pet_result` which returns placeholders `"a"` or `"b"`.
    *   If `first_placeholder == "a"`, `first_id` becomes `track_a_id`. The condition `first_id == track_a_id` is `True`, so it picks `pts_a`. This is correct.
    *   If `first_placeholder == "b"`, `first_id` becomes `track_b_id`. The condition `first_id == track_a_id` is `False`, so it picks `pts_b`. This is also correct.
    *   **However**, the variable names in the dictionary keys are `track_a` and `track_b`, but the values assigned are `first_id` and `second_id`. If `first_placeholder` was `"b"`, then `track_a` in the output dict will actually contain `track_b_id`. This is semantically confusing and likely a bug in the output schema expectation. The keys should probably be `track_first` and `track_second`, or the logic should ensure `track_a` always refers to the original `track_a_id` regardless of which one was "first" in the PET calculation.
    *   **Recommendation:** Rename output keys to `track_first` and `track_second` to avoid ambiguity, or explicitly map back to `track_a_id` and `track_b_id` in the output dict to maintain consistent semantics.

*   **Potential Division by Zero / Invalid FPS:**
    `fps` is used in division (`b_entry / fps`). If `fps` is 0 or `None`, this will crash. Ensure `fps` is validated upstream.

*   **`pet_time_based` Calculation Logic:**
    The calculation `float((b_entry / fps) - (a_exit / fps))` assumes `a_exit` and `b_entry` are frame indices. If `a_exit` is the *last* frame of track A and `b_entry` is the *first* frame of

## src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py (part 6/7)
Here is the review for `src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py` (Part 6/7).

### 1. Critical Bugs & Logic Errors

**A. ID Collision in `_split_tracks_by_gaps`**
The logic `new_id = tid * 1000 + current_sub` is fragile.
*   **Issue:** If `tid` is large (e.g., 1000), `1000 * 1000 + 0 = 1,000,000`. If another track `tid=1` splits into sub-track 0, it becomes `1000`. This is fine. However, if `tid=1` splits into sub-track 1000 (unlikely but possible if `current_sub` grows), it collides with `tid=1, sub=0` of `tid=1`? No, wait.
    *   Real Issue: If `tid` is not unique or if `tid` exceeds a certain magnitude, collisions are possible. More importantly, this **mutates the track ID space** silently. Downstream code expecting original track IDs will break.
    *   **Fix:** Use a tuple `(tid, sub_id)` or a UUID for internal processing, and map back to original IDs in the output if needed. Or, ensure `tid` is small enough that `tid * 1000` never collides with another `tid * 1000 + sub`. Given `tid` is an integer from a tracker, this is risky.

**B. Off-by-One / Boundary Condition in `_split_tracks_by_gaps`**
*   **Issue:** The loop `for i in range(1, len(pts))` checks the gap between `pts[i-1]` and `pts[i]`. If a split occurs at index `i`, the segment `pts[start_idx:i]` is saved. The next segment starts at `i`. This is correct.
*   **However:** The prediction logic uses `pts[i-2]` and `pts[i-1]`. If `i=1`, `i>=2` is false, so `pred_dist = inf`. This is correct.
*   **Subtle Bug:** If `gap == 0` (duplicate frames), `dt1

## src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py (part 7/7)
Here is the review for the provided code snippet.

### 1. Bugs & Logic Issues

*   **Missing Input Validation (Crash Risk):**
    *   `track_a["frames"]` and `track_b["frames"]` are accessed directly. If either list is empty, `track_a["frames"][0]` will raise an `IndexError`.
    *   `max_pet` and `fps` are assumed to be defined in the outer scope. If they are `None` or non-numeric, `int(max_pet * fps)` will fail.
*   **Integer Truncation Bias:**
    *   `int(max_pet * fps)` truncates towards zero. If `max_pet * fps` is `2.9`, it becomes `2`. Using `math.ceil` or `round` might be more appropriate depending on whether you want a strict upper bound or an average estimate.
*   **Magic Number:**
    *   The `+ 5` is a hardcoded buffer. This should be a named constant (e.g., `FRAME_BUFFER = 5`) to clarify its purpose and allow easy tuning.

### 2. Missing Contracts

*   **No Type Hints:** The function lacks type annotations for parameters and return type.
*   **No Docstring:** There is no documentation explaining what `max_pet`, `fps`, `track_a`, and `track_b` represent, or what the boolean return value signifies (e.g., "Returns True if tracks are temporally overlapping or close enough to be fused").
*   **Assumption of Sorted Frames:** The code assumes `track_a["frames"]` and `track_b["frames"]` are sorted in ascending order. If they are not, `min_a`/`max_a` will be incorrect. This should be documented or enforced.

### 3. Style & Readability

*   **Variable Naming:**
    *   `min_a`, `max_a`, `min_b`, `max_b` are generic. Consider `start_a`, `end_a`, `start_b`, `end_b` for clarity.
    *   `max_frames_diff` is slightly ambiguous. `max_temporal_gap` or `fusion_window_frames` might be clearer.
*   **Logic Clarity:**
    *   The return statement `not (min_b > max_a + max_frames_diff or min