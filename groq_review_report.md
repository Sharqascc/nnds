## src/vlm/vlm_enhanced_pipeline.py
### Part 1/3
Here is the review for Part 1/3 of the code.

### Critical Bugs

1.  **Syntax Error / Truncated Code**:
    The code ends abruptly at `vlm_result = self.vlm.analyz`. This is a syntax error. It appears the method `process_video` is incomplete.

2.  **Infinite Loop Risk**:
    The `while cap.isOpened():` loop relies on `cap.read()` returning `False` to break. While `cv2.VideoCapture` usually handles this, if the video file is corrupted or empty, `frame_count` might be 0 or invalid.
    -   `frame_interval = max(1, frame_count // max_frames)` will be `1` if `frame_count` is 0.
    -   The condition `frames_processed < max_frames * frame_interval` will be `0 < 30 * 1` (True).
    -   However, `cap.read()` will immediately return `False`, so the loop breaks. This is safe, but the logic is fragile. It's better to use a `for` loop or check `frame_count` explicitly.

3.  **Incorrect Frame Selection Logic**:
    The condition `if frames_processed % frame_interval == 0 and frames_processed < max_frames * frame_interval:` is flawed.
    -   `frames_processed` starts at 0.
    -   If `frame_interval` is 10, it processes frames 0, 10, 20, ...
    -   But `frames_processed` is incremented *after* the check? No, looking closely, `frames_processed` is **never incremented** in the provided snippet.
    -   **Bug**: `frames_processed` is initialized to 0 but never incremented inside the loop. This means:
        -   The condition `frames_processed % frame_interval == 0` is

### Part 2/3
Here is a concise code review focused on correctness, maintainability,

### Part 3/3
Here is the review for Part 3/3.

### Critical Issues

1.  **Undefined Variable `o`**:
    *   The line `o(video, max_frames=30)` references a variable `o` that is not defined in this snippet. It is likely a typo for `pipeline` or a specific method call (e.g., `pipeline.process_video`). This will cause a `NameError` at runtime.
    *   **Fix**: Replace `o` with the correct object/method name.

2.  **Undefined Variable `video`**:
    *   The variable `video` is used but not defined in this snippet. Ensure it is properly initialized in the preceding code (Part 1 or 2).

3.  **Undefined Variable `pipeline`**:
    *   `pipeline` is used but not defined in this snippet. Ensure it is instantiated earlier in the script.

4.  **Undefined Variable `pd`**:
    *   `pd.DataFrame()` is used, but `pandas` is not imported in this snippet. Ensure `import pandas as pd` is present in the full script.

5.  **Undefined Variable `json`**:
    *   `json.dumps` is used, but `json` is not imported in this snippet. Ensure `import json` is present.

6.  **Undefined Variable `os`**:
    *   `os.path.exists` and `os.listdir` are used, but `os` is not imported in this snippet. Ensure `import os` is present.

### Maintainability & Style Issues

7.  **Hardcoded Paths**:
    *   The path `"uvh26_data/UVH-26-Train/data/000/"` is hardcoded. This reduces portability.
    *   **Fix**: Use a configuration file, environment variable, or command-line argument to specify the data

## src/vlm/utils/visualization.py
### Part 1/1
Here is the review of the provided code, focusing on correctness and maintainability.

### 1. Critical Bugs & Correctness

**A. `plt.tight_layout()` after `plt.suptitle()`**
*   **Issue:** Calling `plt.tight_layout()` after `plt.suptitle()` often causes the suptitle to overlap with the subplot titles or get clipped, because `tight_layout` does not account for the suptitle by default in older matplotlib versions, or it may squeeze the subplots too much.
*   **Fix:** Use `plt.tight_layout(rect=[0, 0, 1, 0.95])` or ensure the suptitle is added *after* `tight_layout` (though this is also fragile). A more robust approach is to use `fig.suptitle()` and then `fig.tight_layout(rect=[0, 0, 1, 0.95])`.

**B. `output_path` Type Inconsistency**
*   **Issue:** In `plot_vlm_results`, `output_path` is typed as `str | Path` and converted to `Path` internally. In `create_heatmap`, it is typed as `str` only. This inconsistency is confusing and less flexible.
*   **Fix:** Standardize to `str | Path` for all path arguments.

**C. `create_heatmap` is a Stub**
*   **Issue:** The function does nothing but print a message. If this is intended to be a placeholder, it should be explicitly marked as such (e.g., `raise NotImplementedError` or a clear docstring warning). As it stands, it silently fails to produce output, which is a bug in any pipeline expecting a file.
*   **Fix:** Either implement it or raise `NotImplementedError`.

**D. `pets` Data Type Assumption**
*   **Issue:** `r.get

## src/vlm/utils/image_utils.py
### Part 1/1
Here is a concise review of the code focusing on correctness and maintainability.

### 1. `extract_frames`

**Bugs & Correctness:**
*   **Silent Failure on Invalid `frame_interval`:** If `frame_interval` is `0`, this raises a `ZeroDivisionError` (or `ValueError` depending on Python version) inside the loop. It should be validated at the start.
*   **Inconsistent Return Type:** The docstring says `list[str]`, but `frame_paths` contains `str` objects. This is correct, but ensure callers expect strings, not `Path` objects.
*   **`print` Statement:** Using `print` for logging is not maintainable in libraries. Use `logging` module instead.

**Maintainability & Style:**
*   **Type Hints:** `str | Path` is fine for Python 3.10+, but consider using `Union[str, Path]` for broader compatibility if needed.
*   **Magic Number:** `05d` in `f"frame_{saved_count:05d}.jpg"` is a magic number. Consider making it a parameter or constant if flexibility is needed.
*   **Resource Management:** The `try/finally` block is good, but consider using a context manager if `cv2.VideoCapture` supported it (it doesn't natively, so this is acceptable).

### 2. `prepare_images_for_vlm`

**Bugs & Correctness:**
*   **Silent Skipping of Invalid Images:** If `cv2.imread` returns `None` (e.g., file not found, corrupt image), the function silently skips it. This can lead to unexpected behavior if the caller expects a 1:1 mapping or fails silently. It should either raise an exception or log a warning.
*   **Overwriting Input Files:** The output path is derived from the input