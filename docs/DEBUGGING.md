
# Debugging Guide for NNDS (Beginner-Friendly)

This guide helps new contributors debug the traffic analysis pipeline easily.

## 1. Enable Debug Mode

Add `--debug` to your command:

```bash
!python traffic_analyzer.py \
  --video sample_data/traffic_video.mp4 \
  --detector uvh-coco-fused \
  --uvh-model uvh26.pt \
  --coco-person-model yolo11n.pt \
  --out-csv outputs/pet_events.csv \
  --max-frames 100 \
  --interactive \
  --debug
```

In debug mode:
- Extra logs are printed.
- Intermediate artifacts (frames, detections, BEV maps) are saved to `outputs/debug/`.
- Errors show a friendly traceback summary.

## 2. Using Debug Helpers

In any Python cell or script:

```python
from utils.debug_helpers import debug_save_image, debug_save_df, debug_print_section, debug_print_tensor_info

# Print a clear section header
debug_print_section("After detection step", {"num_detections": len(dets)})

# Save an image
debug_save_image(frame, "frame_001.png", subdir="detections", show=True)

# Save a DataFrame
debug_save_df(detections_df, "detections.csv", subdir="detections")

# Inspect a tensor
debug_print_tensor_info("trajectory_batch", trajectory_tensor)
```

All outputs go to `/content/nnds/outputs/debug/`.

## 3. Common Issues

### No video/frame display in Colab

- Use `debug_save_image(..., show=True)` to display frames.
- Ensure `%matplotlib inline` is set at the top of your notebook.

### Script crashes with unclear error

- Run with `--debug`.
- Look for the `🔍 DEBUG: Error occurred` section.
- Check `outputs/debug/` for saved intermediate data.

### Want to pause and inspect every N frames?

- Use `--interactive` (already implemented).
- The script will ask every 20 frames: `Continue? [y/n]`.

## 4. Adding Your Own Debug Logs

Example inside a function:

```python
from utils.debug_helpers import debug_print_section, debug_save_image

def process_frame(frame, frame_idx):
    debug_print_section(f"Processing frame {frame_idx}")
    # ... your logic ...
    debug_save_image(frame, f"frame_{frame_idx:04d}.png", subdir="frames")
```

This keeps debugging consistent and beginner-friendly across the repo.
