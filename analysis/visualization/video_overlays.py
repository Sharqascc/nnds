
import os
import cv2
from grid_trajectory.spatial_grid import SpatialGrid


def highlight_conflict_cell_on_frame(frame, grid: SpatialGrid, cell_id, color=(0, 0, 255)):
    """
    Highlight a given cell_id on top of a single RGB frame using a translucent rectangle.
    """
    center = grid.get_cell_center(cell_id)
    if center is None:
        return frame

    cx, cy = center
    half = grid.cell_size // 2

    x1, y1 = int(cx - half), int(cy - half)
    x2, y2 = int(cx + half), int(cy + half)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
    blended = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    return blended


def save_conflict_frame(video_path, grid_config_path, cell_id, frame_idx, out_path, alpha=0.6):
    """
    Save one video frame with:
      - spatial grid overlay, and
      - highlighted conflict cell.

    Parameters
    ----------
    video_path : str
        Path to the input video.
    grid_config_path : str
        Path to the GITI_grid_config.json used for SpatialGrid.
    cell_id : str
        Cell identifier (e.g., G_A_3) to highlight.
    frame_idx : int
        Frame index to capture (0-based).
    out_path : str
        Output image path.
    alpha : float
        Alpha for grid overlay blending.
    """
    grid = SpatialGrid(grid_config_path)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")

    # Draw grid, then highlight cell
    frame_grid = grid.draw_overlay(frame, alpha=alpha)
    frame_final = highlight_conflict_cell_on_frame(frame_grid, grid, cell_id)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, frame_final)
    return out_path
