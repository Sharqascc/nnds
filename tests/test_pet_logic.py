
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))

from src.analysis.grid_trajectory.uvh_coco_fused_grid_pet import _compute_pet_from_windows
from src.analysis.grid_trajectory.spatial_grid import SpatialGrid

class TestPETLogic:
    def test_a_exits_before_b_enters(self):
        res = _compute_pet_from_windows(10, 20, 35, 45, fps=30)
        assert res is not None
        pet, first, second, frame_ref = res
        assert abs(pet - (35 - 20) / 30.0) < 1e-9
        assert first == "a"
        assert second == "b"
        assert frame_ref == 35

    def test_b_exits_before_a_enters(self):
        res = _compute_pet_from_windows(30, 40, 5, 15, fps=25)
        assert res is not None
        pet, first, second, frame_ref = res
        assert abs(pet - (30 - 15) / 25.0) < 1e-9
        assert first == "b"
        assert second == "a"
        assert frame_ref == 30

    def test_overlapping_windows_no_pet(self):
        res = _compute_pet_from_windows(10, 30, 20, 40, fps=30)
        assert res is None

    def test_zero_pet_not_returned(self):
        # If a_exit == b_entry, should compute 0 PET (but caller filters >0)
        res = _compute_pet_from_windows(10, 25, 25, 35, fps=30)
        assert res is not None
        pet, *_ = res
        assert pet == 0.0

class TestGridColumnNaming:
    def test_excel_style_helpers(self):
        from src.analysis.grid_trajectory.spatial_grid import _col_to_letters, _letters_to_col

        # Forward
        assert _col_to_letters(0) == "A"
        assert _col_to_letters(25) == "Z"
        assert _col_to_letters(26) == "AA"
        assert _col_to_letters(27) == "AB"
        assert _col_to_letters(29) == "AD"

        # Inverse
        assert _letters_to_col("A") == 0
        assert _letters_to_col("Z") == 25
        assert _letters_to_col("AA") == 26
        assert _letters_to_col("AB") == 27
        assert _letters_to_col("AD") == 29
