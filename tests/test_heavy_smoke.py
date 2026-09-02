
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

@pytest.mark.smoke
@pytest.mark.integration
def test_sam3_grid_pet_smoke(tmp_path):
    from src.analysis.grid_trajectory.sam3_grid_pet import run_sam3_grid_pet

    # Create dummy required files so file existence checks pass
    (tmp_path / "video.mp4").write_bytes(b"dummy")
    (tmp_path / "sam3.pt").write_bytes(b"dummy")
    (tmp_path / "grid.json").write_text("{}")
    (tmp_path / "bev.json").write_text("{}")

    with patch('cv2.VideoCapture') as cap_mock:
        cap_mock.return_value.get.return_value = 25.0
        cap_mock.return_value.read.return_value = (False, None)
        with pytest.raises(RuntimeError):
            run_sam3_grid_pet(
                project_root=str(tmp_path),
                video_rel_path="video.mp4",
                sam3_rel_path="sam3.pt",
                grid_rel_path="grid.json",
                bev_rel_path="bev.json",
                output_name='test'
            )

@pytest.mark.smoke
@pytest.mark.integration
def test_yolo_cpu_grid_pet_smoke(tmp_path):
    from src.analysis.grid_trajectory.yolo_cpu_grid_pet import run_yolo_cpu_grid_pet

    # Create dummy required files
    (tmp_path / "video.mp4").write_bytes(b"dummy")
    (tmp_path / "yolo.pt").write_bytes(b"dummy")

    # Patch YOLO to avoid model load; patch cv2.VideoCapture to force RuntimeError
    with patch('src.analysis.grid_trajectory.yolo_cpu_grid_pet.YOLO', MagicMock()) as yolo_mock, \
         patch('cv2.VideoCapture') as cap_mock:
        cap_mock.return_value.isOpened.return_value = False
        with pytest.raises(RuntimeError):
            run_yolo_cpu_grid_pet(
                video_path=str(tmp_path / "video.mp4"),
                weights_path=str(tmp_path / "yolo.pt"),
                output_csv_path=str(tmp_path / "out.csv")
            )
