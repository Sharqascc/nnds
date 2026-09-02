import pytest
from unittest.mock import patch

@pytest.mark.smoke
@pytest.mark.integration
def test_sam3_grid_pet_smoke(tmp_path):
    from src.analysis.grid_trajectory.sam3_grid_pet import run_sam3_grid_pet
    with patch('cv2.VideoCapture') as cap_mock:
        cap_mock.return_value.isOpened.return_value = False
        with pytest.raises(RuntimeError):
            run_sam3_grid_pet(
                project_root=str(tmp_path),
                video_rel_path=str(tmp_path / 'missing.mp4'),
                sam3_rel_path=str(tmp_path / 'sam3.pt'),
                grid_rel_path=str(tmp_path / 'grid.json'),
                bev_rel_path=str(tmp_path / 'bev.json'),
                output_name='test'
            )

@pytest.mark.smoke
@pytest.mark.integration
def test_yolo_cpu_grid_pet_smoke(tmp_path):
    from src.analysis.grid_trajectory.yolo_cpu_grid_pet import run_yolo_cpu_grid_pet
    with patch('cv2.VideoCapture') as cap_mock:
        cap_mock.return_value.isOpened.return_value = False
        with pytest.raises(RuntimeError):
            run_yolo_cpu_grid_pet(
                video_path=str(tmp_path / 'missing.mp4'),
                weights_path=str(tmp_path / 'yolo.pt'),
                output_csv_path=str(tmp_path / 'out.csv')
            )
