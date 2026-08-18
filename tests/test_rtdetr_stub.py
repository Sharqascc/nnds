
import pytest
from src.pipeline.traffic_analyzer import run_video_to_pet

def test_rtdetr_detector_raises_not_implemented(tmp_path):
    video = tmp_path / "dummy.mp4"
    weights = tmp_path / "dummy.pt"
    bev = tmp_path / "bev.json"
    grid = tmp_path / "grid.json"

    video.touch()
    weights.touch()
    bev.write_text("{}")
    grid.write_text("{}")

    with pytest.raises(
        NotImplementedError,
        match="RT-DETR video pipeline is not implemented"
    ):
        run_video_to_pet(
            video_path=str(video),
            bev_config_path=str(bev),
            grid_config_path=str(grid),
            detector="rtdetr",
            rtdetr_weights_path=str(weights),
        )
