from pathlib import Path

UVH_PATH = Path(
    "src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py"
)


def test_uvh_exports_structured_pet_fields():
    source = UVH_PATH.read_text()

    expected_fields = [
        '"pet_s": structured_result.pet_s',
        '"pet_status": structured_result.pet_status',
        '"first_actor": structured_result.first_actor',
        '"second_actor": structured_result.second_actor',
        '"overlap_duration_s": structured_result.overlap_duration_s',
    ]

    for field in expected_fields:
        assert field in source
