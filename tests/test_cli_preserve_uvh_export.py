from types import SimpleNamespace

import pandas as pd
import pytest

from src.pipeline import traffic_analyzer as cli


def prepare_dispatch(tmp_path, monkeypatch, backend):
    video = tmp_path / "video.mp4"
    weights = tmp_path / "model.pt"
    video.touch()
    weights.touch()
    monkeypatch.setattr(
        cli,
        "_import_either",
        lambda name: SimpleNamespace(run_uvh_coco_fused_grid_pet=backend),
    )
    return {
        "video_path": video,
        "detector": "uvh-coco-fused",
        "uvh_model_path": weights,
        "coco_person_model_path": weights,
        "out_csv_path": tmp_path / "pet.csv",
    }


@pytest.mark.parametrize("with_events", [False, True])
def test_uvh_backend_csv_is_preserved(tmp_path, monkeypatch, with_events):
    columns = [
        "event_id",
        "pet",
        "pet_s",
        "pet_status",
        "first_actor",
        "second_actor",
        "orig_track_a",
        "seg_a",
        "orig_track_b",
        "seg_b",
        "track_a_exit_time_sec",
        "track_b_entry_time_sec",
        "future_backend_field",
    ]
    row = [
        7,
        0.5,
        0.5,
        "sequential",
        "a",
        "b",
        12,
        1,
        34,
        2,
        1.0,
        1.5,
        "preserve-me",
    ]
    expected = pd.DataFrame([row] if with_events else [], columns=columns)
    written = {}

    def backend(**kwargs):
        path = kwargs["output_csv_path"]
        expected.to_csv(path, index=False)
        written["bytes"] = tmp_path.joinpath("pet.csv").read_bytes()
        # Deliberately incomplete: the CSV, not this list, owns the export.
        return {"pet_csv": path, "pet_events": []}

    kwargs = prepare_dispatch(tmp_path, monkeypatch, backend)

    def forbidden_writer(*args, **kwargs):
        pytest.fail("The generic writer must not overwrite the UVH CSV")

    monkeypatch.setattr(cli, "_write_events_to_csv", forbidden_writer)
    result = cli.run_video_to_pet(**kwargs)

    path = tmp_path / "pet.csv"
    assert path.read_bytes() == written["bytes"]
    assert list(result.columns) == columns
    pd.testing.assert_frame_equal(result, pd.read_csv(path))
    if with_events:
        assert result.loc[0, "future_backend_field"] == "preserve-me"
        assert result.loc[0, "orig_track_a"] == 12


def test_missing_declared_uvh_csv_raises(tmp_path, monkeypatch):
    def backend(**kwargs):
        return {"pet_csv": kwargs["output_csv_path"], "pet_events": []}

    kwargs = prepare_dispatch(tmp_path, monkeypatch, backend)
    with pytest.raises(FileNotFoundError):
        cli.run_video_to_pet(**kwargs)


def test_unexpected_uvh_csv_path_raises(tmp_path, monkeypatch):
    def backend(**kwargs):
        return {"pet_csv": str(tmp_path / "other.csv"), "pet_events": []}

    kwargs = prepare_dispatch(tmp_path, monkeypatch, backend)
    with pytest.raises(ValueError, match="unexpected PET CSV path"):
        cli.run_video_to_pet(**kwargs)


def test_uvh_event_only_result_keeps_legacy_fallback(tmp_path, monkeypatch):
    def backend(**kwargs):
        return {"pet_events": [{"pet": 0.5, "frame": 10, "track_a": 1, "track_b": 2}]}

    kwargs = prepare_dispatch(tmp_path, monkeypatch, backend)
    result = cli.run_video_to_pet(**kwargs)
    assert len(result) == 1
    assert result.loc[0, "pet"] == 0.5
    assert (tmp_path / "pet.csv").is_file()
