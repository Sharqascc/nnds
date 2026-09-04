from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.pipeline.reid_encoder import ReIDEncoder


class DummyModel:
    def __init__(self, output_tensor=None):
        self.classifier = None
        self.output_tensor = (
            output_tensor if output_tensor is not None else torch.tensor([1.0, 2.0, 3.0])
        )

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, *args, **kwargs):
        return self.output_tensor


@pytest.fixture
def dummy_weights():
    class DummyWeights:
        DEFAULT = "dummy"

    return DummyWeights


def test_constructor_patched(dummy_weights):
    """Cover __init__ without downloading real weights."""
    with (
        patch("torchvision.models.MobileNet_V3_Small_Weights", dummy_weights),
        patch("torchvision.models.mobilenet_v3_small", return_value=DummyModel()),
    ):
        encoder = ReIDEncoder(device="cpu")
    assert encoder.model is not None
    assert hasattr(encoder, "transform")


def test_encode_crop_valid():
    """Cover successful encoding path (lines 28-50)."""
    encoder = ReIDEncoder.__new__(ReIDEncoder)
    encoder.device = torch.device("cpu")
    encoder.model = DummyModel(torch.tensor([3.0, 4.0, 0.0]))
    import torchvision.transforms as T

    encoder.transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    result = encoder.encode_crop(frame, 10, 10, 40, 40)
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 3


def test_encode_crop_invalid_coords():
    """Cover x2i <= x1i or y2i <= y1i returns None."""
    encoder = ReIDEncoder.__new__(ReIDEncoder)
    encoder.device = torch.device("cpu")
    # Even if model and transform are not needed for early return
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    result = encoder.encode_crop(frame, 40, 40, 10, 10)  # x2i < x1i
    assert result is None


def test_encode_crop_empty_crop():
    """Cover crop.size == 0 returns None."""
    encoder = ReIDEncoder.__new__(ReIDEncoder)
    encoder.device = torch.device("cpu")
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    result = encoder.encode_crop(frame, 50, 50, 50, 50)  # zero width/height
    assert result is None


def test_encode_crop_zero_norm():
    """Cover norm < 1e-6 returns None."""
    encoder = ReIDEncoder.__new__(ReIDEncoder)
    encoder.device = torch.device("cpu")
    encoder.model = DummyModel(torch.zeros(3))
    import torchvision.transforms as T

    encoder.transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    result = encoder.encode_crop(frame, 10, 10, 40, 40)
    assert result is None


def test_encode_crop_exception():
    """Cover exception handler returns None."""
    encoder = ReIDEncoder.__new__(ReIDEncoder)
    encoder.device = torch.device("cpu")
    # Force transform to raise an exception
    encoder.transform = MagicMock(side_effect=Exception("fail"))
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    result = encoder.encode_crop(frame, 10, 10, 40, 40)
    assert result is None


def test_encode_crop_empty_crop_via_mock_frame():
    """Cover crop.size == 0 return None (line 38)."""

    class DummyFrame:
        shape = (10, 10, 3)

        def __getitem__(self, key):
            return np.array([])

    encoder = ReIDEncoder.__new__(ReIDEncoder)
    result = encoder.encode_crop(DummyFrame(), 0, 0, 10, 10)
    assert result is None
