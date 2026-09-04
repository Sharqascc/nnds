from unittest.mock import MagicMock, patch

from src.vlm.analyzer import VLLMAnalyzer


def test_init_small_model_mocks():
    with (
        patch("src.vlm.analyzer.torch.cuda.is_available", return_value=False),
        patch("src.vlm.analyzer.BlipProcessor.from_pretrained") as mock_proc_from,
        patch("src.vlm.analyzer.BlipForQuestionAnswering.from_pretrained") as mock_model_from,
    ):
        mock_proc = MagicMock()
        mock_model = MagicMock()
        mock_proc_from.return_value = mock_proc
        mock_model_from.return_value = mock_model
        analyzer = VLLMAnalyzer(model_name="Salesforce/blip-vqa-base", device="cpu")
    assert analyzer.processor is mock_proc
    assert analyzer.model is mock_model
    mock_model.to.assert_called_with("cpu")
    mock_model.eval.assert_called_once()


def test_init_blip2_model_mocks():
    with (
        patch("src.vlm.analyzer.torch.cuda.is_available", return_value=True),
        patch("src.vlm.analyzer.Blip2Processor.from_pretrained") as mock_proc2_from,
        patch("src.vlm.analyzer.Blip2ForConditionalGeneration.from_pretrained") as mock_model2_from,
    ):
        mock_proc2 = MagicMock()
        mock_model2 = MagicMock()
        mock_proc2_from.return_value = mock_proc2
        mock_model2_from.return_value = mock_model2
        analyzer = VLLMAnalyzer(model_name="Salesforce/blip2-opt-2.7b", device="cpu")
    assert analyzer.processor is mock_proc2
    assert analyzer.model is mock_model2
    # .to should not be called for blip2
    mock_model2.to.assert_not_called()


def test_analyze_image_mocks():
    with (
        patch("src.vlm.analyzer.torch.cuda.is_available", return_value=False),
        patch("src.vlm.analyzer.BlipProcessor.from_pretrained") as mock_proc_from,
        patch("src.vlm.analyzer.BlipForQuestionAnswering.from_pretrained") as mock_model_from,
        patch("PIL.Image.open") as mock_image_open,
        patch("torch.no_grad"),
    ):
        mock_proc = MagicMock()
        mock_model = MagicMock()
        mock_proc_from.return_value = mock_proc
        mock_model_from.return_value = mock_model
        mock_img = MagicMock()
        mock_img.convert.return_value = "image"
        mock_image_open.return_value = mock_img
        # Configure processor
        mock_proc.return_value = {
            "pixel_values": MagicMock(),
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }
        mock_proc.decode.return_value = "mocked answer"
        mock_model.generate.return_value = MagicMock()
        mock_model.generate.return_value[0] = "ids"
        analyzer = VLLMAnalyzer(model_name="Salesforce/blip-vqa-base", device="cpu")
        result = analyzer.analyze_image("dummy.jpg", prompt="question")
    assert result == "mocked answer"
