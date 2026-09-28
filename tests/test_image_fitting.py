import pytest
import torch

from open_vocabulary_detector.backends.owl_vit import ImageFitting
from open_vocabulary_detector.result import ImageSize

WIDE_IMAGE: ImageSize = ImageSize(width=200, height=100)
TALL_IMAGE: ImageSize = ImageSize(width=100, height=400)


def test_stretch_keeps_boxes() -> None:
    input_cxcywh: torch.Tensor = torch.tensor([[[0.5, 0.25, 0.2, 0.1]]])
    assert torch.equal(ImageFitting.STRETCH.boxes_to_image(input_cxcywh, [WIDE_IMAGE]), input_cxcywh)
    normalized_xyxy: torch.Tensor = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64)
    assert torch.equal(ImageFitting.STRETCH.boxes_to_input(normalized_xyxy, WIDE_IMAGE), normalized_xyxy)


def test_pad_to_square_coverage_follows_the_longer_side() -> None:
    assert ImageFitting.PAD_TO_SQUARE.coverage(WIDE_IMAGE).tolist() == [1.0, 0.5, 1.0, 0.5]
    assert ImageFitting.PAD_TO_SQUARE.coverage(TALL_IMAGE).tolist() == [0.25, 1.0, 0.25, 1.0]


def test_pad_to_square_rescales_each_image_of_a_batch() -> None:
    input_cxcywh: torch.Tensor = torch.tensor([[[0.5, 0.25, 0.2, 0.1]], [[0.125, 0.5, 0.25, 1.0]]])
    image_cxcywh: torch.Tensor = ImageFitting.PAD_TO_SQUARE.boxes_to_image(input_cxcywh, [WIDE_IMAGE, TALL_IMAGE])
    assert image_cxcywh.dtype == torch.float32
    assert image_cxcywh[0, 0].tolist() == pytest.approx([0.5, 0.5, 0.2, 0.2])
    assert image_cxcywh[1, 0].tolist() == pytest.approx([0.5, 0.5, 1.0, 1.0])


def test_pad_to_square_maps_image_boxes_into_the_input() -> None:
    normalized_xyxy: torch.Tensor = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.5, 0.5, 1.0, 1.0]], dtype=torch.float64)
    input_xyxy: torch.Tensor = ImageFitting.PAD_TO_SQUARE.boxes_to_input(normalized_xyxy, WIDE_IMAGE)
    assert input_xyxy.dtype == torch.float64
    assert input_xyxy[0].tolist() == pytest.approx([0.0, 0.0, 1.0, 0.5])
    assert input_xyxy[1].tolist() == pytest.approx([0.5, 0.25, 1.0, 0.5])


def test_boxes_to_image_rejects_mismatched_image_sizes() -> None:
    with pytest.raises(ValueError, match="image sizes"):
        ImageFitting.PAD_TO_SQUARE.boxes_to_image(torch.zeros(2, 1, 4), [WIDE_IMAGE])
