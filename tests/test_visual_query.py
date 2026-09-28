import torch

from open_vocabulary_detector.backends.owl_vit.box_query_selector import OwlBoxQuerySelector


def test_box_query_selector_picks_overlapping_and_least_background_like_patch() -> None:
    predicted_cxcywh: torch.Tensor = torch.tensor(
        [[0.25, 0.25, 0.5, 0.5], [0.26, 0.26, 0.5, 0.5], [0.75, 0.75, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]]
    )
    class_embeddings: torch.Tensor = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    selected: torch.Tensor = OwlBoxQuerySelector.select(
        target_xyxy=torch.tensor([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]]),
        predicted_cxcywh=predicted_cxcywh,
        class_embeddings=class_embeddings,
    )
    torch.testing.assert_close(selected, torch.tensor([[0.0, 1.0], [1.0, 1.0]]))


def test_box_query_selector_falls_back_to_generalized_iou() -> None:
    selected: torch.Tensor = OwlBoxQuerySelector.select(
        target_xyxy=torch.tensor([[0.0, 0.0, 0.1, 0.1]]),
        predicted_cxcywh=torch.tensor([[0.3, 0.3, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1]]),
        class_embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    torch.testing.assert_close(selected, torch.tensor([[1.0, 0.0]]))
