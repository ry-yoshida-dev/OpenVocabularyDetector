import pytest
import torch
from transformers import BatchEncoding

from open_vocabulary_detector.backends.grounding_dino import GroundingCaption, TokenizedCaption
from open_vocabulary_detector.runtime import ClassPrediction


def build_tokenized_caption() -> TokenizedCaption:
    caption: GroundingCaption = GroundingCaption.from_class_names(("cat", "traffic cone"))
    offsets: list[tuple[int, int]] = [(0, 0), (0, 3), (4, 5), (6, 13), (14, 18), (19, 20), (0, 0)]
    token_count: int = len(offsets)
    encoding: BatchEncoding = BatchEncoding(
        {
            "input_ids": torch.arange(token_count).unsqueeze(0),
            "attention_mask": torch.ones(1, token_count, dtype=torch.long),
            "token_type_ids": torch.zeros(1, token_count, dtype=torch.long),
            "offset_mapping": torch.tensor([offsets]),
        }
    )
    return TokenizedCaption.from_encoding(caption, encoding)


def test_from_encoding_maps_class_spans_to_tokens() -> None:
    tokenized_caption: TokenizedCaption = build_tokenized_caption()
    assert tokenized_caption.token_count == 7
    assert tokenized_caption.class_token_mask.tolist() == [
        [False, True, False, False, False, False, False],
        [False, False, False, True, True, False, False],
    ]


def test_class_probabilities_take_the_best_token_of_each_class() -> None:
    tokenized_caption: TokenizedCaption = build_tokenized_caption()
    token_probabilities: torch.Tensor = torch.tensor([[[0.9, 0.2, 0.9, 0.1, 0.6, 0.9, 0.9, 0.9, 0.9]]])
    logits: torch.Tensor = torch.logit(token_probabilities)
    class_probabilities: torch.Tensor = tokenized_caption.class_probabilities(logits)
    assert class_probabilities.shape == (1, 1, 2)
    torch.testing.assert_close(class_probabilities, torch.tensor([[[0.2, 0.6]]]))
    prediction: ClassPrediction = ClassPrediction.from_probabilities(class_probabilities)
    assert prediction.class_ids.tolist() == [[1]]


def test_tokenized_caption_rejects_class_without_tokens() -> None:
    with pytest.raises(ValueError, match="at least one token"):
        TokenizedCaption(
            input_ids=torch.zeros(1, 3, dtype=torch.long),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            token_type_ids=torch.zeros(1, 3, dtype=torch.long),
            class_token_mask=torch.tensor([[True, False, False], [False, False, False]]),
        )
