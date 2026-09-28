import pytest
import torch

from open_vocabulary_detector import Prompt
from open_vocabulary_detector.backends.omdet_turbo import PromptEncoding


def build_encoding(prompt: Prompt, class_rows: int, task_rows: int = 1) -> PromptEncoding:
    return PromptEncoding(
        prompt=prompt,
        class_input_ids=torch.zeros((class_rows, 77), dtype=torch.long),
        class_attention_mask=torch.ones((class_rows, 77), dtype=torch.long),
        task_input_ids=torch.zeros((task_rows, 77), dtype=torch.long),
        task_attention_mask=torch.ones((task_rows, 77), dtype=torch.long),
    )


def test_encoding_matches_its_prompt() -> None:
    prompt: Prompt = Prompt.from_texts({"car": ("car", "suv"), "person": ("person",)})
    encoding: PromptEncoding = build_encoding(prompt, class_rows=3)
    assert encoding.query_count == 3
    assert encoding.is_for(Prompt.from_texts({"car": ("car", "suv"), "person": ("person",)}))
    assert not encoding.is_for(Prompt.from_class_names(("car", "person")))


def test_encoding_rejects_mismatched_rows() -> None:
    prompt: Prompt = Prompt.from_class_names(("car", "person"))
    with pytest.raises(ValueError, match="class tokens"):
        build_encoding(prompt, class_rows=3)
    with pytest.raises(ValueError, match="task tokens"):
        build_encoding(prompt, class_rows=2, task_rows=2)
