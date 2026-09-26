import numpy as np
import pytest
from PIL import Image

from geometry import Box2DFormat, Boxes2D
from open_vocabulary_detector import PromptKind, TextPrompt, TextVisualPrompt, VisualReference


def build_reference(image: Image.Image, xyxy: list[list[float]], class_ids: list[int]) -> VisualReference:
    return VisualReference(
        image=image,
        boxes=Boxes2D.register(value=np.array(xyxy, dtype=np.float64), box2d_format=Box2DFormat.XYXY),
        class_ids=np.array(class_ids, dtype=np.int64),
    )


def test_text_prompt_strips_names() -> None:
    prompt: TextPrompt = TextPrompt(class_names=(" cat ", "dog"))
    assert prompt.class_names == ("cat", "dog")
    assert prompt.kind is PromptKind.TEXT


@pytest.mark.parametrize("class_names", [(), ("cat", " "), ("cat", "cat ")])
def test_invalid_class_names_raise(class_names: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        TextPrompt(class_names=class_names)


def test_text_visual_prompt_splits_text_and_visual_classes() -> None:
    reference: VisualReference = build_reference(Image.new("RGB", (100, 50)), [[10, 10, 40, 40]], [1])
    prompt: TextVisualPrompt = TextVisualPrompt(class_names=("cat", "my mug"), visual_references=(reference,))
    assert prompt.kind is PromptKind.TEXT_VISUAL
    assert prompt.visual_class_ids == frozenset({1})
    assert prompt.text_class_ids == (0,)
    np.testing.assert_allclose(reference.normalized_xyxy, [[0.1, 0.2, 0.4, 0.8]])
    assert prompt == TextVisualPrompt(class_names=("cat", "my mug"), visual_references=(reference,))
    assert hash(prompt) == hash(TextVisualPrompt(class_names=("cat", "my mug"), visual_references=(reference,)))


def test_text_visual_prompt_validation() -> None:
    image: Image.Image = Image.new("RGB", (100, 50))
    with pytest.raises(ValueError, match="at least one reference"):
        TextVisualPrompt(class_names=("mug",), visual_references=())
    with pytest.raises(ValueError, match="class_ids must be in"):
        TextVisualPrompt(class_names=("mug",), visual_references=(build_reference(image, [[0, 0, 10, 10]], [1]),))
    with pytest.raises(ValueError, match="inside the 100x50 image"):
        build_reference(image, [[0, 0, 10, 60]], [0])
    with pytest.raises(ValueError, match="class_ids must have shape"):
        build_reference(image, [[0, 0, 10, 10]], [0, 1])
