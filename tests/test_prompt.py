from collections.abc import Mapping, Sequence

import numpy as np
import pytest
from PIL import Image

from geometry import Box2DFormat, Boxes2D
from open_vocabulary_detector import Prompt, PromptKind, PromptQuery, TextQuery, VisualQuery, VisualReference


def build_reference(xyxy: list[list[float]]) -> VisualReference:
    return VisualReference(
        image=Image.new("RGB", (100, 50)),
        boxes=Boxes2D.register(value=np.array(xyxy, dtype=np.float64), box2d_format=Box2DFormat.XYXY),
    )


def test_text_prompt_classes_follow_the_keys() -> None:
    prompt: Prompt = Prompt.from_texts({" car ": ("car", " suv", "taxi "), "person": ("person",)})
    assert prompt.kinds == frozenset({PromptKind.TEXT})
    assert prompt.class_names == ("car", "person")
    assert prompt.queries == (TextQuery("car"), TextQuery("suv"), TextQuery("taxi"), TextQuery("person"))
    assert prompt.query_class_ids == (0, 0, 0, 1)
    assert prompt.query_texts == ("car", "suv", "taxi", "person")
    assert prompt.class_ids_of(np.array([3, 1, 2], dtype=np.int64)).tolist() == [1, 0, 0]


def test_class_names_query_themselves() -> None:
    prompt: Prompt = Prompt.from_class_names((" person", "traffic light"))
    assert prompt == Prompt.from_texts({"person": ("person",), "traffic light": ("traffic light",)})
    assert prompt.query_class_ids == (0, 1)


def test_one_class_mixes_text_and_visual_queries() -> None:
    suv_reference: VisualReference = build_reference([[10, 10, 40, 40]])
    van_references: tuple[VisualReference, ...] = (build_reference([[0, 0, 5, 5]]), build_reference([[1, 1, 9, 9]]))
    prompt: Prompt = Prompt(
        {
            "car": (TextQuery("car"), VisualQuery((suv_reference,)), VisualQuery(van_references)),
            "person": (TextQuery("person"),),
        }
    )
    assert prompt.kinds == frozenset({PromptKind.TEXT, PromptKind.VISUAL})
    assert prompt.query_class_ids == (0, 0, 0, 1)
    assert prompt.query_names == ("car", "car", "car", "person")
    assert prompt.queries[2] == VisualQuery(van_references)
    with pytest.raises(ValueError, match="visual queries"):
        _ = prompt.query_texts


def test_visual_only_prompt() -> None:
    prompt: Prompt = Prompt({"my mug": (VisualQuery((build_reference([[10, 10, 40, 40]]),)),)})
    assert prompt.kinds == frozenset({PromptKind.VISUAL})
    assert prompt.query_names == ("my mug",)


def test_prompt_copies_its_input_and_compares_by_value() -> None:
    class_texts: dict[str, tuple[str, ...]] = {"car": ("suv",)}
    prompt: Prompt = Prompt.from_texts(class_texts)
    class_texts["car"] = ("van",)
    assert prompt.query_texts == ("suv",)
    assert prompt == Prompt.from_texts({"car": ("suv",)})
    assert hash(prompt) == hash(Prompt.from_texts({"car": ("suv",)}))
    assert prompt != Prompt.from_texts({"vehicle": ("suv",)})


def test_visual_queries_compare_by_reference_identity() -> None:
    reference: VisualReference = build_reference([[0, 0, 10, 10]])
    same_pixels: VisualReference = build_reference([[0, 0, 10, 10]])
    assert Prompt({"mug": (VisualQuery((reference,)),)}) == Prompt({"mug": (VisualQuery((reference,)),)})
    assert Prompt({"mug": (VisualQuery((reference,)),)}) != Prompt({"mug": (VisualQuery((same_pixels,)),)})


REFERENCE: VisualReference = build_reference([[0, 0, 10, 10]])
OTHER_REFERENCE: VisualReference = build_reference([[0, 0, 20, 20]])


@pytest.mark.parametrize(
    ("class_queries", "message"),
    [
        ({}, "at least one class"),
        ({" ": (TextQuery("x"),)}, "must not be blank"),
        ({"car": (TextQuery("car"),), "car ": (TextQuery("suv"),)}, "class names must be unique"),
        ({"car": ()}, "needs at least one query"),
        ({"car": (TextQuery("suv"), TextQuery("SUV"))}, "more than once"),
        ({"car": (TextQuery("van"),), "truck": (TextQuery("van"),)}, "more than once"),
        ({"mug": (VisualQuery((REFERENCE,)), VisualQuery((REFERENCE,)))}, "more than once"),
        ({"mug": (VisualQuery((REFERENCE,)),), "cup": (VisualQuery((REFERENCE, OTHER_REFERENCE)),)}, "only one class"),
    ],
)
def test_invalid_prompts_raise(class_queries: Mapping[str, Sequence[PromptQuery]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Prompt(class_queries)


def test_query_validation() -> None:
    with pytest.raises(ValueError, match="blank"):
        TextQuery(" ")
    assert TextQuery(" suv ").text == "suv"
    with pytest.raises(ValueError, match="at least one reference"):
        VisualQuery(())
    with pytest.raises(ValueError, match="must not repeat"):
        VisualQuery((REFERENCE, REFERENCE))


def test_reference_validation() -> None:
    np.testing.assert_allclose(build_reference([[10, 10, 40, 40]]).normalized_xyxy, [[0.1, 0.2, 0.4, 0.8]])
    with pytest.raises(ValueError, match="inside the 100x50 image"):
        build_reference([[0, 0, 10, 60]])
