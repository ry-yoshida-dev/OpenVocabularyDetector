import gc
from collections.abc import Sequence

import numpy as np
import torch
from geometry import Box2DFormat, Boxes2D
from PIL import Image

from open_vocabulary_detector import Prompt, TextQuery, VisualQuery, VisualReference
from open_vocabulary_detector.cache import QueryEmbeddingStore

TEXT_EMBEDDINGS: dict[str, list[float]] = {"cat": [1.0, 0.0], "dog": [0.0, 1.0], "bird": [1.0, 1.0]}


class CountingEmbedder:
    def __init__(self) -> None:
        self.embedded_texts: list[list[str]] = []
        self.embedded_references: list[VisualReference] = []

    def embed_texts(self, text_queries: Sequence[str]) -> list[torch.Tensor]:
        self.embedded_texts.append(list(text_queries))
        return [torch.tensor(TEXT_EMBEDDINGS[text_query]) for text_query in text_queries]

    def embed_references(self, references: Sequence[VisualReference]) -> list[torch.Tensor]:
        self.embedded_references.extend(references)
        return [torch.from_numpy(reference.xyxy[:, 2:]) for reference in references]


def build_store() -> tuple[QueryEmbeddingStore, CountingEmbedder]:
    embedder: CountingEmbedder = CountingEmbedder()
    return QueryEmbeddingStore(embedder.embed_texts, embedder.embed_references), embedder


def build_reference(xyxy: list[list[float]]) -> VisualReference:
    return VisualReference(
        image=Image.new("RGB", (10, 10)),
        boxes=Boxes2D.register(value=np.array(xyxy, dtype=np.float64), box2d_format=Box2DFormat.XYXY),
    )


def test_text_queries_are_embedded_once_across_prompts() -> None:
    store, embedder = build_store()
    torch.testing.assert_close(
        store.embed(Prompt.from_class_names(("cat", "dog"))), torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    )
    torch.testing.assert_close(
        store.embed(Prompt.from_class_names(("dog", "bird"))), torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    )
    store.embed(Prompt.from_class_names(("cat", "dog")))
    assert embedder.embedded_texts == [["cat", "dog"], ["bird"]]


def test_every_text_query_of_a_class_gets_its_own_embedding() -> None:
    store, _ = build_store()
    prompt: Prompt = Prompt.from_texts({"pet": ("cat", "dog"), "bird": ("bird",)})
    torch.testing.assert_close(store.embed(prompt), torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))


def test_visual_query_averages_normalized_boxes_of_its_references() -> None:
    store, embedder = build_store()
    mug_references: tuple[VisualReference, ...] = (
        build_reference([[0.0, 0.0, 3.0, 4.0]]),
        build_reference([[0.0, 0.0, 4.0, 3.0], [1.0, 1.0, 3.0, 4.0]]),
    )
    cup_reference: VisualReference = build_reference([[0.0, 0.0, 8.0, 6.0]])
    prompt: Prompt = Prompt({"my mug": (VisualQuery(mug_references),), "my cup": (VisualQuery((cup_reference,)),)})
    torch.testing.assert_close(
        store.embed(prompt), torch.tensor([[2.0 / 3.0, 22.0 / 30.0], [0.8, 0.6]]), atol=1e-6, rtol=0.0
    )
    store.embed(Prompt({"my cup": (VisualQuery((cup_reference,)),)}))
    assert embedder.embedded_references == [*mug_references, cup_reference]


def test_unused_references_are_released() -> None:
    store, embedder = build_store()
    store.embed(Prompt({"my mug": (VisualQuery((build_reference([[0.0, 0.0, 5.0, 5.0]]),)),)}))
    embedder.embedded_references.clear()
    gc.collect()
    assert len(store._references.entries) == 0


def test_text_and_visual_queries_keep_query_order() -> None:
    store, embedder = build_store()
    reference: VisualReference = build_reference([[0.0, 0.0, 8.0, 6.0]])
    prompt: Prompt = Prompt(
        {"pet": (TextQuery("cat"), VisualQuery((reference,)), TextQuery("dog")), "bird": (TextQuery("bird"),)}
    )
    torch.testing.assert_close(
        store.embed(prompt), torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [1.0, 1.0]]), atol=1e-6, rtol=0.0
    )
    assert embedder.embedded_texts == [["cat", "dog", "bird"]]
