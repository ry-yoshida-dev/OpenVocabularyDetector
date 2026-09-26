import gc
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image

from geometry import Box2DFormat, Boxes2D
from open_vocabulary_detector import TextPrompt, TextVisualPrompt, VisualReference
from open_vocabulary_detector.cache import QueryEmbeddingStore, ReferenceEmbeddings

NAME_EMBEDDINGS: dict[str, list[float]] = {"cat": [1.0, 0.0], "dog": [0.0, 1.0], "bird": [1.0, 1.0]}


class CountingEmbedder:
    def __init__(self) -> None:
        self.embedded_names: list[list[str]] = []
        self.embedded_references: list[VisualReference] = []

    def embed_class_names(self, class_names: Sequence[str]) -> list[torch.Tensor]:
        self.embedded_names.append(list(class_names))
        return [torch.tensor(NAME_EMBEDDINGS[name]) for name in class_names]

    def embed_references(self, references: Sequence[VisualReference]) -> list[ReferenceEmbeddings]:
        self.embedded_references.extend(references)
        return [
            ReferenceEmbeddings(
                class_ids=tuple(int(class_id) for class_id in reference.class_ids.tolist()),
                embeddings=torch.tensor([[3.0, 4.0]] * len(reference.class_ids)),
            )
            for reference in references
        ]


def build_store() -> tuple[QueryEmbeddingStore, CountingEmbedder]:
    embedder: CountingEmbedder = CountingEmbedder()
    return QueryEmbeddingStore(embedder.embed_class_names, embedder.embed_references), embedder


def build_reference(class_id: int) -> VisualReference:
    return VisualReference(
        image=Image.new("RGB", (10, 10)),
        boxes=Boxes2D.register(value=np.array([[0.0, 0.0, 5.0, 5.0]]), box2d_format=Box2DFormat.XYXY),
        class_ids=np.array([class_id], dtype=np.int64),
    )


def test_class_names_are_embedded_once_across_prompts() -> None:
    store, embedder = build_store()
    torch.testing.assert_close(
        store.embed(TextPrompt(class_names=("cat", "dog"))), torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    )
    torch.testing.assert_close(
        store.embed(TextPrompt(class_names=("dog", "bird"))), torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    )
    store.embed(TextPrompt(class_names=("cat", "dog")))
    assert embedder.embedded_names == [["cat", "dog"], ["bird"]]


def test_text_visual_prompt_mixes_cached_reference_and_name_embeddings() -> None:
    store, embedder = build_store()
    reference: VisualReference = build_reference(class_id=1)
    prompt: TextVisualPrompt = TextVisualPrompt(class_names=("cat", "my mug"), visual_references=(reference,))
    torch.testing.assert_close(store.embed(prompt), torch.tensor([[1.0, 0.0], [0.6, 0.8]]))
    store.embed(TextVisualPrompt(class_names=("dog", "my mug"), visual_references=(reference,)))
    assert embedder.embedded_references == [reference]


def test_unused_references_are_released() -> None:
    store, embedder = build_store()
    store.embed(TextVisualPrompt(class_names=("my mug",), visual_references=(build_reference(class_id=0),)))
    embedder.embedded_references.clear()
    gc.collect()
    assert len(store._references._embeddings) == 0
