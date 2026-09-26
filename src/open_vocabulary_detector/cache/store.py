from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from ..prompt import Prompt, TextPrompt, TextVisualPrompt, VisualReference
from .caches import ClassNameEmbeddingCache, VisualReferenceEmbeddingCache
from .class_embedding_averager import ClassEmbeddingAverager
from .reference_embeddings import ReferenceEmbeddings


@dataclass
class QueryEmbeddingStore:
    """
    Builds the per-class query embeddings of a prompt from embeddings cached per class name and per reference.

    Prompts sharing class names or references reuse the cached embeddings; only the cheap assembly
    runs per prompt.

    Attributes
    ----------
    embed_class_names : Callable[[Sequence[str]], Sequence[torch.Tensor]]
        Computes the text embedding of each class name, shape (D,) each.
    embed_references : Callable[[Sequence[VisualReference]], Sequence[ReferenceEmbeddings]]
        Computes the box embeddings of each visual reference.
    """

    embed_class_names: Callable[[Sequence[str]], Sequence[torch.Tensor]]
    embed_references: Callable[[Sequence[VisualReference]], Sequence[ReferenceEmbeddings]]
    _class_names: ClassNameEmbeddingCache = field(init=False, repr=False)
    _references: VisualReferenceEmbeddingCache = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._class_names = ClassNameEmbeddingCache(self.embed_class_names)
        self._references = VisualReferenceEmbeddingCache(self.embed_references)

    def embed(self, prompt: Prompt) -> torch.Tensor:
        """
        Query embedding of every class of a prompt.

        Parameters
        ----------
        prompt : Prompt
            Text or text-visual prompt.

        Raises
        ------
        TypeError
            If the prompt is of another kind.

        Returns
        -------
        torch.Tensor
            Float32 embeddings in class-id order, shape (C, D).
        """
        match prompt:
            case TextPrompt():
                return torch.stack(self._class_names.get(prompt.class_names)).float()
            case TextVisualPrompt():
                class_embeddings: dict[int, torch.Tensor] = self._average_references(prompt.visual_references)
                text_class_ids: tuple[int, ...] = prompt.text_class_ids
                text_embeddings: list[torch.Tensor] = self._class_names.get(
                    [prompt.class_names[class_id] for class_id in text_class_ids]
                )
                class_embeddings.update(zip(text_class_ids, text_embeddings, strict=True))
                return torch.stack([class_embeddings[class_id].float() for class_id in range(len(prompt.class_names))])
            case _:
                raise TypeError(f"cannot embed {prompt.kind} prompts.")

    def _average_references(self, references: Sequence[VisualReference]) -> dict[int, torch.Tensor]:
        reference_embeddings: list[ReferenceEmbeddings] = self._references.get(references)
        return ClassEmbeddingAverager.average(
            [class_id for embeddings in reference_embeddings for class_id in embeddings.class_ids],
            torch.cat([embeddings.embeddings for embeddings in reference_embeddings], dim=0),
        )
