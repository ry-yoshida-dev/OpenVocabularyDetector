from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from weakref import WeakKeyDictionary

import torch

from ..prompt import Prompt, PromptQuery, TextQuery, VisualQuery, VisualReference
from .core import EmbeddingCache


@dataclass
class QueryEmbeddingStore:
    """
    Builds the query embeddings of a prompt from embeddings cached per text query and per reference.

    Prompts sharing text queries or references reuse the cached embeddings; only the cheap assembly
    runs per prompt. Reference embeddings are held weakly, so they are dropped together with their reference.

    Attributes
    ----------
    embed_texts : Callable[[Sequence[str]], Sequence[torch.Tensor]]
        Computes the text embedding of each text query, shape (D,) each.
    embed_references : Callable[[Sequence[VisualReference]], Sequence[torch.Tensor]]
        Computes the box embeddings of each visual reference, one row per box, shape (N, D) each.
    """

    embed_texts: Callable[[Sequence[str]], Sequence[torch.Tensor]]
    embed_references: Callable[[Sequence[VisualReference]], Sequence[torch.Tensor]]
    _texts: EmbeddingCache[str, torch.Tensor] = field(init=False, repr=False)
    _references: EmbeddingCache[VisualReference, torch.Tensor] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._texts = EmbeddingCache(compute=self.embed_texts, entries=dict[str, torch.Tensor]())
        self._references = EmbeddingCache(
            compute=self.embed_references, entries=WeakKeyDictionary[VisualReference, torch.Tensor]()
        )

    def embed(self, prompt: Prompt) -> torch.Tensor:
        """
        Embedding of every query of a prompt.

        A text query yields the embedding of its phrase; a visual query the L2-normalized box embeddings of all
        its references, averaged.

        Parameters
        ----------
        prompt : Prompt
            Prompt with text queries, visual queries or both.

        Returns
        -------
        torch.Tensor
            Float32 embeddings in query-id order, shape (Q, D).
        """
        texts: list[str] = [query.text for query in prompt.queries if isinstance(query, TextQuery)]
        text_embeddings: dict[str, torch.Tensor] = dict(zip(texts, self._texts.get(texts), strict=True))
        return torch.stack([self._embed_query(query, text_embeddings).float() for query in prompt.queries])

    def _embed_query(self, query: PromptQuery, text_embeddings: Mapping[str, torch.Tensor]) -> torch.Tensor:
        match query:
            case TextQuery(text=text):
                return text_embeddings[text]
            case VisualQuery(references=references):
                box_embeddings: torch.Tensor = torch.cat(self._references.get(references), dim=0).float()
                return torch.nn.functional.normalize(box_embeddings, dim=-1).mean(dim=0)
