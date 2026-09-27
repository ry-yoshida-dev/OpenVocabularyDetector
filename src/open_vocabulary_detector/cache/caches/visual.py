from collections.abc import MutableMapping
from dataclasses import dataclass, field
from weakref import WeakKeyDictionary

import torch

from ...prompt import VisualReference
from ..core import EmbeddingCache


@dataclass
class VisualEmbeddingCache(EmbeddingCache[VisualReference, torch.Tensor]):
    """
    Box embeddings per visual reference, shape (N, D) each, held weakly so they are dropped with the reference.
    """

    _embeddings: WeakKeyDictionary[VisualReference, torch.Tensor] = field(
        default_factory=WeakKeyDictionary[VisualReference, torch.Tensor], init=False, repr=False
    )

    @property
    def _entries(self) -> MutableMapping[VisualReference, torch.Tensor]:
        return self._embeddings
