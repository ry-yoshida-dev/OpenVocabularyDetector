from collections.abc import MutableMapping
from dataclasses import dataclass, field
from weakref import WeakKeyDictionary

from ...prompt import VisualReference
from ..core import EmbeddingCache
from ..reference_embeddings import ReferenceEmbeddings


@dataclass
class VisualReferenceEmbeddingCache(EmbeddingCache[VisualReference, ReferenceEmbeddings]):
    """
    Box embeddings per visual reference, held weakly so they are dropped with the reference.
    """

    _embeddings: WeakKeyDictionary[VisualReference, ReferenceEmbeddings] = field(
        default_factory=WeakKeyDictionary[VisualReference, ReferenceEmbeddings], init=False, repr=False
    )

    @property
    def _entries(self) -> MutableMapping[VisualReference, ReferenceEmbeddings]:
        return self._embeddings
