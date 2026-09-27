from collections.abc import MutableMapping
from dataclasses import dataclass, field

import torch

from ..core import EmbeddingCache


@dataclass
class TextEmbeddingCache(EmbeddingCache[str, torch.Tensor]):
    """
    Text embedding per text query; a query's embedding does not depend on the other queries of a prompt.
    """

    _embeddings: dict[str, torch.Tensor] = field(default_factory=dict[str, torch.Tensor], init=False, repr=False)

    @property
    def _entries(self) -> MutableMapping[str, torch.Tensor]:
        return self._embeddings
