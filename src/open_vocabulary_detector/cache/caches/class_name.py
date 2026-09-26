from collections.abc import MutableMapping
from dataclasses import dataclass, field

import torch

from ..core import EmbeddingCache


@dataclass
class ClassNameEmbeddingCache(EmbeddingCache[str, torch.Tensor]):
    """
    Text embedding per class name; a name's embedding does not depend on the other classes of a prompt.
    """

    _embeddings: dict[str, torch.Tensor] = field(default_factory=dict[str, torch.Tensor], init=False, repr=False)

    @property
    def _entries(self) -> MutableMapping[str, torch.Tensor]:
        return self._embeddings
