from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, MutableMapping, Sequence
from dataclasses import dataclass


@dataclass
class EmbeddingCache[KeyT: Hashable, ValueT](ABC):
    """
    Base of caches that compute embeddings per key and keep them for later prompts.

    Missing keys are computed together in one call; subclasses decide how entries are stored.

    Attributes
    ----------
    compute : Callable[[Sequence[KeyT]], Sequence[ValueT]]
        Computes the values of keys, one per key in order.
    """

    compute: Callable[[Sequence[KeyT]], Sequence[ValueT]]

    @property
    @abstractmethod
    def _entries(self) -> MutableMapping[KeyT, ValueT]:
        """
        Storage of the cached values.

        Returns
        -------
        MutableMapping[KeyT, ValueT]
            Cached value per key.
        """

    def get(self, keys: Sequence[KeyT]) -> list[ValueT]:
        """
        Return the value of every key, computing only the keys not cached yet.

        Parameters
        ----------
        keys : Sequence[KeyT]
            Keys to look up.

        Raises
        ------
        ValueError
            If ``compute`` does not return one value per missing key.

        Returns
        -------
        list[ValueT]
            Value of each key, in order.
        """
        missing_keys: list[KeyT] = [key for key in dict.fromkeys(keys) if key not in self._entries]
        if missing_keys:
            values: Sequence[ValueT] = self.compute(missing_keys)
            if len(values) != len(missing_keys):
                raise ValueError(f"expected {len(missing_keys)} computed values. got {len(values)}")
            self._entries.update(zip(missing_keys, values, strict=True))
        return [self._entries[key] for key in keys]
