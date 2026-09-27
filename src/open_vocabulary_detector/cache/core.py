from collections.abc import Callable, Hashable, MutableMapping, Sequence
from dataclasses import dataclass, field


@dataclass
class EmbeddingCache[KeyT: Hashable, ValueT]:
    """
    Cache that computes embeddings per key and keeps them for later prompts.

    Missing keys are computed together in one call. The storage decides how long entries live, e.g. a ``dict`` keeps
    them for the lifetime of the cache and a ``WeakKeyDictionary`` drops them together with their key.

    Attributes
    ----------
    compute : Callable[[Sequence[KeyT]], Sequence[ValueT]]
        Computes the values of keys, one per key in order.
    entries : MutableMapping[KeyT, ValueT]
        Storage of the cached value per key.
    """

    compute: Callable[[Sequence[KeyT]], Sequence[ValueT]]
    entries: MutableMapping[KeyT, ValueT] = field(repr=False)

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
        missing_keys: list[KeyT] = [key for key in dict.fromkeys(keys) if key not in self.entries]
        if missing_keys:
            values: Sequence[ValueT] = self.compute(missing_keys)
            if len(values) != len(missing_keys):
                raise ValueError(f"expected {len(missing_keys)} computed values. got {len(values)}")
            self.entries.update(zip(missing_keys, values, strict=True))
        return [self.entries[key] for key in keys]
