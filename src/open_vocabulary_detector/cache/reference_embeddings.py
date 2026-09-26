from dataclasses import dataclass

import torch


@dataclass(frozen=True, eq=False)
class ReferenceEmbeddings:
    """
    Query embeddings extracted from one visual reference.

    Attributes
    ----------
    class_ids : tuple[int, ...]
        Prompt class id of each embedding.
    embeddings : torch.Tensor
        One embedding per entry of ``class_ids``, shape (N, D).

    Raises
    ------
    ValueError
        If the number of class ids and embeddings differ.
    """

    class_ids: tuple[int, ...]
    embeddings: torch.Tensor

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] != len(self.class_ids):
            raise ValueError(
                f"embeddings must have shape ({len(self.class_ids)}, D). got {tuple(self.embeddings.shape)}"
            )
