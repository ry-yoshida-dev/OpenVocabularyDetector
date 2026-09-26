from collections.abc import Sequence

import torch


class ClassEmbeddingAverager:
    """
    Merges query embeddings taken from several visual reference boxes into one embedding per class.
    """

    @staticmethod
    def average(class_ids: Sequence[int], embeddings: torch.Tensor) -> dict[int, torch.Tensor]:
        """
        L2-normalize the embedding of every box and average them per class.

        Parameters
        ----------
        class_ids : Sequence[int]
            Class id of each box.
        embeddings : torch.Tensor
            Embedding of each box, shape (N, D).

        Raises
        ------
        ValueError
            If the number of class ids and embeddings differ.

        Returns
        -------
        dict[int, torch.Tensor]
            Averaged float32 embedding of shape (D,) per class id.
        """
        if embeddings.ndim != 2 or embeddings.shape[0] != len(class_ids):
            raise ValueError(f"embeddings must have shape ({len(class_ids)}, D). got {tuple(embeddings.shape)}")
        normalized_embeddings: torch.Tensor = torch.nn.functional.normalize(embeddings.float(), dim=-1)
        return {
            class_id: normalized_embeddings[[box_class_id == class_id for box_class_id in class_ids]].mean(dim=0)
            for class_id in dict.fromkeys(class_ids)
        }
