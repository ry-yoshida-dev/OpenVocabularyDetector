from dataclasses import dataclass

import numpy as np
import torch

from ..result import RawDetections


@dataclass(frozen=True, eq=False)
class QueryPrediction:
    """
    Best prompt query of every predicted box in a mini-batch.

    Attributes
    ----------
    confidences : torch.Tensor
        Probability of the best query, shape (B, N).
    query_ids : torch.Tensor
        Index of the best query, shape (B, N).

    Raises
    ------
    ValueError
        If ``confidences`` is not two-dimensional or the shapes differ.
    """

    confidences: torch.Tensor
    query_ids: torch.Tensor

    def __post_init__(self) -> None:
        if self.confidences.ndim != 2 or self.confidences.shape != self.query_ids.shape:
            raise ValueError(
                f"confidences and query_ids must share shape (B, N). got {tuple(self.confidences.shape)} "
                + f"and {tuple(self.query_ids.shape)}"
            )

    @classmethod
    def from_probabilities(cls, query_probabilities: torch.Tensor) -> "QueryPrediction":
        """
        Pick the most probable query of every predicted box.

        Parameters
        ----------
        query_probabilities : torch.Tensor
            Per-query probabilities, shape (B, N, Q).

        Raises
        ------
        ValueError
            If ``query_probabilities`` is not three-dimensional or has no query.

        Returns
        -------
        QueryPrediction
            Best confidence and query id of every box.
        """
        if query_probabilities.ndim != 3 or query_probabilities.shape[-1] == 0:
            raise ValueError(f"query_probabilities must have shape (B, N, Q). got {tuple(query_probabilities.shape)}")
        confidences, query_ids = query_probabilities.float().max(dim=-1)
        return cls(confidences=confidences, query_ids=query_ids)

    def to_raw_detections(self, normalized_cxcywh: torch.Tensor) -> RawDetections:
        """
        Pair the predictions with their boxes as runtime-independent arrays.

        Parameters
        ----------
        normalized_cxcywh : torch.Tensor
            Predicted boxes as normalized (cx, cy, w, h), shape (B, N, 4).

        Returns
        -------
        RawDetections
            Boxes, confidences and query ids on the CPU.
        """
        return RawDetections(
            normalized_cxcywh=normalized_cxcywh.float().cpu().numpy().astype(np.float64),
            confidences=self.confidences.float().cpu().numpy().astype(np.float64),
            query_ids=self.query_ids.cpu().numpy().astype(np.int64),
        )
