from dataclasses import dataclass

import numpy as np
import torch

from ..result import RawDetections


@dataclass(frozen=True, eq=False)
class ClassPrediction:
    """
    Best class of every query box in a mini-batch.

    Attributes
    ----------
    confidences : torch.Tensor
        Probability of the best class, shape (B, Q).
    class_ids : torch.Tensor
        Index of the best class, shape (B, Q).

    Raises
    ------
    ValueError
        If ``confidences`` is not two-dimensional or the shapes differ.
    """

    confidences: torch.Tensor
    class_ids: torch.Tensor

    def __post_init__(self) -> None:
        if self.confidences.ndim != 2 or self.confidences.shape != self.class_ids.shape:
            raise ValueError(
                f"confidences and class_ids must share shape (B, Q). got {tuple(self.confidences.shape)} "
                + f"and {tuple(self.class_ids.shape)}"
            )

    @classmethod
    def from_probabilities(cls, class_probabilities: torch.Tensor) -> "ClassPrediction":
        """
        Pick the most probable class of every query box.

        Parameters
        ----------
        class_probabilities : torch.Tensor
            Per-class probabilities, shape (B, Q, C).

        Raises
        ------
        ValueError
            If ``class_probabilities`` is not three-dimensional or has no class.

        Returns
        -------
        ClassPrediction
            Best confidence and class id of every query box.
        """
        if class_probabilities.ndim != 3 or class_probabilities.shape[-1] == 0:
            raise ValueError(f"class_probabilities must have shape (B, Q, C). got {tuple(class_probabilities.shape)}")
        confidences, class_ids = class_probabilities.float().max(dim=-1)
        return cls(confidences=confidences, class_ids=class_ids)

    def to_raw_detections(self, normalized_cxcywh: torch.Tensor) -> RawDetections:
        """
        Pair the predictions with their query boxes as runtime-independent arrays.

        Parameters
        ----------
        normalized_cxcywh : torch.Tensor
            Query boxes as normalized (cx, cy, w, h), shape (B, Q, 4).

        Returns
        -------
        RawDetections
            Boxes, confidences and class ids on the CPU.
        """
        return RawDetections(
            normalized_cxcywh=normalized_cxcywh.float().cpu().numpy().astype(np.float64),
            confidences=self.confidences.float().cpu().numpy().astype(np.float64),
            class_ids=self.class_ids.cpu().numpy().astype(np.int64),
        )
