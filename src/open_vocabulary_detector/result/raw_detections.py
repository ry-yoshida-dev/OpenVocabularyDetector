from dataclasses import dataclass

from ..array_types import FloatArray, IntArray


@dataclass(frozen=True, eq=False)
class RawDetections:
    """
    Per-box model output of a mini-batch before post-processing, independent of the inference runtime.

    Attributes
    ----------
    normalized_cxcywh : FloatArray
        Predicted boxes as (cx, cy, w, h) relative to the image size, shape (B, N, 4).
    confidences : FloatArray
        Confidence of the best prompt query of each box, shape (B, N).
    query_ids : IntArray
        Best prompt query of each box, indexing ``Prompt.queries``, shape (B, N).

    Raises
    ------
    ValueError
        If the shapes do not agree.
    """

    normalized_cxcywh: FloatArray
    confidences: FloatArray
    query_ids: IntArray

    def __post_init__(self) -> None:
        if self.confidences.ndim != 2:
            raise ValueError(f"confidences must have shape (B, N). got {self.confidences.shape}")
        if self.query_ids.shape != self.confidences.shape:
            raise ValueError(f"query_ids must have shape {self.confidences.shape}. got {self.query_ids.shape}")
        if self.normalized_cxcywh.shape != (*self.confidences.shape, 4):
            raise ValueError(
                f"normalized_cxcywh must have shape {(*self.confidences.shape, 4)}. got {self.normalized_cxcywh.shape}"
            )

    def __len__(self) -> int:
        return int(self.confidences.shape[0])
