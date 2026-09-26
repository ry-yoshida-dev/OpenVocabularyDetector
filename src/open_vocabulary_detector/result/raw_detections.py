from dataclasses import dataclass

from ..array_types import FloatArray, IntArray


@dataclass(frozen=True, eq=False)
class RawDetections:
    """
    Per-query model output of a mini-batch before post-processing, independent of the inference runtime.

    Attributes
    ----------
    normalized_cxcywh : FloatArray
        Query boxes as (cx, cy, w, h) relative to the image size, shape (B, Q, 4).
    confidences : FloatArray
        Confidence of the best class of each query, shape (B, Q).
    class_ids : IntArray
        Best class of each query, shape (B, Q).

    Raises
    ------
    ValueError
        If the shapes do not agree.
    """

    normalized_cxcywh: FloatArray
    confidences: FloatArray
    class_ids: IntArray

    def __post_init__(self) -> None:
        if self.confidences.ndim != 2:
            raise ValueError(f"confidences must have shape (B, Q). got {self.confidences.shape}")
        if self.class_ids.shape != self.confidences.shape:
            raise ValueError(f"class_ids must have shape {self.confidences.shape}. got {self.class_ids.shape}")
        if self.normalized_cxcywh.shape != (*self.confidences.shape, 4):
            raise ValueError(
                f"normalized_cxcywh must have shape {(*self.confidences.shape, 4)}. got {self.normalized_cxcywh.shape}"
            )

    def __len__(self) -> int:
        return int(self.confidences.shape[0])
