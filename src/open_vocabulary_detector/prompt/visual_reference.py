from dataclasses import dataclass

import numpy as np
from PIL import Image

from geometry import Box2DFormat, Boxes2D

from ..array_types import FloatArray, IntArray


@dataclass(frozen=True, eq=False)
class VisualReference:
    """
    Reference image with boxes showing instances of prompt classes.

    References compare by identity, so a prompt reusing the same reference objects reuses its cached encoding.

    Attributes
    ----------
    image : Image.Image
        Reference image.
    boxes : Boxes2D
        Boxes around the shown instances in absolute XYXY pixel coordinates, shape (N, 4).
    class_ids : IntArray
        Prompt class id of each box, shape (N,).

    Raises
    ------
    ValueError
        If there is no box, boxes are not XYXY or lie outside the image,
        or the number of class ids differs from the number of boxes.
    """

    image: Image.Image
    boxes: Boxes2D
    class_ids: IntArray

    def __post_init__(self) -> None:
        box_count: int = len(self.boxes)
        if box_count == 0:
            raise ValueError("a visual reference needs at least one box.")
        if self.boxes.box_format is not Box2DFormat.XYXY:
            raise ValueError(f"boxes must be in XYXY format. got {self.boxes.box_format}")
        if self.class_ids.shape != (box_count,):
            raise ValueError(f"class_ids must have shape ({box_count},). got {self.class_ids.shape}")
        width, height = self.image.size
        xyxy: FloatArray = self.xyxy
        if bool(((xyxy[:, :2] < 0).any()) or (xyxy[:, 2] > width).any() or (xyxy[:, 3] > height).any()):
            raise ValueError(f"boxes must lie inside the {width}x{height} image. got {xyxy.tolist()}")

    @property
    def xyxy(self) -> FloatArray:
        """
        Boxes as a plain array.

        Returns
        -------
        FloatArray
            Absolute XYXY pixel boxes, shape (N, 4).
        """
        return np.asarray(self.boxes.value, dtype=np.float64).reshape(-1, 4)

    @property
    def normalized_xyxy(self) -> FloatArray:
        """
        Boxes relative to the image size.

        Returns
        -------
        FloatArray
            XYXY boxes in ``[0, 1]``, shape (N, 4).
        """
        width, height = self.image.size
        return self.xyxy / np.array([width, height, width, height], dtype=np.float64)
