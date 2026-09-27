from dataclasses import dataclass

import numpy as np
from geometry import Box2DFormat, Boxes2D
from PIL import Image

from ..array_types import FloatArray


@dataclass(frozen=True, eq=False)
class VisualReference:
    """
    Reference image with boxes showing instances of one prompt class.

    References compare by identity, so a prompt reusing the same reference objects reuses its cached encoding.

    Attributes
    ----------
    image : Image.Image
        Reference image.
    boxes : Boxes2D
        Boxes around the shown instances in absolute XYXY pixel coordinates, shape (N, 4).

    Raises
    ------
    ValueError
        If there is no box, or boxes are not XYXY or lie outside the image.
    """

    image: Image.Image
    boxes: Boxes2D

    def __post_init__(self) -> None:
        if len(self.boxes) == 0:
            raise ValueError("a visual reference needs at least one box.")
        if self.boxes.box_format is not Box2DFormat.XYXY:
            raise ValueError(f"boxes must be in XYXY format. got {self.boxes.box_format}")
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
