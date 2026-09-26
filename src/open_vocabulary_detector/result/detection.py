from dataclasses import dataclass

from geometry import Box2D


@dataclass(frozen=True)
class Detection:
    """
    Single detected object.

    Attributes
    ----------
    box : Box2D
        Bounding box in absolute XYXY pixel coordinates.
    confidence : float
        Confidence in ``[0, 1]``.
    class_id : int
        Index into the classes of the originating prompt.
    class_name : str
        Class name corresponding to ``class_id``.
    """

    box: Box2D
    confidence: float
    class_id: int
    class_name: str
