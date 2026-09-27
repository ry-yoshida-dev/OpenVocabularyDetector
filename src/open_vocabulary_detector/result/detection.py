from dataclasses import dataclass

from geometry import Box2D

from ..prompt import PromptQuery


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
        Output class name corresponding to ``class_id``.
    matched_query : PromptQuery
        Query that matched the box, e.g. ``TextQuery("taxi")`` or a ``VisualQuery`` for class ``"car"``.
    """

    box: Box2D
    confidence: float
    class_id: int
    class_name: str
    matched_query: PromptQuery
