from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from geometry import BboxCalculator, Box2D, Box2dConverter, Box2DFormat, Boxes2D

from ..array_types import BoolArray, FloatArray, IntArray
from ..prompt import Prompt
from .detection import Detection
from .image_size import ImageSize


@dataclass(frozen=True, eq=False)
class DetectionResult:
    """
    Model-independent detections for one image.

    Each detection keeps the prompt query that matched it; its class is the class of that query,
    so several queries (e.g. ``"suv"``, ``"taxi"`` and a van reference image) report the same class (``"car"``).

    Attributes
    ----------
    boxes : Boxes2D
        Bounding boxes in absolute XYXY pixel coordinates, shape (N, 4).
    confidences : FloatArray
        Confidence of each detection in ``[0, 1]``, shape (N,).
    query_ids : IntArray
        Matched query of each detection, indexing ``prompt.queries``, shape (N,).
    prompt : Prompt
        Prompt whose queries were scored.
    image_size : ImageSize
        Size of the image the boxes refer to.

    Raises
    ------
    ValueError
        If array lengths disagree, boxes are not XYXY, or a query id is out of range.
    """

    boxes: Boxes2D
    confidences: FloatArray
    query_ids: IntArray
    prompt: Prompt
    image_size: ImageSize

    def __post_init__(self) -> None:
        if self.boxes.box_format is not Box2DFormat.XYXY:
            raise ValueError(f"boxes must be in XYXY format. got {self.boxes.box_format}")
        detection_count: int = len(self.boxes)
        if self.confidences.shape != (detection_count,):
            raise ValueError(f"confidences must have shape ({detection_count},). got {self.confidences.shape}")
        if self.query_ids.shape != (detection_count,):
            raise ValueError(f"query_ids must have shape ({detection_count},). got {self.query_ids.shape}")
        query_count: int = len(self.prompt.queries)
        if detection_count and (self.query_ids.min() < 0 or self.query_ids.max() >= query_count):
            raise ValueError(f"query_ids must be in [0, {query_count}). got {self.query_ids}")

    @classmethod
    def from_xyxy(
        cls,
        xyxy: FloatArray,
        confidences: FloatArray,
        query_ids: IntArray,
        prompt: Prompt,
        image_size: ImageSize,
    ) -> "DetectionResult":
        """
        Build a result from raw XYXY pixel boxes.

        Boxes are clipped to the image, and boxes with zero width or height
        after clipping are dropped together with their confidences and class ids.

        Parameters
        ----------
        xyxy : FloatArray
            Absolute pixel boxes, shape (N, 4).
        confidences : FloatArray
            Confidence of each detection in ``[0, 1]``, shape (N,).
        query_ids : IntArray
            Matched query of each detection, indexing ``prompt.queries``, shape (N,).
        prompt : Prompt
            Prompt whose queries were scored.
        image_size : ImageSize
            Size of the source image.

        Returns
        -------
        DetectionResult
            Sanitized detections.
        """
        boxes_xyxy: FloatArray = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4).copy()
        boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clip(0.0, float(image_size.width))
        boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clip(0.0, float(image_size.height))
        is_valid: BoolArray = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
        return cls(
            boxes=Boxes2D.register(value=boxes_xyxy[is_valid], box2d_format=Box2DFormat.XYXY),
            confidences=np.asarray(confidences, dtype=np.float64)[is_valid],
            query_ids=np.asarray(query_ids, dtype=np.int64)[is_valid],
            prompt=prompt,
            image_size=image_size,
        )

    @classmethod
    def from_normalized_cxcywh(
        cls,
        normalized_cxcywh: FloatArray,
        confidences: FloatArray,
        query_ids: IntArray,
        prompt: Prompt,
        image_size: ImageSize,
    ) -> "DetectionResult":
        """
        Build a result from center-size boxes normalized to ``[0, 1]``.

        Parameters
        ----------
        normalized_cxcywh : FloatArray
            Boxes as (cx, cy, w, h) relative to the image size, shape (N, 4).
        confidences : FloatArray
            Confidence of each detection in ``[0, 1]``, shape (N,).
        query_ids : IntArray
            Matched query of each detection, indexing ``prompt.queries``, shape (N,).
        prompt : Prompt
            Prompt whose queries were scored.
        image_size : ImageSize
            Size of the source image.

        Returns
        -------
        DetectionResult
            Detections in absolute XYXY pixel coordinates.
        """
        scale: FloatArray = np.array(
            [image_size.width, image_size.height, image_size.width, image_size.height], dtype=np.float64
        )
        absolute_cxcywh: FloatArray = np.asarray(normalized_cxcywh, dtype=np.float64).reshape(-1, 4) * scale
        xyxy: FloatArray = np.asarray(
            Box2dConverter.convert_format(absolute_cxcywh, Box2DFormat.CXCYWH, Box2DFormat.XYXY), dtype=np.float64
        )
        return cls.from_xyxy(
            xyxy=xyxy, confidences=confidences, query_ids=query_ids, prompt=prompt, image_size=image_size
        )

    @classmethod
    def empty(cls, prompt: Prompt, image_size: ImageSize) -> "DetectionResult":
        """
        Build a result without detections.

        Parameters
        ----------
        prompt : Prompt
            Prompt whose queries were scored.
        image_size : ImageSize
            Size of the source image.

        Returns
        -------
        DetectionResult
            Result with zero detections.
        """
        return cls.from_xyxy(
            xyxy=np.zeros((0, 4), dtype=np.float64),
            confidences=np.zeros((0,), dtype=np.float64),
            query_ids=np.zeros((0,), dtype=np.int64),
            prompt=prompt,
            image_size=image_size,
        )

    @property
    def xyxy(self) -> FloatArray:
        """
        Boxes as a plain array.

        Returns
        -------
        FloatArray
            Absolute XYXY pixel boxes, shape (N, 4).
        """
        return np.asarray(self.boxes.value, dtype=np.float64)

    @property
    def class_names(self) -> tuple[str, ...]:
        """
        Output class names of the prompt.

        Returns
        -------
        tuple[str, ...]
            Names in class-id order.
        """
        return self.prompt.class_names

    @property
    def class_ids(self) -> IntArray:
        """
        Output class of each detection, i.e. the class of its matched query.

        Returns
        -------
        IntArray
            Indices into ``class_names``, shape (N,).
        """
        return self.prompt.class_ids_of(self.query_ids)

    def __len__(self) -> int:
        return len(self.boxes)

    def __iter__(self) -> Iterator[Detection]:
        xyxy: FloatArray = self.xyxy
        for index in range(len(self)):
            query_id: int = int(self.query_ids[index])
            class_id: int = self.prompt.query_class_ids[query_id]
            yield Detection(
                box=Box2D.register(value=xyxy[index], box2d_format=Box2DFormat.XYXY),
                confidence=float(self.confidences[index]),
                class_id=class_id,
                class_name=self.class_names[class_id],
                matched_query=self.prompt.queries[query_id],
            )

    def select(self, indices: BoolArray | IntArray) -> "DetectionResult":
        """
        Keep a subset of detections.

        Parameters
        ----------
        indices : BoolArray | IntArray
            Boolean mask of shape (N,) or integer indices.

        Returns
        -------
        DetectionResult
            Detections at the given indices, in the given order.
        """
        return DetectionResult(
            boxes=Boxes2D.register(value=self.xyxy[indices].reshape(-1, 4), box2d_format=Box2DFormat.XYXY),
            confidences=self.confidences[indices],
            query_ids=self.query_ids[indices],
            prompt=self.prompt,
            image_size=self.image_size,
        )

    def filter_by_confidence(self, confidence_threshold: float) -> "DetectionResult":
        """
        Keep detections whose confidence is at least ``confidence_threshold``.

        Parameters
        ----------
        confidence_threshold : float
            Minimum confidence to keep.

        Returns
        -------
        DetectionResult
            Filtered detections.
        """
        return self.select(self.confidences >= confidence_threshold)

    def sort_by_confidence(self) -> "DetectionResult":
        """
        Order detections by descending confidence.

        Returns
        -------
        DetectionResult
            Sorted detections.
        """
        order: IntArray = np.argsort(-self.confidences, kind="stable").astype(np.int64)
        return self.select(order)

    def non_maximum_suppression(self, iou_threshold: float, is_class_agnostic: bool = False) -> "DetectionResult":
        """
        Remove overlapping detections, keeping the most confident box.

        Parameters
        ----------
        iou_threshold : float
            Boxes overlapping a kept box with IoU above this value are removed.
        is_class_agnostic : bool, optional
            If True, suppress across classes; otherwise only within each class, so boxes matched by different
            queries of one class suppress each other.

        Raises
        ------
        ValueError
            If ``iou_threshold`` is outside ``[0, 1]``.

        Returns
        -------
        DetectionResult
            Kept detections ordered by descending confidence.
        """
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in [0, 1]. got {iou_threshold}")
        if not len(self):
            return self
        overlaps: FloatArray = np.asarray(BboxCalculator.compute_iou(self.xyxy, self.xyxy), dtype=np.float64)
        if not is_class_agnostic:
            class_ids: IntArray = self.class_ids
            is_same_class: BoolArray = class_ids[:, None] == class_ids[None, :]
            overlaps = np.where(is_same_class, overlaps, 0.0)
        is_suppressed: BoolArray = np.zeros(len(self), dtype=np.bool_)
        kept_indices: list[int] = []
        for index in np.argsort(-self.confidences, kind="stable").tolist():
            if is_suppressed[index]:
                continue
            kept_indices.append(index)
            is_suppressed |= overlaps[index] > iou_threshold
        return self.select(np.array(kept_indices, dtype=np.int64))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(detections={len(self)}, "
            + f"image_size=({self.image_size.width}, {self.image_size.height}), "
            + f"class_names={self.class_names})"
        )
