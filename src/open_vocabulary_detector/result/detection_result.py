from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from geometry import BboxCalculator, Box2D, Box2dConverter, Box2DFormat, Boxes2D

from ..array_types import BoolArray, FloatArray, IntArray
from .detection import Detection
from .image_size import ImageSize


@dataclass(frozen=True, eq=False)
class DetectionResult:
    """
    Model-independent detections for one image.

    Attributes
    ----------
    boxes : Boxes2D
        Bounding boxes in absolute XYXY pixel coordinates, shape (N, 4).
    confidences : FloatArray
        Confidence of each detection in ``[0, 1]``, shape (N,).
    class_ids : IntArray
        Class indices into ``class_names``, shape (N,).
    class_names : tuple[str, ...]
        Class names of the prompt; ``class_names[class_id]`` is the label.
    image_size : ImageSize
        Size of the image the boxes refer to.

    Raises
    ------
    ValueError
        If array lengths disagree, boxes are not XYXY, or a class id is out of range.
    """

    boxes: Boxes2D
    confidences: FloatArray
    class_ids: IntArray
    class_names: tuple[str, ...]
    image_size: ImageSize

    def __post_init__(self) -> None:
        if self.boxes.box_format is not Box2DFormat.XYXY:
            raise ValueError(f"boxes must be in XYXY format. got {self.boxes.box_format}")
        detection_count: int = len(self.boxes)
        if self.confidences.shape != (detection_count,):
            raise ValueError(f"confidences must have shape ({detection_count},). got {self.confidences.shape}")
        if self.class_ids.shape != (detection_count,):
            raise ValueError(f"class_ids must have shape ({detection_count},). got {self.class_ids.shape}")
        if detection_count and (self.class_ids.min() < 0 or self.class_ids.max() >= len(self.class_names)):
            raise ValueError(f"class_ids must be in [0, {len(self.class_names)}). got {self.class_ids}")

    @classmethod
    def from_xyxy(
        cls,
        xyxy: FloatArray,
        confidences: FloatArray,
        class_ids: IntArray,
        class_names: tuple[str, ...],
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
        class_ids : IntArray
            Class indices into ``class_names``, shape (N,).
        class_names : tuple[str, ...]
            Class names of the prompt.
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
            class_ids=np.asarray(class_ids, dtype=np.int64)[is_valid],
            class_names=class_names,
            image_size=image_size,
        )

    @classmethod
    def from_normalized_cxcywh(
        cls,
        normalized_cxcywh: FloatArray,
        confidences: FloatArray,
        class_ids: IntArray,
        class_names: tuple[str, ...],
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
        class_ids : IntArray
            Class indices into ``class_names``, shape (N,).
        class_names : tuple[str, ...]
            Class names of the prompt.
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
            xyxy=xyxy, confidences=confidences, class_ids=class_ids, class_names=class_names, image_size=image_size
        )

    @classmethod
    def empty(cls, class_names: tuple[str, ...], image_size: ImageSize) -> "DetectionResult":
        """
        Build a result without detections.

        Parameters
        ----------
        class_names : tuple[str, ...]
            Class names of the prompt.
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
            class_ids=np.zeros((0,), dtype=np.int64),
            class_names=class_names,
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

    def __len__(self) -> int:
        return len(self.boxes)

    def __iter__(self) -> Iterator[Detection]:
        xyxy: FloatArray = self.xyxy
        for index in range(len(self)):
            class_id: int = int(self.class_ids[index])
            yield Detection(
                box=Box2D.register(value=xyxy[index], box2d_format=Box2DFormat.XYXY),
                confidence=float(self.confidences[index]),
                class_id=class_id,
                class_name=self.class_names[class_id],
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
            class_ids=self.class_ids[indices],
            class_names=self.class_names,
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
            If True, suppress across classes; otherwise only within each class.

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
            is_same_class: BoolArray = self.class_ids[:, None] == self.class_ids[None, :]
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
