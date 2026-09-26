import numpy as np
import pytest

from geometry import Box2DFormat
from open_vocabulary_detector import DetectionResult, ImageSize

CLASS_NAMES: tuple[str, ...] = ("cat", "dog")
IMAGE_SIZE: ImageSize = ImageSize(width=100, height=50)


def build_result() -> DetectionResult:
    return DetectionResult.from_xyxy(
        xyxy=np.array([[10, 10, 40, 40], [12, 12, 42, 42], [-5, 0, 120, 60], [60, 10, 60, 20]], dtype=np.float64),
        confidences=np.array([0.9, 0.8, 0.3, 0.7], dtype=np.float64),
        class_ids=np.array([0, 0, 1, 1], dtype=np.int64),
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
    )


def test_from_xyxy_clips_and_drops_degenerate_boxes() -> None:
    result: DetectionResult = build_result()
    assert len(result) == 3
    assert result.boxes.box_format is Box2DFormat.XYXY
    np.testing.assert_allclose(result.xyxy[2], [0, 0, 100, 50])
    assert result.class_ids.tolist() == [0, 0, 1]


def test_from_normalized_cxcywh() -> None:
    result: DetectionResult = DetectionResult.from_normalized_cxcywh(
        normalized_cxcywh=np.array([[0.5, 0.5, 0.2, 0.4]], dtype=np.float64),
        confidences=np.array([0.5], dtype=np.float64),
        class_ids=np.array([1], dtype=np.int64),
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
    )
    np.testing.assert_allclose(result.xyxy, [[40, 15, 60, 35]], rtol=1e-6)


def test_filters_and_sort() -> None:
    result: DetectionResult = build_result()
    assert len(result.filter_by_confidence(0.5)) == 2
    assert result.sort_by_confidence().confidences.tolist() == [0.9, 0.8, 0.3]


def test_non_maximum_suppression() -> None:
    result: DetectionResult = build_result()
    assert result.non_maximum_suppression(0.5).confidences.tolist() == [0.9, 0.3]
    assert result.non_maximum_suppression(0.1, is_class_agnostic=True).confidences.tolist() == [0.9]


def test_iteration_yields_geometry_boxes() -> None:
    detections = list(build_result())
    assert detections[0].class_name == "cat"
    assert float(detections[0].box.area) == pytest.approx(900.0)


def test_empty_and_validation() -> None:
    assert len(DetectionResult.empty(CLASS_NAMES, IMAGE_SIZE)) == 0
    with pytest.raises(ValueError):
        DetectionResult.from_xyxy(
            xyxy=np.array([[0, 0, 1, 1]], dtype=np.float64),
            confidences=np.array([0.5], dtype=np.float64),
            class_ids=np.array([2], dtype=np.int64),
            class_names=CLASS_NAMES,
            image_size=IMAGE_SIZE,
        )
