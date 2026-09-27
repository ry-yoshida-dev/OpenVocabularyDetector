from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import pytest
from geometry import Box2DFormat, Boxes2D
from PIL import Image

from open_vocabulary_detector import (
    DetectionResult,
    DetectionThresholds,
    DetectorBackend,
    DetectorSettings,
    Device,
    ImageSize,
    OpenVocabularyDetector,
    Prompt,
    TextQuery,
    VisualQuery,
    VisualReference,
)
from open_vocabulary_detector.result import RawDetections

QUERY_BOXES: list[list[float]] = [
    [0.5, 0.5, 0.2, 0.2],
    [0.51, 0.51, 0.2, 0.2],
    [0.2, 0.2, 0.1, 0.1],
    [0.8, 0.8, 0.1, 0.1],
]
QUERY_CONFIDENCES: list[float] = [0.7, 0.9, 0.1, 0.6]
QUERY_IDS: list[int] = [0, 0, 0, 1]


class StubDetector(OpenVocabularyDetector):
    BACKEND: ClassVar[DetectorBackend] = DetectorBackend.GROUNDING_DINO

    def __init__(self, settings: DetectorSettings) -> None:
        super().__init__(settings)
        self.mini_batch_sizes: list[int] = []
        self.query_ids: list[int] = QUERY_IDS

    def _detect_mini_batch(self, images: Sequence[Image.Image], prompt: Prompt) -> list[DetectionResult]:
        self.mini_batch_sizes.append(len(images))
        batch_size: int = len(images)
        return self._postprocess(
            raw_detections=RawDetections(
                normalized_cxcywh=np.array([QUERY_BOXES] * batch_size, dtype=np.float64),
                confidences=np.array([QUERY_CONFIDENCES] * batch_size, dtype=np.float64),
                query_ids=np.array([self.query_ids] * batch_size, dtype=np.int64),
            ),
            prompt=prompt,
            image_sizes=[ImageSize.from_image(image) for image in images],
        )


def build_settings(
    backend: DetectorBackend = DetectorBackend.GROUNDING_DINO, nms_iou_threshold: float | None = None
) -> DetectorSettings:
    return DetectorSettings(
        backend=backend,
        weights_path="stub",
        thresholds=DetectionThresholds(confidence_threshold=0.5, nms_iou_threshold=nms_iou_threshold),
        batch_size=2,
        device=Device.CPU,
    )


def test_detect_images_thresholds_sorts_and_batches() -> None:
    detector: StubDetector = StubDetector(build_settings())
    images: list[Image.Image] = [Image.new("L", (100, 50))] * 5
    results: list[DetectionResult] = detector.detect_images(images, Prompt.from_class_names(("cat", "dog")))
    assert detector.mini_batch_sizes == [2, 2, 1]
    assert len(results) == 5
    assert results[0].confidences.tolist() == pytest.approx([0.9, 0.7, 0.6])
    assert results[0].class_ids.tolist() == [0, 0, 1]
    assert results[0].image_size == ImageSize(width=100, height=50)


def test_detect_applies_class_wise_nms() -> None:
    detector: StubDetector = StubDetector(build_settings(nms_iou_threshold=0.5))
    result: DetectionResult = detector.detect(Image.new("RGB", (100, 50)), Prompt.from_class_names(("cat", "dog")))
    assert result.confidences.tolist() == pytest.approx([0.9, 0.6])


def test_queries_of_one_class_are_merged_by_class_wise_nms() -> None:
    detector: StubDetector = StubDetector(build_settings(nms_iou_threshold=0.5))
    detector.query_ids = [1, 2, 0, 3]
    prompt: Prompt = Prompt.from_texts({"car": ("car", "suv", "taxi"), "dog": ("dog",)})
    result: DetectionResult = detector.detect(Image.new("RGB", (100, 50)), prompt)
    assert result.confidences.tolist() == pytest.approx([0.9, 0.6])
    assert result.class_ids.tolist() == [0, 1]
    assert [(detection.class_name, detection.matched_query) for detection in result] == [
        ("car", TextQuery("taxi")),
        ("dog", TextQuery("dog")),
    ]


def test_queries_are_kept_without_nms() -> None:
    detector: StubDetector = StubDetector(build_settings())
    detector.query_ids = [1, 2, 0, 3]
    prompt: Prompt = Prompt.from_texts({"car": ("car", "suv", "taxi"), "dog": ("dog",)})
    result: DetectionResult = detector.detect(Image.new("RGB", (100, 50)), prompt)
    assert [detection.matched_query for detection in result] == [TextQuery("taxi"), TextQuery("suv"), TextQuery("dog")]
    assert result.class_names == ("car", "dog")


def build_mixed_prompt() -> Prompt:
    reference: VisualReference = VisualReference(
        image=Image.new("RGB", (8, 8)),
        boxes=Boxes2D.register(value=np.array([[0.0, 0.0, 4.0, 4.0]]), box2d_format=Box2DFormat.XYXY),
    )
    return Prompt(
        {
            "car": (TextQuery("car"), TextQuery("suv"), VisualQuery((reference,))),
            "dog": (TextQuery("dog"),),
        }
    )


def test_unsupported_query_kind_is_rejected_before_inference() -> None:
    detector: StubDetector = StubDetector(build_settings())
    with pytest.raises(ValueError, match=r"does not support \['visual'\] queries"):
        detector.detect(Image.new("RGB", (100, 50)), build_mixed_prompt())
    assert detector.mini_batch_sizes == []
    assert detector.supported_prompt_kinds == DetectorBackend.GROUNDING_DINO.supported_prompt_kinds


class VisualStubDetector(StubDetector):
    BACKEND: ClassVar[DetectorBackend] = DetectorBackend.OWL_VIT


def test_text_and_visual_queries_of_one_class_are_merged() -> None:
    detector: VisualStubDetector = VisualStubDetector(build_settings(DetectorBackend.OWL_VIT, nms_iou_threshold=0.5))
    detector.query_ids = [1, 2, 0, 3]
    prompt: Prompt = build_mixed_prompt()
    result: DetectionResult = detector.detect(Image.new("RGB", (100, 50)), prompt)
    assert result.class_ids.tolist() == [0, 1]
    assert [(detection.class_name, detection.matched_query) for detection in result] == [
        ("car", prompt.queries[2]),
        ("dog", TextQuery("dog")),
    ]


def test_settings_of_another_backend_are_rejected() -> None:
    with pytest.raises(ValueError, match="needs grounding_dino settings"):
        StubDetector(build_settings(backend=DetectorBackend.YOLOE))


def test_raw_detections_validate_shapes() -> None:
    with pytest.raises(ValueError, match="query_ids must have shape"):
        RawDetections(
            normalized_cxcywh=np.zeros((1, 2, 4)),
            confidences=np.zeros((1, 2)),
            query_ids=np.zeros((1, 3), dtype=np.int64),
        )
