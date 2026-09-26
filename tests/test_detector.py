from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import pytest
from PIL import Image

from geometry import Box2DFormat, Boxes2D
from open_vocabulary_detector import (
    DetectionResult,
    DetectionThresholds,
    Device,
    ImageSize,
    OpenVocabularyDetector,
    OVDBackend,
    OVDSettings,
    Prompt,
    TextPrompt,
    TextVisualPrompt,
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
QUERY_CLASS_IDS: list[int] = [0, 0, 0, 1]


class StubDetector(OpenVocabularyDetector):
    BACKEND: ClassVar[OVDBackend] = OVDBackend.GROUNDING_DINO

    def __init__(self, settings: OVDSettings) -> None:
        super().__init__(settings)
        self.mini_batch_sizes: list[int] = []

    def _detect_mini_batch(self, images: Sequence[Image.Image], prompt: Prompt) -> list[DetectionResult]:
        self.mini_batch_sizes.append(len(images))
        batch_size: int = len(images)
        return self._postprocess(
            raw_detections=RawDetections(
                normalized_cxcywh=np.array([QUERY_BOXES] * batch_size, dtype=np.float64),
                confidences=np.array([QUERY_CONFIDENCES] * batch_size, dtype=np.float64),
                class_ids=np.array([QUERY_CLASS_IDS] * batch_size, dtype=np.int64),
            ),
            class_names=prompt.class_names,
            image_sizes=[ImageSize.from_image(image) for image in images],
        )


def build_settings(
    backend: OVDBackend = OVDBackend.GROUNDING_DINO, nms_iou_threshold: float | None = None
) -> OVDSettings:
    return OVDSettings(
        backend=backend,
        weights_path="stub",
        thresholds=DetectionThresholds(confidence_threshold=0.5, nms_iou_threshold=nms_iou_threshold),
        batch_size=2,
        device=Device.CPU,
    )


def test_detect_images_thresholds_sorts_and_batches() -> None:
    detector: StubDetector = StubDetector(build_settings())
    images: list[Image.Image] = [Image.new("L", (100, 50))] * 5
    results: list[DetectionResult] = detector.detect_images(images, TextPrompt(class_names=("cat", "dog")))
    assert detector.mini_batch_sizes == [2, 2, 1]
    assert len(results) == 5
    assert results[0].confidences.tolist() == pytest.approx([0.9, 0.7, 0.6])
    assert results[0].class_ids.tolist() == [0, 0, 1]
    assert results[0].image_size == ImageSize(width=100, height=50)


def test_detect_applies_class_wise_nms() -> None:
    detector: StubDetector = StubDetector(build_settings(nms_iou_threshold=0.5))
    result: DetectionResult = detector.detect(Image.new("RGB", (100, 50)), TextPrompt(class_names=("cat", "dog")))
    assert result.confidences.tolist() == pytest.approx([0.9, 0.6])


def test_unsupported_prompt_kind_is_rejected_before_inference() -> None:
    detector: StubDetector = StubDetector(build_settings())
    visual_prompt: TextVisualPrompt = TextVisualPrompt(
        class_names=("mug",),
        visual_references=(
            VisualReference(
                image=Image.new("RGB", (8, 8)),
                boxes=Boxes2D.register(value=np.array([[0.0, 0.0, 4.0, 4.0]]), box2d_format=Box2DFormat.XYXY),
                class_ids=np.array([0], dtype=np.int64),
            ),
        ),
    )
    with pytest.raises(ValueError, match="does not support text_visual prompts"):
        detector.detect(Image.new("RGB", (100, 50)), visual_prompt)
    assert detector.mini_batch_sizes == []
    assert detector.supported_prompt_kinds == OVDBackend.GROUNDING_DINO.supported_prompt_kinds


def test_settings_of_another_backend_are_rejected() -> None:
    with pytest.raises(ValueError, match="needs grounding_dino settings"):
        StubDetector(build_settings(backend=OVDBackend.YOLOE))


def test_raw_detections_validate_shapes() -> None:
    with pytest.raises(ValueError, match="class_ids must have shape"):
        RawDetections(
            normalized_cxcywh=np.zeros((1, 2, 4)),
            confidences=np.zeros((1, 2)),
            class_ids=np.zeros((1, 3), dtype=np.int64),
        )
