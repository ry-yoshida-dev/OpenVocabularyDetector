from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import ClassVar

from PIL import Image

from .array_types import BoolArray
from .backend import OVDBackend
from .options import DetectionThresholds
from .prompt import Prompt, PromptKind
from .result import DetectionResult, ImageSize, RawDetections
from .settings import OVDSettings


class OpenVocabularyDetector(ABC):
    """
    Common base of every open-vocabulary detector, independent of the inference runtime.

    Subclasses declare the ``BACKEND`` they implement, which defines the prompt kinds they accept;
    a prompt of another kind is rejected before any computation. The base owns the settings,
    mini-batching and the post-processing shared by every backend (confidence threshold, optional
    class-wise NMS, sort by confidence). How the model runs (PyTorch, ONNX Runtime, TensorRT, ...)
    is up to each subclass and its runtime.

    Where a backend can, query embeddings are cached per class name and per visual reference,
    so prompts sharing class names or references do not recompute them.

    Attributes
    ----------
    settings : OVDSettings
        Backend, weights, batching, device and thresholds.
    """

    BACKEND: ClassVar[OVDBackend]

    def __init__(self, settings: OVDSettings) -> None:
        """
        Parameters
        ----------
        settings : OVDSettings
            Backend, weights, batching, device and thresholds.

        Raises
        ------
        ValueError
            If the settings are for another backend.
        """
        if settings.backend is not self.BACKEND:
            raise ValueError(f"{type(self).__name__} needs {self.BACKEND} settings. got {settings.backend}")
        self.settings: OVDSettings = settings

    @property
    def supported_prompt_kinds(self) -> frozenset[PromptKind]:
        """
        Prompt kinds the detector accepts.

        Returns
        -------
        frozenset[PromptKind]
            Kinds supported by ``BACKEND``.
        """
        return self.BACKEND.supported_prompt_kinds

    @property
    def batch_size(self) -> int:
        """
        Number of images processed per forward pass.

        Returns
        -------
        int
            Mini-batch size used by ``detect_images``.
        """
        return self.settings.batch_size

    @property
    def thresholds(self) -> DetectionThresholds:
        """
        Post-processing thresholds in use.

        Returns
        -------
        DetectionThresholds
            Configured thresholds.
        """
        return self.settings.thresholds

    @abstractmethod
    def _detect_mini_batch(self, images: Sequence[Image.Image], prompt: Prompt) -> list[DetectionResult]:
        """
        Run one forward pass over at most ``batch_size`` RGB images.

        Parameters
        ----------
        images : Sequence[Image.Image]
            RGB images.
        prompt : Prompt
            Classes to detect.

        Returns
        -------
        list[DetectionResult]
            One result per image, in input order.
        """

    def detect(self, image: Image.Image, prompt: Prompt) -> DetectionResult:
        """
        Detect the prompt classes in a single image.

        Parameters
        ----------
        image : Image.Image
            Input image.
        prompt : Prompt
            Classes to detect.

        Returns
        -------
        DetectionResult
            Detections for the image.
        """
        return self.detect_images([image], prompt)[0]

    def detect_images(self, images: Sequence[Image.Image], prompt: Prompt) -> list[DetectionResult]:
        """
        Detect the prompt classes in many images, processed in mini-batches.

        Parameters
        ----------
        images : Sequence[Image.Image]
            Input images of arbitrary size and mode.
        prompt : Prompt
            Classes to detect.

        Raises
        ------
        ValueError
            If ``images`` is empty or the prompt kind is not supported.

        Returns
        -------
        list[DetectionResult]
            One result per image, in input order.
        """
        if not images:
            raise ValueError("images must contain at least one image.")
        self._validate_prompt(prompt)
        results: list[DetectionResult] = []
        for mini_batch in self._mini_batches(images):
            results.extend(self._detect_mini_batch([self._to_rgb(image) for image in mini_batch], prompt))
        return results

    def _validate_prompt(self, prompt: Prompt) -> None:
        if prompt.kind not in self.supported_prompt_kinds:
            raise ValueError(
                f"{type(self).__name__} does not support {prompt.kind} prompts. "
                + f"supported: {sorted(self.supported_prompt_kinds)}"
            )

    def _mini_batches(self, images: Sequence[Image.Image]) -> Iterator[Sequence[Image.Image]]:
        for start in range(0, len(images), self.batch_size):
            yield images[start : start + self.batch_size]

    @staticmethod
    def _to_rgb(image: Image.Image) -> Image.Image:
        return image if image.mode == "RGB" else image.convert("RGB")

    def _postprocess(
        self,
        raw_detections: RawDetections,
        class_names: tuple[str, ...],
        image_sizes: Sequence[ImageSize],
    ) -> list[DetectionResult]:
        """
        Apply the confidence threshold, optional class-wise NMS and sorting to raw model output.

        Parameters
        ----------
        raw_detections : RawDetections
            Per-query boxes, confidences and class ids of a mini-batch.
        class_names : tuple[str, ...]
            Class names of the prompt.
        image_sizes : Sequence[ImageSize]
            Size of each source image, one per batch entry.

        Raises
        ------
        ValueError
            If the number of image sizes differs from the batch size.

        Returns
        -------
        list[DetectionResult]
            One result per image, ordered by descending confidence.
        """
        if len(image_sizes) != len(raw_detections):
            raise ValueError(f"expected {len(raw_detections)} image sizes. got {len(image_sizes)}")
        thresholds: DetectionThresholds = self.thresholds
        results: list[DetectionResult] = []
        for index, image_size in enumerate(image_sizes):
            is_kept: BoolArray = raw_detections.confidences[index] >= thresholds.confidence_threshold
            result: DetectionResult = DetectionResult.from_normalized_cxcywh(
                normalized_cxcywh=raw_detections.normalized_cxcywh[index][is_kept],
                confidences=raw_detections.confidences[index][is_kept],
                class_ids=raw_detections.class_ids[index][is_kept],
                class_names=class_names,
                image_size=image_size,
            )
            if thresholds.nms_iou_threshold is not None:
                result = result.non_maximum_suppression(thresholds.nms_iou_threshold)
            results.append(result.sort_by_confidence())
        return results
