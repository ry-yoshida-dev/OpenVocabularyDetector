from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import ClassVar

from PIL import Image

from .array_types import BoolArray
from .options import DetectionThresholds, DetectorBackend
from .prompt import Prompt, PromptKind
from .result import DetectionResult, ImageSize, RawDetections
from .settings import DetectorSettings


class OpenVocabularyDetector(ABC):
    """
    Common base of every open-vocabulary detector, independent of the inference runtime.

    Subclasses declare the ``BACKEND`` they implement, which defines the query kinds they accept;
    a prompt using another kind is rejected before any computation. The base owns the settings,
    mini-batching and the post-processing shared by every backend (confidence threshold, optional
    class-wise NMS, sort by confidence). How the model runs (PyTorch, ONNX Runtime, TensorRT, ...)
    is up to each subclass and its runtime.

    Models score every query of the prompt, and each box keeps its best query. NMS then runs per output class,
    so boxes found by different queries of one class (``"suv"``, ``"taxi"``, a van reference image) collapse into
    the most confident one, which still records the query that matched.

    Where a backend can, query embeddings are cached per text query and per visual reference,
    so prompts sharing queries or references do not recompute them.

    Attributes
    ----------
    settings : DetectorSettings
        Backend, weights, batching, device and thresholds.
    """

    BACKEND: ClassVar[DetectorBackend]

    def __init__(self, settings: DetectorSettings) -> None:
        """
        Parameters
        ----------
        settings : DetectorSettings
            Backend, weights, batching, device and thresholds.

        Raises
        ------
        ValueError
            If the settings are for another backend.
        """
        if settings.backend is not self.BACKEND:
            raise ValueError(f"{type(self).__name__} needs {self.BACKEND} settings. got {settings.backend}")
        self.settings: DetectorSettings = settings

    @property
    def supported_prompt_kinds(self) -> frozenset[PromptKind]:
        """
        Query kinds the detector accepts.

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
            If ``images`` is empty or the prompt uses an unsupported query kind.

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
        unsupported_kinds: frozenset[PromptKind] = prompt.kinds - self.supported_prompt_kinds
        if unsupported_kinds:
            raise ValueError(
                f"{type(self).__name__} does not support {sorted(kind.value for kind in unsupported_kinds)} queries. "
                + f"supported: {sorted(kind.value for kind in self.supported_prompt_kinds)}"
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
        prompt: Prompt,
        image_sizes: Sequence[ImageSize],
    ) -> list[DetectionResult]:
        """
        Apply the confidence threshold, then merge queries into classes, to raw model output.

        Parameters
        ----------
        raw_detections : RawDetections
            Per-box boxes, confidences and query ids of a mini-batch.
        prompt : Prompt
            Prompt whose queries the model scored.
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
        confidence_threshold: float = self.thresholds.confidence_threshold
        results: list[DetectionResult] = []
        for index, image_size in enumerate(image_sizes):
            is_kept: BoolArray = raw_detections.confidences[index] >= confidence_threshold
            result: DetectionResult = DetectionResult.from_normalized_cxcywh(
                normalized_cxcywh=raw_detections.normalized_cxcywh[index][is_kept],
                confidences=raw_detections.confidences[index][is_kept],
                query_ids=raw_detections.query_ids[index][is_kept],
                prompt=prompt,
                image_size=image_size,
            )
            results.append(self._suppress_and_sort(result))
        return results

    def _suppress_and_sort(self, result: DetectionResult) -> DetectionResult:
        """
        Apply the optional class-wise NMS and sort by confidence.

        NMS compares output classes, not queries, so overlapping boxes of different queries of one class
        keep only the most confident box.

        Parameters
        ----------
        result : DetectionResult
            Thresholded detections of one image.

        Returns
        -------
        DetectionResult
            Detections ordered by descending confidence.
        """
        nms_iou_threshold: float | None = self.thresholds.nms_iou_threshold
        if nms_iou_threshold is not None:
            result = result.non_maximum_suppression(nms_iou_threshold)
        return result.sort_by_confidence()
