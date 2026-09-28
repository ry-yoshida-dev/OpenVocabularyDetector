from abc import abstractmethod
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results

from ...detector import OpenVocabularyDetector
from ...prompt import Prompt
from ...result import DetectionResult, ImageSize
from ...runtime import TorchRuntime
from ...settings import DetectorSettings


class UltralyticsDetector(OpenVocabularyDetector):
    """
    Base of detectors whose queries are baked into an Ultralytics model.

    The model holds the queries of one prompt at a time, as its "classes". A prompt is applied to the model
    only when it differs from the active one, so reusing one prompt across
    many batches avoids re-configuring the model. Thresholding and query-wise NMS run
    inside Ultralytics (YOLOE would otherwise switch to class-agnostic NMS); the predicted query ids are then
    folded into output classes and NMS runs again per class, like every other backend.
    Ultralytics keeps at most 300 detections per image, a limit end-to-end heads fix inside the model.
    """

    DISABLED_NMS_IOU_THRESHOLD: float = 1.0
    HALF_PRECISION_QUANTIZE_BITS: int = 16

    def __init__(self, settings: DetectorSettings) -> None:
        """
        Parameters
        ----------
        settings : DetectorSettings
            Weights, batching, device and thresholds.
        """
        super().__init__(settings)
        self._runtime: TorchRuntime = TorchRuntime(settings)
        self._active_prompt: Prompt | None = None

    @property
    @abstractmethod
    def model(self) -> Model:
        """
        Underlying Ultralytics model.

        Returns
        -------
        Model
            Loaded model.
        """

    @abstractmethod
    def _apply_prompt(self, prompt: Prompt) -> None:
        """
        Configure the model to score the queries of a prompt, in query-id order.

        Parameters
        ----------
        prompt : Prompt
            Prompt to activate.
        """

    def _activate(self, prompt: Prompt) -> None:
        if self._active_prompt != prompt:
            self._apply_prompt(prompt)
            self._active_prompt = prompt

    def _detect_mini_batch(
        self,
        images: Sequence[Image.Image],
        prompt: Prompt,
    ) -> list[DetectionResult]:
        self._activate(prompt)
        nms_iou_threshold: float | None = self.thresholds.nms_iou_threshold
        predictions: list[Results] = [
            prediction
            for prediction in self.model.predict(
                source=list(images),
                conf=self.thresholds.confidence_threshold,
                iou=self.DISABLED_NMS_IOU_THRESHOLD if nms_iou_threshold is None else nms_iou_threshold,
                device=self._runtime.device.type,
                quantize=self.HALF_PRECISION_QUANTIZE_BITS if self.settings.is_half_precision_enabled else None,
                agnostic_nms=False,
                verbose=False,
            )
            if isinstance(prediction, Results)
        ]
        return [
            self._suppress_and_sort(self._build_result(prediction, prompt, ImageSize.from_image(image)))
            for prediction, image in zip(predictions, images, strict=True)
        ]

    @staticmethod
    def _build_result(
        prediction: Results,
        prompt: Prompt,
        image_size: ImageSize,
    ) -> DetectionResult:
        if prediction.boxes is None:
            return DetectionResult.empty(prompt=prompt, image_size=image_size)
        return DetectionResult.from_xyxy(
            xyxy=torch.as_tensor(prediction.boxes.xyxy).cpu().numpy().astype(np.float64),
            confidences=torch.as_tensor(prediction.boxes.conf).cpu().numpy().astype(np.float64),
            query_ids=torch.as_tensor(prediction.boxes.cls).cpu().numpy().astype(np.int64),
            prompt=prompt,
            image_size=image_size,
        )
