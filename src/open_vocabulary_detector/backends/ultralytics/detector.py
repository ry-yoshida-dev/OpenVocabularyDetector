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
from ...settings import OVDSettings


class UltralyticsDetector(OpenVocabularyDetector):
    """
    Base of detectors whose class names are baked into an Ultralytics model.

    The model holds one set of classes at a time. A prompt is applied to the model
    only when it differs from the active one, so reusing one prompt across
    many batches avoids re-configuring the model. Thresholding and NMS run
    inside Ultralytics; NMS is forced to be class-wise like every other backend
    (YOLOE would otherwise switch to class-agnostic NMS).
    """

    DISABLED_NMS_IOU_THRESHOLD: float = 1.0

    def __init__(self, settings: OVDSettings) -> None:
        """
        Parameters
        ----------
        settings : OVDSettings
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
        Configure the model to detect the prompt classes.

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
                half=self.settings.is_half_precision_enabled,
                agnostic_nms=False,
                verbose=False,
            )
            if isinstance(prediction, Results)
        ]
        return [
            self._build_result(prediction, prompt.class_names, ImageSize.from_image(image))
            for prediction, image in zip(predictions, images, strict=True)
        ]

    @staticmethod
    def _build_result(
        prediction: Results,
        class_names: tuple[str, ...],
        image_size: ImageSize,
    ) -> DetectionResult:
        if prediction.boxes is None:
            return DetectionResult.empty(class_names=class_names, image_size=image_size)
        return DetectionResult.from_xyxy(
            xyxy=torch.as_tensor(prediction.boxes.xyxy).cpu().numpy().astype(np.float64),
            confidences=torch.as_tensor(prediction.boxes.conf).cpu().numpy().astype(np.float64),
            class_ids=torch.as_tensor(prediction.boxes.cls).cpu().numpy().astype(np.int64),
            class_names=class_names,
            image_size=image_size,
        ).sort_by_confidence()
