from typing import ClassVar

from ultralytics.engine.model import Model
from ultralytics.models.yolo.model import YOLOWorld

from ....options import DetectorBackend
from ....prompt import Prompt
from ....settings import DetectorSettings
from ..core import UltralyticsDetector


class YoloWorldDetector(UltralyticsDetector):
    """
    YOLO-World detector backed by Ultralytics.

    The CLIP text encoder is cached by Ultralytics, so switching vocabularies
    only re-encodes the text queries.
    """

    BACKEND: ClassVar[DetectorBackend] = DetectorBackend.YOLO_WORLD

    def __init__(self, settings: DetectorSettings) -> None:
        """
        Load the checkpoint.

        Parameters
        ----------
        settings : DetectorSettings
            YOLO-World checkpoint, batching, device and thresholds.
        """
        super().__init__(settings)
        self._model: YOLOWorld = YOLOWorld(settings.weights_path)

    @property
    def model(self) -> Model:
        return self._model

    def _apply_prompt(self, prompt: Prompt) -> None:
        self._model.set_classes(list(prompt.query_texts))
