from typing import ClassVar

from ultralytics.engine.model import Model
from ultralytics.models.yolo.model import YOLOWorld

from ...backend import OVDBackend
from ...prompt import Prompt
from ...settings import OVDSettings
from .detector import UltralyticsDetector


class YoloWorldDetector(UltralyticsDetector):
    """
    YOLO-World detector backed by Ultralytics.

    The CLIP text encoder is cached by Ultralytics, so switching vocabularies
    only re-encodes the class names.
    """

    BACKEND: ClassVar[OVDBackend] = OVDBackend.YOLO_WORLD

    def __init__(self, settings: OVDSettings) -> None:
        """
        Load the checkpoint.

        Parameters
        ----------
        settings : OVDSettings
            YOLO-World checkpoint, batching, device and thresholds.
        """
        super().__init__(settings)
        self._model: YOLOWorld = YOLOWorld(settings.weights_path)

    @property
    def model(self) -> Model:
        return self._model

    def _apply_prompt(self, prompt: Prompt) -> None:
        self._model.set_classes(list(prompt.class_names))
