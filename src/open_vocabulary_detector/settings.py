from dataclasses import dataclass
from typing import TYPE_CHECKING

from .backend import OVDBackend
from .options import DetectionThresholds, Device

if TYPE_CHECKING:
    from .detector import OpenVocabularyDetector


@dataclass(frozen=True, kw_only=True)
class OVDSettings:
    """
    Everything needed to load and run a detector, identical for every backend.

    The fields map one-to-one onto the ``detector`` section of the YAML presets in this package,
    so a preset can be turned into settings by any dataclass builder (enums are written by value).

    Attributes
    ----------
    backend : OVDBackend
        Detector family able to load ``weights_path``.
    weights_path : str
        Hugging Face Hub model id, Ultralytics asset name or local weights path.
    thresholds : DetectionThresholds
        Post-processing thresholds.
    batch_size : int
        Number of images per forward pass.
    device : Device
        Device to run on.
    is_half_precision_enabled : bool
        Whether to run the model in float16; GPU only.

    Raises
    ------
    ValueError
        If ``weights_path`` is blank or ``batch_size`` is not positive.
    """

    backend: OVDBackend
    weights_path: str
    thresholds: DetectionThresholds
    batch_size: int = 8
    device: Device = Device.AUTO
    is_half_precision_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.weights_path.strip():
            raise ValueError("weights_path must not be blank.")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive. got {self.batch_size}")

    def build(self) -> "OpenVocabularyDetector":
        """
        Load the detector of ``backend`` with these settings.

        Ultralytics is imported only for ``YOLO_WORLD`` and ``YOLOE``, so the other backends work without the
        ``ultralytics`` extra installed.

        Returns
        -------
        OpenVocabularyDetector
            Loaded detector.

        Raises
        ------
        ModuleNotFoundError
            If ``backend`` is ``YOLO_WORLD`` or ``YOLOE`` and Ultralytics is not installed.
        """
        match self.backend:
            case OVDBackend.GROUNDING_DINO:
                from .backends.grounding_dino import GroundingDinoDetector

                return GroundingDinoDetector(self)
            case OVDBackend.OWL_VIT:
                from .backends.owl_vit import OwlViTDetector

                return OwlViTDetector(self)
            case OVDBackend.YOLO_WORLD:
                from .backends.ultralytics import YoloWorldDetector

                return YoloWorldDetector(self)
            case OVDBackend.YOLOE:
                from .backends.ultralytics import YoloEDetector

                return YoloEDetector(self)
