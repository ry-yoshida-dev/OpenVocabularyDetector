from dataclasses import dataclass
from typing import TYPE_CHECKING

from .options import DetectionThresholds, DetectorBackend, Device

if TYPE_CHECKING:
    from .detector import OpenVocabularyDetector


@dataclass(frozen=True, kw_only=True)
class DetectorSettings:
    """
    Everything needed to load and run a detector, identical for every backend.

    The fields map one-to-one onto the ``detector`` section of the YAML presets in this package,
    so a preset can be turned into settings by any dataclass builder (enums are written by value).

    Attributes
    ----------
    backend : DetectorBackend
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

    backend: DetectorBackend
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

        Only the sub-package of ``backend`` is imported, so Ultralytics is needed only for ``YOLO_WORLD`` and
        ``YOLOE``, and ``timm`` only for ``OMDET_TURBO``. For ``GROUNDING_DINO`` and ``OWL_VIT``, the ``model_type``
        of the checkpoint configuration selects the architecture (Grounding DINO or MM-Grounding-DINO, OWL-ViT or
        OWLv2).

        Returns
        -------
        OpenVocabularyDetector
            Loaded detector.

        Raises
        ------
        ValueError
            If the checkpoint of ``GROUNDING_DINO`` or ``OWL_VIT`` has an architecture of another backend.
        ModuleNotFoundError
            If ``backend`` is ``YOLO_WORLD`` or ``YOLOE`` and Ultralytics is not installed.
        ImportError
            If ``backend`` is ``OMDET_TURBO`` and ``timm`` is not installed.
        """
        match self.backend:
            case DetectorBackend.GROUNDING_DINO:
                from .backends.grounding_dino import GroundingDinoVariant

                return GroundingDinoVariant.of_checkpoint(self.weights_path).build(self)
            case DetectorBackend.OWL_VIT:
                from .backends.owl_vit import OwlVersion

                return OwlVersion.of_checkpoint(self.weights_path).build(self)
            case DetectorBackend.OMDET_TURBO:
                from .backends.omdet_turbo import OmDetTurboDetector

                return OmDetTurboDetector(self)
            case DetectorBackend.FLORENCE2:
                from .backends.florence2 import Florence2Detector

                return Florence2Detector(self)
            case DetectorBackend.YOLO_WORLD:
                from .backends.ultralytics import YoloWorldDetector

                return YoloWorldDetector(self)
            case DetectorBackend.YOLOE:
                from .backends.ultralytics import YoloEDetector

                return YoloEDetector(self)
