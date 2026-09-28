from enum import StrEnum

from ...detector import OpenVocabularyDetector
from ...settings import DetectorSettings
from ..checkpoint_config import CheckpointConfig
from .detector import GroundingDinoDetector, MMGroundingDinoDetector


class GroundingDinoVariant(StrEnum):
    """
    Grounding DINO architecture variant, identified by the ``model_type`` of a checkpoint.

    Attributes
    ----------
    ORIGINAL : str
        Grounding DINO.
    MM : str
        MM-Grounding-DINO, also used by LLMDet.
    """

    ORIGINAL = "grounding-dino"
    MM = "mm-grounding-dino"

    @classmethod
    def of_checkpoint(cls, weights_path: str) -> "GroundingDinoVariant":
        """
        Read the variant of a checkpoint from its configuration.

        Parameters
        ----------
        weights_path : str
            Hugging Face Hub model id or local checkpoint directory.

        Raises
        ------
        ValueError
            If the checkpoint is not a Grounding DINO or MM-Grounding-DINO model.

        Returns
        -------
        GroundingDinoVariant
            Variant of the checkpoint.
        """
        model_type: str = CheckpointConfig.read_model_type(weights_path)
        if model_type not in {variant.value for variant in cls}:
            raise ValueError(
                f"{weights_path} is a {model_type!r} model, not one of {sorted(variant.value for variant in cls)}."
            )
        return cls(model_type)

    def build(self, settings: DetectorSettings) -> OpenVocabularyDetector:
        """
        Load the detector of this variant.

        Parameters
        ----------
        settings : DetectorSettings
            Checkpoint of this variant, batching, device and thresholds.

        Returns
        -------
        OpenVocabularyDetector
            Loaded ``GroundingDinoDetector`` or ``MMGroundingDinoDetector``.
        """
        match self:
            case GroundingDinoVariant.ORIGINAL:
                return GroundingDinoDetector(settings)
            case GroundingDinoVariant.MM:
                return MMGroundingDinoDetector(settings)
