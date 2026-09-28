from enum import StrEnum

from ...detector import OpenVocabularyDetector
from ...settings import DetectorSettings
from ..checkpoint_config import CheckpointConfig
from .detector import Owlv2Detector, OwlViTDetector


class OwlVersion(StrEnum):
    """
    OWL model version, identified by the ``model_type`` of a checkpoint.

    Attributes
    ----------
    V1 : str
        OWL-ViT.
    V2 : str
        OWLv2.
    """

    V1 = "owlvit"
    V2 = "owlv2"

    @classmethod
    def of_checkpoint(cls, weights_path: str) -> "OwlVersion":
        """
        Read the version of a checkpoint from its configuration.

        Parameters
        ----------
        weights_path : str
            Hugging Face Hub model id or local checkpoint directory.

        Raises
        ------
        ValueError
            If the checkpoint is not an OWL-ViT or OWLv2 model.

        Returns
        -------
        OwlVersion
            Version of the checkpoint.
        """
        model_type: str = CheckpointConfig.read_model_type(weights_path)
        if model_type not in {version.value for version in cls}:
            raise ValueError(
                f"{weights_path} is a {model_type!r} model, not one of {sorted(version.value for version in cls)}."
            )
        return cls(model_type)

    def build(self, settings: DetectorSettings) -> OpenVocabularyDetector:
        """
        Load the detector of this version.

        Parameters
        ----------
        settings : DetectorSettings
            Checkpoint of this version, batching, device and thresholds.

        Returns
        -------
        OpenVocabularyDetector
            Loaded ``OwlViTDetector`` or ``Owlv2Detector``.
        """
        match self:
            case OwlVersion.V1:
                return OwlViTDetector(settings)
            case OwlVersion.V2:
                return Owlv2Detector(settings)
