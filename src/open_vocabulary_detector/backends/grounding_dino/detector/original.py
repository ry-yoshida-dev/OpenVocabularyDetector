from transformers import GroundingDinoForObjectDetection

from ..core import GroundingDetector


class GroundingDinoDetector(GroundingDetector[GroundingDinoForObjectDetection]):
    """
    Grounding DINO detector backed by ``transformers.GroundingDinoForObjectDetection``.
    """

    def _load_model(self, weights_path: str) -> GroundingDinoForObjectDetection:
        return GroundingDinoForObjectDetection.from_pretrained(weights_path)
