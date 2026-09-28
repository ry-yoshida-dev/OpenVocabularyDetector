from ..core import GroundingDetector
from ..separate_box_heads import MMGroundingDinoWithSeparateBoxHeads


class MMGroundingDinoDetector(GroundingDetector[MMGroundingDinoWithSeparateBoxHeads]):
    """
    MM-Grounding-DINO detector backed by ``transformers.MMGroundingDinoForObjectDetection``.

    LLMDet checkpoints share the architecture and load through this detector as well. The model is loaded as
    ``MMGroundingDinoWithSeparateBoxHeads`` so that the per-layer box heads of the checkpoints survive loading.
    """

    def _load_model(self, weights_path: str) -> MMGroundingDinoWithSeparateBoxHeads:
        return MMGroundingDinoWithSeparateBoxHeads.from_pretrained(weights_path)
