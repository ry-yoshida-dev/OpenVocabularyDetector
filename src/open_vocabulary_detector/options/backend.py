from enum import StrEnum

from ..prompt import PromptKind


class DetectorBackend(StrEnum):
    """
    Detector family.

    Model-specific values (weights, recommended thresholds) live in the YAML presets;
    only what code must know about a family, the prompt kinds it accepts, is defined here.

    Attributes
    ----------
    GROUNDING_DINO : str
        Grounding DINO through Hugging Face ``transformers``.
    OWL_VIT : str
        OWL-ViT (v1) through Hugging Face ``transformers``.
    YOLO_WORLD : str
        YOLO-World through Ultralytics.
    YOLOE : str
        YOLOE (text and visual queries) through Ultralytics.
    """

    GROUNDING_DINO = "grounding_dino"
    OWL_VIT = "owl_vit"
    YOLO_WORLD = "yolo_world"
    YOLOE = "yoloe"

    @property
    def supported_prompt_kinds(self) -> frozenset[PromptKind]:
        """
        Query kinds the backend accepts; OWL-ViT and YOLOE also accept both kinds mixed in one prompt.

        Returns
        -------
        frozenset[PromptKind]
            ``TEXT`` for every backend, plus ``VISUAL`` for OWL-ViT and YOLOE.
        """
        match self:
            case DetectorBackend.OWL_VIT | DetectorBackend.YOLOE:
                return frozenset({PromptKind.TEXT, PromptKind.VISUAL})
            case DetectorBackend.GROUNDING_DINO | DetectorBackend.YOLO_WORLD:
                return frozenset({PromptKind.TEXT})
