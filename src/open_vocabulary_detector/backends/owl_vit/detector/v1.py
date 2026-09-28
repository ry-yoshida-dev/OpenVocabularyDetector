from typing import ClassVar

from transformers import OwlViTForObjectDetection, OwlViTModel, OwlViTProcessor

from ..core import OwlDetector
from ..image_fitting import ImageFitting


class OwlViTDetector(OwlDetector[OwlViTForObjectDetection, OwlViTProcessor]):
    """
    OWL-ViT (v1) detector backed by ``transformers.OwlViTForObjectDetection``.

    Images are resized to the square model input without padding, so boxes relative to the input are relative to
    the image as well.
    """

    IMAGE_FITTING: ClassVar[ImageFitting] = ImageFitting.STRETCH

    def _load_processor(self, weights_path: str) -> OwlViTProcessor:
        return OwlViTProcessor.from_pretrained(weights_path)

    def _load_model(self, weights_path: str) -> OwlViTForObjectDetection:
        return OwlViTForObjectDetection.from_pretrained(weights_path)

    def _text_encoder(self) -> OwlViTModel:
        return self._model.owlvit
