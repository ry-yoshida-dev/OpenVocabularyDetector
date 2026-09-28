from typing import ClassVar

from transformers import Owlv2ForObjectDetection, Owlv2Model, Owlv2Processor

from ..core import OwlDetector
from ..image_fitting import ImageFitting


class Owlv2Detector(OwlDetector[Owlv2ForObjectDetection, Owlv2Processor]):
    """
    OWLv2 detector backed by ``transformers.Owlv2ForObjectDetection``.

    The processor pads every image at the bottom and right to a square before resizing it, so the model predicts
    boxes relative to that square; they are rescaled to the image, and reference boxes are mapped into it.
    """

    IMAGE_FITTING: ClassVar[ImageFitting] = ImageFitting.PAD_TO_SQUARE

    def _load_processor(self, weights_path: str) -> Owlv2Processor:
        return Owlv2Processor.from_pretrained(weights_path)

    def _load_model(self, weights_path: str) -> Owlv2ForObjectDetection:
        return Owlv2ForObjectDetection.from_pretrained(weights_path)

    def _text_encoder(self) -> Owlv2Model:
        return self._model.owlv2
