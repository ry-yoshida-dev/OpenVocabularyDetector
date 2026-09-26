from collections.abc import Sequence
from typing import ClassVar

import torch
from PIL import Image
from transformers import BatchEncoding, GroundingDinoForObjectDetection, GroundingDinoProcessor

from ...backend import OVDBackend
from ...detector import OpenVocabularyDetector
from ...prompt import Prompt
from ...result import DetectionResult, ImageSize
from ...runtime import ClassPrediction, TorchRuntime
from ...settings import OVDSettings
from .caption import GroundingCaption
from .tokenized_caption import TokenizedCaption


class GroundingDinoDetector(OpenVocabularyDetector):
    """
    Grounding DINO detector backed by Hugging Face ``transformers``.

    Class names are joined into a caption (``"cat . dog ."``). Each query box is assigned the class
    whose tokens have the highest probability, so class ids map back to the prompt without phrase matching.
    """

    BACKEND: ClassVar[OVDBackend] = OVDBackend.GROUNDING_DINO

    def __init__(self, settings: OVDSettings) -> None:
        """
        Load the processor and model.

        Parameters
        ----------
        settings : OVDSettings
            Grounding DINO checkpoint, batching, device and thresholds.
        """
        super().__init__(settings)
        self._runtime: TorchRuntime = TorchRuntime(settings)
        self._processor: GroundingDinoProcessor = GroundingDinoProcessor.from_pretrained(settings.weights_path)
        self._model: GroundingDinoForObjectDetection = GroundingDinoForObjectDetection.from_pretrained(
            settings.weights_path
        )
        self._runtime.prepare_model(self._model)

    def _tokenize(self, prompt: Prompt) -> TokenizedCaption:
        caption: GroundingCaption = GroundingCaption.from_class_names(prompt.class_names)
        encoding: BatchEncoding = self._processor.tokenizer(
            caption.text, return_offsets_mapping=True, return_tensors="pt"
        )
        tokenized_caption: TokenizedCaption = TokenizedCaption.from_encoding(caption, encoding)
        max_text_length: int = self._model.config.max_text_len
        if tokenized_caption.token_count > max_text_length:
            raise ValueError(
                f"caption has {tokenized_caption.token_count} tokens, exceeding the limit of {max_text_length}. "
                + "split the classes into several prompts."
            )
        return tokenized_caption.to(self._runtime.device)

    def _detect_mini_batch(self, images: Sequence[Image.Image], prompt: Prompt) -> list[DetectionResult]:
        tokenized_caption: TokenizedCaption = self._tokenize(prompt)
        image_inputs: BatchEncoding = self._processor.image_processor(images=list(images), return_tensors="pt")
        batch_size: int = len(images)
        with torch.inference_mode():
            outputs = self._model(
                pixel_values=self._runtime.to_model_input(image_inputs["pixel_values"]),
                pixel_mask=image_inputs["pixel_mask"].to(self._runtime.device),
                input_ids=tokenized_caption.input_ids.expand(batch_size, -1),
                attention_mask=tokenized_caption.attention_mask.expand(batch_size, -1),
                token_type_ids=tokenized_caption.token_type_ids.expand(batch_size, -1),
            )
        prediction: ClassPrediction = ClassPrediction.from_probabilities(
            tokenized_caption.class_probabilities(outputs.logits)
        )
        return self._postprocess(
            raw_detections=prediction.to_raw_detections(outputs.pred_boxes),
            class_names=prompt.class_names,
            image_sizes=[ImageSize.from_image(image) for image in images],
        )
