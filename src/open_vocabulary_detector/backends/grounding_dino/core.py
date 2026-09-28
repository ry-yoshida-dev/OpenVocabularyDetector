from abc import abstractmethod
from collections.abc import Sequence
from typing import ClassVar

import torch
from PIL import Image
from transformers import (
    BatchEncoding,
    GroundingDinoForObjectDetection,
    GroundingDinoProcessor,
    MMGroundingDinoForObjectDetection,
)

from ...detector import OpenVocabularyDetector
from ...options import DetectorBackend
from ...prompt import Prompt
from ...result import DetectionResult, ImageSize
from ...runtime import QueryPrediction, TorchRuntime
from ...settings import DetectorSettings
from .active_caption import ActiveCaption
from .caption import GroundingCaption
from .tokenized_caption import TokenizedCaption


class GroundingDetector[ModelT: GroundingDinoForObjectDetection | MMGroundingDinoForObjectDetection](
    OpenVocabularyDetector
):
    """
    Base of Grounding DINO family detectors backed by Hugging Face ``transformers``.

    Text queries are joined into a caption (``"car . suv . taxi . dog ."``). Each predicted box is assigned the
    query whose tokens have the highest probability, so queries and their classes map back to the prompt without
    phrase matching. The caption is tokenized once per prompt and reused while the same prompt is detected.
    Subclasses only choose the model architecture; the processor, caption handling and the ``GROUNDING_DINO``
    backend are shared.
    """

    BACKEND: ClassVar[DetectorBackend] = DetectorBackend.GROUNDING_DINO

    def __init__(self, settings: DetectorSettings) -> None:
        """
        Load the processor and model.

        Parameters
        ----------
        settings : DetectorSettings
            Checkpoint, batching, device and thresholds.
        """
        super().__init__(settings)
        self._runtime: TorchRuntime = TorchRuntime(settings)
        self._processor: GroundingDinoProcessor = GroundingDinoProcessor.from_pretrained(settings.weights_path)
        self._model: ModelT = self._load_model(settings.weights_path)
        self._runtime.prepare_model(self._model)
        self._active_caption: ActiveCaption | None = None

    @abstractmethod
    def _load_model(self, weights_path: str) -> ModelT:
        """
        Load the detection model of the checkpoint.

        Parameters
        ----------
        weights_path : str
            Hugging Face Hub model id or local checkpoint directory.

        Returns
        -------
        ModelT
            Loaded model.
        """

    def _caption_of(self, prompt: Prompt) -> TokenizedCaption:
        if self._active_caption is None or not self._active_caption.is_for(prompt):
            self._active_caption = ActiveCaption(prompt=prompt, tokenized_caption=self._tokenize(prompt))
        return self._active_caption.tokenized_caption

    def _tokenize(self, prompt: Prompt) -> TokenizedCaption:
        caption: GroundingCaption = GroundingCaption.from_queries(prompt.query_texts)
        encoding: BatchEncoding = self._processor.tokenizer(
            caption.text, return_offsets_mapping=True, return_tensors="pt"
        )
        tokenized_caption: TokenizedCaption = TokenizedCaption.from_encoding(caption, encoding)
        max_text_length: int = self._model.config.max_text_len
        if tokenized_caption.token_count > max_text_length:
            raise ValueError(
                f"caption has {tokenized_caption.token_count} tokens, exceeding the limit of {max_text_length}. "
                + "split the queries into several prompts."
            )
        return tokenized_caption.to(self._runtime.device)

    def _detect_mini_batch(self, images: Sequence[Image.Image], prompt: Prompt) -> list[DetectionResult]:
        tokenized_caption: TokenizedCaption = self._caption_of(prompt)
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
        prediction: QueryPrediction = QueryPrediction.from_probabilities(
            tokenized_caption.query_probabilities(outputs.logits)
        )
        return self._postprocess(
            raw_detections=prediction.to_raw_detections(outputs.pred_boxes),
            prompt=prompt,
            image_sizes=[ImageSize.from_image(image) for image in images],
        )
