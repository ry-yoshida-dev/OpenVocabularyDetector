from collections.abc import Sequence
from typing import ClassVar, cast

import numpy as np
import torch
from PIL import Image
from transformers import BatchFeature, Florence2ForConditionalGeneration, Florence2Processor

from ...array_types import FloatArray, IntArray
from ...detector import OpenVocabularyDetector
from ...options import DetectorBackend
from ...prompt import Prompt
from ...result import DetectionResult, ImageSize
from ...runtime import TorchRuntime
from ...settings import DetectorSettings
from .location_parser import LocationParser
from .token_generator import TokenGenerator


class Florence2Detector(OpenVocabularyDetector):
    """
    Florence-2 detector backed by ``transformers.Florence2ForConditionalGeneration``.

    Florence-2 is generative: for the ``<OPEN_VOCABULARY_DETECTION>`` task it writes the boxes of one phrase as
    location tokens. Every text query is therefore generated separately over the mini-batch, and its boxes are
    assigned to that query, so no phrase matching is needed. The model gives no score, so every box has the
    confidence ``DETECTION_CONFIDENCE``; the confidence threshold keeps every box and NMS keeps the first of equally
    confident overlapping boxes, in query order.
    """

    BACKEND: ClassVar[DetectorBackend] = DetectorBackend.FLORENCE2
    TASK_TOKEN: ClassVar[str] = "<OPEN_VOCABULARY_DETECTION>"
    DETECTION_CONFIDENCE: ClassVar[float] = 1.0
    MAX_NEW_TOKENS: ClassVar[int] = 1024
    BEAM_COUNT: ClassVar[int] = 3

    def __init__(self, settings: DetectorSettings) -> None:
        """
        Load the processor and model.

        Parameters
        ----------
        settings : DetectorSettings
            Florence-2 checkpoint, batching, device and thresholds.
        """
        super().__init__(settings)
        self._runtime: TorchRuntime = TorchRuntime(settings)
        self._processor: Florence2Processor = Florence2Processor.from_pretrained(settings.weights_path)
        self._model: Florence2ForConditionalGeneration = Florence2ForConditionalGeneration.from_pretrained(
            settings.weights_path
        )
        self._runtime.prepare_model(self._model)

    def _detect_mini_batch(self, images: Sequence[Image.Image], prompt: Prompt) -> list[DetectionResult]:
        image_boxes: list[list[FloatArray]] = [[] for _ in images]
        image_query_ids: list[list[IntArray]] = [[] for _ in images]
        for query_id, query_text in enumerate(prompt.query_texts):
            for image_index, normalized_xyxy in enumerate(self._generate_boxes(images, query_text)):
                image_boxes[image_index].append(normalized_xyxy)
                image_query_ids[image_index].append(np.full(len(normalized_xyxy), query_id, dtype=np.int64))
        return [
            self._build_result(
                normalized_xyxy=np.concatenate(boxes),
                query_ids=np.concatenate(query_ids),
                prompt=prompt,
                image_size=ImageSize.from_image(image),
            )
            for boxes, query_ids, image in zip(image_boxes, image_query_ids, images, strict=True)
        ]

    def _generate_boxes(self, images: Sequence[Image.Image], query_text: str) -> list[FloatArray]:
        inputs: BatchFeature = self._processor(
            text=[self.TASK_TOKEN + query_text] * len(images),
            images=list(images),
            text_kwargs={"return_tensors": "pt"},
            images_kwargs={"return_tensors": "pt"},
        )
        with torch.inference_mode():
            generated_ids: torch.Tensor = cast(TokenGenerator, self._model).generate(
                input_ids=inputs["input_ids"].to(self._runtime.device),
                attention_mask=inputs["attention_mask"].to(self._runtime.device),
                pixel_values=self._runtime.to_model_input(inputs["pixel_values"]),
                max_new_tokens=self.MAX_NEW_TOKENS,
                num_beams=self.BEAM_COUNT,
                do_sample=False,
            )
        generated_texts: list[str] = self._processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        return [LocationParser.parse(generated_text) for generated_text in generated_texts]

    def _build_result(
        self,
        normalized_xyxy: FloatArray,
        query_ids: IntArray,
        prompt: Prompt,
        image_size: ImageSize,
    ) -> DetectionResult:
        scale: FloatArray = np.array(
            [image_size.width, image_size.height, image_size.width, image_size.height], dtype=np.float64
        )
        result: DetectionResult = DetectionResult.from_xyxy(
            xyxy=normalized_xyxy * scale,
            confidences=np.full(len(query_ids), self.DETECTION_CONFIDENCE, dtype=np.float64),
            query_ids=query_ids,
            prompt=prompt,
            image_size=image_size,
        )
        return self._suppress_and_sort(result.filter_by_confidence(self.thresholds.confidence_threshold))
