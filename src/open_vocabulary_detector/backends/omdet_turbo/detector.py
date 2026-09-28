from collections.abc import Sequence
from typing import ClassVar

import torch
from PIL import Image
from transformers import BatchEncoding, OmDetTurboForObjectDetection, OmDetTurboProcessor

from ...detector import OpenVocabularyDetector
from ...options import DetectorBackend
from ...prompt import Prompt
from ...result import DetectionResult, ImageSize
from ...runtime import QueryPrediction, TorchRuntime
from ...settings import DetectorSettings
from .prompt_encoding import PromptEncoding


class OmDetTurboDetector(OpenVocabularyDetector):
    """
    OmDet-Turbo detector backed by ``transformers.OmDetTurboForObjectDetection``.

    Every text query is encoded as one class, and the prompt as the task sentence ``"Detect car, suv, person."``.
    Both are tokenized once per prompt and reused while the same prompt is detected; the model itself caches the
    text embeddings of recently seen classes and tasks. Each predicted box takes its most probable query.
    Queries and the task are padded to ``TEXT_TOKEN_LIMIT`` CLIP tokens, like the ``transformers`` processor; a
    longer query is rejected, while a longer task is truncated.
    """

    BACKEND: ClassVar[DetectorBackend] = DetectorBackend.OMDET_TURBO
    TEXT_TOKEN_LIMIT: ClassVar[int] = 77

    def __init__(self, settings: DetectorSettings) -> None:
        """
        Load the processor and model.

        Parameters
        ----------
        settings : DetectorSettings
            OmDet-Turbo checkpoint, batching, device and thresholds.

        Raises
        ------
        ImportError
            If ``timm``, which provides the vision backbone, is not installed.
        """
        super().__init__(settings)
        self._runtime: TorchRuntime = TorchRuntime(settings)
        self._processor: OmDetTurboProcessor = OmDetTurboProcessor.from_pretrained(settings.weights_path)
        self._model: OmDetTurboForObjectDetection = OmDetTurboForObjectDetection.from_pretrained(settings.weights_path)
        self._runtime.prepare_model(self._model)
        self._active_encoding: PromptEncoding | None = None

    def _encoding_of(self, prompt: Prompt) -> PromptEncoding:
        if self._active_encoding is None or not self._active_encoding.is_for(prompt):
            self._active_encoding = self._encode(prompt)
        return self._active_encoding

    def _encode(self, prompt: Prompt) -> PromptEncoding:
        query_texts: tuple[str, ...] = prompt.query_texts
        class_encoding: BatchEncoding = self._processor.tokenizer(
            list(query_texts), padding="max_length", max_length=self.TEXT_TOKEN_LIMIT, return_tensors="pt"
        )
        if class_encoding["input_ids"].shape[1] > self.TEXT_TOKEN_LIMIT:
            raise ValueError(
                f"text queries must fit in {self.TEXT_TOKEN_LIMIT} tokens. "
                + f"got {class_encoding['input_ids'].shape[1]} tokens"
            )
        task_encoding: BatchEncoding = self._processor.tokenizer(
            [f"Detect {', '.join(query_texts)}."],
            padding="max_length",
            truncation=True,
            max_length=self.TEXT_TOKEN_LIMIT,
            return_tensors="pt",
        )
        return PromptEncoding(
            prompt=prompt,
            class_input_ids=class_encoding["input_ids"],
            class_attention_mask=class_encoding["attention_mask"],
            task_input_ids=task_encoding["input_ids"],
            task_attention_mask=task_encoding["attention_mask"],
        ).to(self._runtime.device)

    def _detect_mini_batch(self, images: Sequence[Image.Image], prompt: Prompt) -> list[DetectionResult]:
        encoding: PromptEncoding = self._encoding_of(prompt)
        image_inputs: BatchEncoding = self._processor.image_processor(images=list(images), return_tensors="pt")
        batch_size: int = len(images)
        with torch.inference_mode():
            outputs = self._model(
                pixel_values=self._runtime.to_model_input(image_inputs["pixel_values"]),
                classes_input_ids=encoding.class_input_ids.repeat(batch_size, 1),
                classes_attention_mask=encoding.class_attention_mask.repeat(batch_size, 1),
                tasks_input_ids=encoding.task_input_ids.expand(batch_size, -1),
                tasks_attention_mask=encoding.task_attention_mask.expand(batch_size, -1),
                classes_structure=torch.full(
                    (batch_size,), encoding.query_count, dtype=torch.long, device=self._runtime.device
                ),
            )
        prediction: QueryPrediction = QueryPrediction.from_probabilities(
            outputs.decoder_class_logits[..., : encoding.query_count].float().sigmoid()
        )
        return self._postprocess(
            raw_detections=prediction.to_raw_detections(outputs.decoder_coord_logits),
            prompt=prompt,
            image_sizes=[ImageSize.from_image(image) for image in images],
        )
