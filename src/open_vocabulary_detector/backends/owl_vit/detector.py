from collections.abc import Sequence
from typing import ClassVar, cast

import torch
from PIL import Image
from transformers import BatchEncoding, OwlViTForObjectDetection, OwlViTProcessor
from transformers.modeling_outputs import BaseModelOutputWithPooling

from ...cache import QueryEmbeddingStore
from ...detector import OpenVocabularyDetector
from ...options import DetectorBackend
from ...prompt import Prompt, VisualReference
from ...result import DetectionResult, ImageSize
from ...runtime import QueryPrediction, TorchRuntime
from ...settings import DetectorSettings
from .box_query_selector import OwlViTBoxQuerySelector


class OwlViTDetector(OpenVocabularyDetector):
    """
    OWL-ViT detector backed by Hugging Face ``transformers``.

    Text queries are embedded by the text encoder; visual queries by the patch embeddings that represent their
    reference boxes (image-guided detection), averaged per query. Both kinds may be mixed in one prompt, even
    within one class. Text embeddings are cached per phrase and reference embeddings per reference.
    """

    BACKEND: ClassVar[DetectorBackend] = DetectorBackend.OWL_VIT

    def __init__(self, settings: DetectorSettings) -> None:
        """
        Load the processor and model.

        Parameters
        ----------
        settings : DetectorSettings
            OWL-ViT checkpoint, batching, device and thresholds.
        """
        super().__init__(settings)
        self._runtime: TorchRuntime = TorchRuntime(settings)
        self._processor: OwlViTProcessor = OwlViTProcessor.from_pretrained(settings.weights_path)
        self._model: OwlViTForObjectDetection = OwlViTForObjectDetection.from_pretrained(settings.weights_path)
        self._runtime.prepare_model(self._model)
        self._text_token_limit: int = self._model.owlvit.text_model.embeddings.position_embedding.num_embeddings
        self._query_embeddings: QueryEmbeddingStore = QueryEmbeddingStore(
            embed_texts=self._embed_texts, embed_references=self._embed_references
        )

    def _detect_mini_batch(self, images: Sequence[Image.Image], prompt: Prompt) -> list[DetectionResult]:
        feature_map, patch_features = self._extract_patch_features(images)
        query_embeddings: torch.Tensor = self._query_embeddings.embed(prompt).to(patch_features)
        with torch.inference_mode():
            normalized_cxcywh: torch.Tensor = self._model.box_predictor(
                self._typed_as_float_tensor(patch_features), self._typed_as_float_tensor(feature_map)
            )
            class_outputs = self._model.class_predictor(
                self._typed_as_float_tensor(patch_features),
                self._typed_as_float_tensor(query_embeddings.unsqueeze(0).expand(len(images), -1, -1)),
                None,
            )
        logits: torch.Tensor = class_outputs[0]
        prediction: QueryPrediction = QueryPrediction.from_probabilities(logits.float().sigmoid())
        return self._postprocess(
            raw_detections=prediction.to_raw_detections(normalized_cxcywh),
            prompt=prompt,
            image_sizes=[ImageSize.from_image(image) for image in images],
        )

    def _embed_texts(self, text_queries: Sequence[str]) -> list[torch.Tensor]:
        encoding: BatchEncoding = self._processor.tokenizer(
            list(text_queries),
            padding="max_length",
            max_length=self._text_token_limit,
            return_tensors="pt",
        )
        input_ids: torch.Tensor = encoding["input_ids"]
        if input_ids.shape[1] > self._text_token_limit:
            raise ValueError(
                f"text queries must fit in {self._text_token_limit} tokens. got {input_ids.shape[1]} tokens"
            )
        with torch.inference_mode():
            text_outputs: object = cast(
                object,
                self._model.owlvit.get_text_features(
                    input_ids=input_ids.to(self._runtime.device),
                    attention_mask=encoding["attention_mask"].to(self._runtime.device),
                ),
            )
        if not isinstance(text_outputs, BaseModelOutputWithPooling) or text_outputs.pooler_output is None:
            raise RuntimeError(f"unexpected text encoder output: {type(text_outputs)}")
        return list(text_outputs.pooler_output.float())

    def _embed_references(self, references: Sequence[VisualReference]) -> list[torch.Tensor]:
        return [self._embed_reference(reference) for reference in references]

    def _embed_reference(self, reference: VisualReference) -> torch.Tensor:
        feature_map, patch_features = self._extract_patch_features([self._to_rgb(reference.image)])
        with torch.inference_mode():
            class_outputs: tuple[torch.Tensor, torch.Tensor] = cast(
                tuple[torch.Tensor, torch.Tensor],
                cast(object, self._model.class_predictor(self._typed_as_float_tensor(patch_features))),
            )
            predicted_cxcywh: torch.Tensor = self._model.box_predictor(
                self._typed_as_float_tensor(patch_features), self._typed_as_float_tensor(feature_map)
            )
        return OwlViTBoxQuerySelector.select(
            target_xyxy=torch.from_numpy(reference.normalized_xyxy),
            predicted_cxcywh=predicted_cxcywh[0],
            class_embeddings=class_outputs[1][0],
        )

    def _extract_patch_features(self, images: Sequence[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        image_inputs: BatchEncoding = self._processor.image_processor(images=list(images), return_tensors="pt")
        pixel_values: torch.Tensor = self._runtime.to_model_input(image_inputs["pixel_values"])
        with torch.inference_mode():
            feature_map: torch.Tensor = self._model.image_embedder(
                pixel_values=self._typed_as_float_tensor(pixel_values)
            )[0]
        batch_size, patch_rows, patch_columns, hidden_size = feature_map.shape
        patch_features: torch.Tensor = feature_map.reshape(batch_size, patch_rows * patch_columns, hidden_size)
        return feature_map, patch_features

    @staticmethod
    def _typed_as_float_tensor(tensor: torch.Tensor) -> torch.FloatTensor:
        return cast(torch.FloatTensor, tensor)
