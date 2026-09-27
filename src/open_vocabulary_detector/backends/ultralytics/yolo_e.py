from collections.abc import Sequence
from typing import ClassVar, cast

import numpy as np
import torch
from ultralytics.engine.model import Model
from ultralytics.models.yolo.model import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPDetectPredictor

from ...cache import QueryEmbeddingStore
from ...options import DetectorBackend
from ...prompt import Prompt, VisualReference
from ...settings import DetectorSettings
from .detector import UltralyticsDetector


class YoloEDetector(UltralyticsDetector):
    """
    YOLOE detector backed by Ultralytics, with text and visual queries.

    Text queries are embedded by the text prompt encoder; visual queries by the visual prompt embedding of every
    reference box, averaged per query. Both kinds may be mixed in one prompt, even within one class.
    Text embeddings are cached per phrase and reference embeddings per reference; the assembled embeddings stay in
    the model while the same prompt is reused. Visual prompt embeddings are computed by a predictor holding its own
    copy of the network, built on the first visual query and reused afterwards.
    """

    BACKEND: ClassVar[DetectorBackend] = DetectorBackend.YOLOE

    def __init__(self, settings: DetectorSettings) -> None:
        """
        Load the checkpoint.

        Parameters
        ----------
        settings : DetectorSettings
            YOLOE checkpoint, batching, device and thresholds.
        """
        super().__init__(settings)
        self._model: YOLOE = YOLOE(settings.weights_path)
        self._visual_prompt_predictor: YOLOEVPDetectPredictor | None = None
        self._query_embeddings: QueryEmbeddingStore = QueryEmbeddingStore(
            embed_texts=self._embed_texts, embed_references=self._embed_references
        )

    @property
    def model(self) -> Model:
        return self._model

    def _apply_prompt(self, prompt: Prompt) -> None:
        embeddings: torch.Tensor = self._query_embeddings.embed(prompt)
        self._model.set_classes(list(prompt.query_names), embeddings.unsqueeze(0))

    def _embed_texts(self, text_queries: Sequence[str]) -> list[torch.Tensor]:
        text_embeddings: torch.Tensor | None = cast(torch.Tensor | None, self._model.get_text_pe(list(text_queries)))
        if text_embeddings is None:
            raise TypeError(f"YOLOE returned no text embeddings for {list(text_queries)}")
        return list(text_embeddings.reshape(len(text_queries), -1).float())

    def _embed_references(self, references: Sequence[VisualReference]) -> list[torch.Tensor]:
        return [self._embed_reference(reference) for reference in references]

    def _embed_reference(self, reference: VisualReference) -> torch.Tensor:
        predictor: YOLOEVPDetectPredictor = self._get_visual_prompt_predictor()
        box_count: int = len(reference.boxes)
        predictor.set_prompts({"bboxes": reference.xyxy, "cls": np.arange(box_count, dtype=np.int64)})
        visual_embeddings: torch.Tensor = cast(torch.Tensor, predictor.get_vpe(self._to_rgb(reference.image)))
        return visual_embeddings.reshape(box_count, -1).float()

    def _get_visual_prompt_predictor(self) -> YOLOEVPDetectPredictor:
        if self._visual_prompt_predictor is None:
            predictor: YOLOEVPDetectPredictor = YOLOEVPDetectPredictor(
                overrides={
                    "task": self._model.task,
                    "mode": "predict",
                    "save": False,
                    "verbose": False,
                    "batch": 1,
                    "device": self._runtime.device.type,
                }
            )
            predictor.setup_model(model=self._model.model, verbose=False)
            self._visual_prompt_predictor = predictor
        return self._visual_prompt_predictor
