from collections.abc import Sequence
from typing import ClassVar, cast

import numpy as np
import torch
from ultralytics.engine.model import Model
from ultralytics.models.yolo.model import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPDetectPredictor
from ultralytics.nn.tasks import YOLOEModel

from ...backend import OVDBackend
from ...cache import QueryEmbeddingStore
from ...prompt import Prompt, VisualReference
from ...settings import OVDSettings
from .detection_head import DetectionHead
from .detector import UltralyticsDetector


class YoloEDetector(UltralyticsDetector):
    """
    YOLOE detector backed by Ultralytics, with text and visual queries.

    Text queries are embedded by the text prompt encoder; visual queries by the visual prompt embeddings of
    their reference boxes, averaged per query. Both kinds may be mixed in one prompt, even within one class.
    Text embeddings are cached per phrase and reference embeddings per reference; the assembled embeddings stay in the model while the same
    prompt is reused.
    """

    BACKEND: ClassVar[OVDBackend] = OVDBackend.YOLOE

    def __init__(self, settings: OVDSettings) -> None:
        """
        Load the checkpoint.

        Parameters
        ----------
        settings : OVDSettings
            YOLOE checkpoint, batching, device and thresholds.
        """
        super().__init__(settings)
        self._model: YOLOE = YOLOE(settings.weights_path)
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
        network: YOLOEModel = cast(YOLOEModel, self._model.model)
        head: DetectionHead = cast(DetectionHead, cast(torch.nn.Sequential, network.model)[-1])
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
        active_class_count: int = head.nc
        head.nc = 1
        try:
            predictor.set_prompts({"bboxes": reference.xyxy, "cls": np.zeros(len(reference.boxes), dtype=np.int64)})
            predictor.setup_model(model=network, verbose=False)
            visual_embeddings: torch.Tensor = cast(torch.Tensor, predictor.get_vpe(self._to_rgb(reference.image)))
        finally:
            head.nc = active_class_count
        return visual_embeddings.reshape(1, -1)
