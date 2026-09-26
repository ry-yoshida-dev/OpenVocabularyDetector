from collections.abc import Sequence
from typing import ClassVar, cast

import numpy as np
import torch
from ultralytics.engine.model import Model
from ultralytics.models.yolo.model import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPDetectPredictor
from ultralytics.nn.tasks import YOLOEModel

from ...array_types import IntArray
from ...backend import OVDBackend
from ...cache import QueryEmbeddingStore, ReferenceEmbeddings
from ...prompt import Prompt, VisualReference
from ...settings import OVDSettings
from .detection_head import DetectionHead
from .detector import UltralyticsDetector


class YoloEDetector(UltralyticsDetector):
    """
    YOLOE detector backed by Ultralytics, with text and visual prompts.

    Classes are queried by their text embeddings, or, in a text-visual prompt, by the visual prompt
    embeddings of their reference boxes, averaged per class. Text embeddings are cached per class name
    and reference embeddings per reference; the assembled embeddings stay in the model while the same
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
            embed_class_names=self._embed_class_names, embed_references=self._embed_references
        )

    @property
    def model(self) -> Model:
        return self._model

    def _apply_prompt(self, prompt: Prompt) -> None:
        embeddings: torch.Tensor = self._query_embeddings.embed(prompt)
        self._model.set_classes(list(prompt.class_names), embeddings.unsqueeze(0))

    def _embed_class_names(self, class_names: Sequence[str]) -> list[torch.Tensor]:
        text_embeddings: torch.Tensor | None = cast(torch.Tensor | None, self._model.get_text_pe(list(class_names)))
        if text_embeddings is None:
            raise TypeError(f"YOLOE returned no text embeddings for {list(class_names)}")
        return list(text_embeddings.reshape(len(class_names), -1).float())

    def _embed_references(self, references: Sequence[VisualReference]) -> list[ReferenceEmbeddings]:
        return [self._embed_reference(reference) for reference in references]

    def _embed_reference(self, reference: VisualReference) -> ReferenceEmbeddings:
        class_ids: list[int] = sorted({int(class_id) for class_id in reference.class_ids.tolist()})
        local_class_ids: IntArray = np.array(
            [class_ids.index(int(class_id)) for class_id in reference.class_ids.tolist()], dtype=np.int64
        )
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
        head.nc = len(class_ids)
        predictor.set_prompts({"bboxes": reference.xyxy, "cls": local_class_ids})
        predictor.setup_model(model=network, verbose=False)
        visual_embeddings: torch.Tensor = cast(torch.Tensor, predictor.get_vpe(self._to_rgb(reference.image)))
        return ReferenceEmbeddings(class_ids=tuple(class_ids), embeddings=visual_embeddings.reshape(len(class_ids), -1))
