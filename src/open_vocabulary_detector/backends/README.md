# backends

## Overview

Concrete `OpenVocabularyDetector` implementations, one sub-package per `DetectorBackend`. `DetectorSettings.build()` imports only the sub-package of the requested backend, so importing `open_vocabulary_detector` loads neither `transformers` models nor Ultralytics.

## Components

| Component | Description |
| --------- | ----------- |
| [checkpoint_config.py](./checkpoint_config.py) | `CheckpointConfig`: reads the `model_type` of a `transformers` checkpoint without loading its weights. |
| [grounding_dino/](./grounding_dino/README.md) | Grounding DINO and MM-Grounding-DINO (also LLMDet); text queries. |
| [owl_vit/](./owl_vit/README.md) | OWL-ViT and OWLv2; text and visual queries, also mixed. |
| [omdet_turbo/](./omdet_turbo/README.md) | OmDet-Turbo; text queries. |
| [florence2/](./florence2/README.md) | Florence-2 open-vocabulary detection; text queries, no scores. |
| [ultralytics/](./ultralytics/README.md) | YOLO-World (text queries) and YOLOE (text and visual queries, also mixed). |
