# backends

## Overview

Concrete `OpenVocabularyDetector` implementations. Every detector takes the common `DetectorSettings`, declares the
`DetectorBackend` it serves, turns a prompt into its model-specific encoding, and runs its model through a
`TorchRuntime`. Model outputs are handed to the shared post-processing of `OpenVocabularyDetector`, so thresholding,
NMS and sorting behave identically across backends.

Each backend package keeps its shared base class in `core.py` and the concrete detectors in `detector/`. A backend
groups the models sharing one implementation: `grounding_dino` and `owl_vit` choose their detector from the
`model_type` of the checkpoint (`CheckpointConfig`), so `weights_path` alone decides the architecture.

Nothing is imported eagerly: `DetectorSettings.build()` imports only the sub-package of the requested backend, so
importing `open_vocabulary_detector` loads neither `transformers` models nor Ultralytics. Import a detector class
directly from its sub-package (e.g. `backends.grounding_dino`) when needed. OmDet-Turbo needs the optional
`omdet-turbo` extra (`timm`) and the Ultralytics detectors the optional `ultralytics` extra.

## Components

| Component | Description |
| --------- | ----------- |
| [checkpoint_config.py](./checkpoint_config.py) | `CheckpointConfig`: reads the `model_type` of a `transformers` checkpoint without loading its weights. |
| [grounding_dino/](./grounding_dino/README.md) | Grounding DINO and MM-Grounding-DINO (also LLMDet); text queries. |
| [owl_vit/](./owl_vit/README.md) | OWL-ViT and OWLv2; text and visual queries, also mixed. |
| [omdet_turbo/](./omdet_turbo/README.md) | OmDet-Turbo; text queries. |
| [florence2/](./florence2/README.md) | Florence-2 open-vocabulary detection; text queries, no scores. |
| [ultralytics/](./ultralytics/README.md) | YOLO-World (text queries) and YOLOE (text and visual queries, also mixed). |
