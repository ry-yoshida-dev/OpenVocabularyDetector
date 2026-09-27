# backends

## Overview

Concrete `OpenVocabularyDetector` implementations. Every detector takes the common `DetectorSettings`, declares the
`DetectorBackend` it serves, turns a prompt into its model-specific encoding, and runs its model through a
`TorchRuntime`. Model outputs are handed to the shared post-processing of `OpenVocabularyDetector` as
`RawDetections`, so thresholding, NMS and sorting behave identically across backends.

Nothing is imported eagerly: `DetectorSettings.build()` imports only the sub-package of the requested backend, so importing
`open_vocabulary_detector` loads neither `transformers` models nor Ultralytics. Import a detector class directly from
its sub-package (e.g. `backends.grounding_dino`) when needed. The Ultralytics detectors also need the optional
`ultralytics` extra.

## Components

| Component | Description |
| --------- | ----------- |
| [grounding_dino/](./grounding_dino/README.md) | Grounding DINO; text queries. |
| [owl_vit/](./owl_vit/README.md) | OWL-ViT; text and visual queries, also mixed. |
| [ultralytics/](./ultralytics/README.md) | YOLO-World (text queries) and YOLOE (text and visual queries, also mixed). |
