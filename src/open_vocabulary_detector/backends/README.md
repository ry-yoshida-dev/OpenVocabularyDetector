# backends

## Overview

Concrete `OpenVocabularyDetector` implementations. Every detector takes the common `OVDSettings`, declares the
`OVDBackend` it serves, turns a prompt into its model-specific encoding, and runs its model through a
`TorchRuntime`. Model outputs are handed to the shared post-processing of `OpenVocabularyDetector` as
`RawDetections`, so thresholding, NMS and sorting behave identically across backends.

## Components

| Component | Description |
| --------- | ----------- |
| [grounding_dino/](./grounding_dino/README.md) | Grounding DINO; text prompts. |
| [owl_vit/](./owl_vit/README.md) | OWL-ViT; text and text-visual prompts. |
| [ultralytics/](./ultralytics/README.md) | YOLO-World (text prompts) and YOLOE (text and text-visual prompts). |
