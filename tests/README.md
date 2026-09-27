# tests

## Overview

Unit tests that run without downloading model weights.

## Components

| Component | Description |
| --------- | ----------- |
| [test_prompt.py](./test_prompt.py) | Prompts with text, visual and mixed queries: validation, kinds, query-to-class mapping, visual references and equality. |
| [test_configuration.py](./test_configuration.py) | YAML presets matching the `OVDSettings` schema, backend capabilities and settings validation. |
| [test_detector.py](./test_detector.py) | Shared mini-batching, post-processing (threshold, NMS, sorting), merging of mixed queries per class, query kind and backend checks. |
| [test_package_import.py](./test_package_import.py) | Importing the package loads no backend library (`transformers` models, Ultralytics). |
| [test_torch_runtime.py](./test_torch_runtime.py) | `TorchRuntime` device resolution, dtype and model preparation. |
| [test_detection_result.py](./test_detection_result.py) | `DetectionResult` construction, filtering, sorting and NMS. |
| [test_grounding_caption.py](./test_grounding_caption.py) | Grounding DINO captions joined from text queries. |
| [test_tokenized_caption.py](./test_tokenized_caption.py) | Grounding DINO class-to-token mapping and per-class confidence. |
| [test_visual_query.py](./test_visual_query.py) | Reference box embedding selection (OWL-ViT). |
| [test_query_embedding_store.py](./test_query_embedding_store.py) | Query embeddings cached per text query and per visual reference, averaging of the references of a visual query, mixed query order, and release of unused references. |
