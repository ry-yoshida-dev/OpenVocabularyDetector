# grounding_dino

## Overview

Grounding DINO and MM-Grounding-DINO (also LLMDet) through `transformers` (`DetectorBackend.GROUNDING_DINO`), with text queries joined into one caption. The checkpoint's `model_type` selects the architecture.

## Components

| Component | Description |
| --------- | ----------- |
| [core.py](./core.py) | `GroundingDetector`, the shared base of the family. |
| [detector/](./detector/README.md) | `GroundingDinoDetector` and `MMGroundingDinoDetector`. |
| [variant.py](./variant.py) | `GroundingDinoVariant`, the architecture of a checkpoint. |
| [separate_box_heads.py](./separate_box_heads.py) | `MMGroundingDinoWithSeparateBoxHeads`, MM-Grounding-DINO keeping per-layer box heads. |
| [active_caption.py](./active_caption.py) | `ActiveCaption`, the tokenized caption of the current prompt. |
| [caption.py](./caption.py) | `GroundingCaption`, the caption joined from text queries. |
| [tokenized_caption.py](./tokenized_caption.py) | `TokenizedCaption`, a caption's tokens and their mapping to queries. |
| [character_span.py](./character_span.py) | `CharacterSpan`, a character range in a caption. |
