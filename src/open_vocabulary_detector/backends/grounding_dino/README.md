# grounding_dino

## Overview

Grounding DINO through `transformers.GroundingDinoForObjectDetection` (`OVDBackend.GROUNDING_DINO`).

A prompt becomes a caption: its class names are joined into `"cat . traffic cone ."`.
Class spans are mapped to token positions once per prompt, and each query box takes the class whose tokens have the
highest probability, so class ids come back without phrase matching.

Grounding DINO fuses text and image early, so text features cannot be cached per class and the caption is encoded
together with every batch. The caption must fit in `max_text_len` (256) tokens.

## Components

| Component | Description |
| --------- | ----------- |
| [detector.py](./detector.py) | `GroundingDinoDetector`. |
| [caption.py](./caption.py) | `GroundingCaption`: caption text joined from class names and per-class character spans. |
| [tokenized_caption.py](./tokenized_caption.py) | `TokenizedCaption`: token tensors, class-to-token mask and per-class probabilities. |
| [character_span.py](./character_span.py) | `CharacterSpan`: half-open character range. |
