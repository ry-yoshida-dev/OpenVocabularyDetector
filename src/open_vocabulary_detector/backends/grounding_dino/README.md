# grounding_dino

## Overview

Grounding DINO through `transformers.GroundingDinoForObjectDetection` (`DetectorBackend.GROUNDING_DINO`).

A prompt becomes a caption: its text queries are joined into `"car . suv . traffic cone ."`.
The caption is tokenized and query spans are mapped to token positions once per prompt, then reused while the same
prompt is detected. Each predicted box takes the query whose tokens have the highest probability, so query ids (and
their classes) come back without phrase matching.

Grounding DINO fuses text and image early, so text features cannot be cached per query and the caption is encoded
together with every batch. The caption must fit in `max_text_len` (256) tokens.
Grounding DINO splits phrases at `.` and `?`, so text queries containing them are rejected.

## Components

| Component | Description |
| --------- | ----------- |
| [detector.py](./detector.py) | `GroundingDinoDetector`. |
| [active_caption.py](./active_caption.py) | `ActiveCaption`: tokenized caption of the prompt currently detected, reused across batches. |
| [caption.py](./caption.py) | `GroundingCaption`: caption text joined from text queries and per-query character spans. |
| [tokenized_caption.py](./tokenized_caption.py) | `TokenizedCaption`: token tensors, query-to-token mask and per-query probabilities. |
| [character_span.py](./character_span.py) | `CharacterSpan`: half-open character range. |
