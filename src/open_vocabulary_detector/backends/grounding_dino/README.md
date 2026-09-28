# grounding_dino

## Overview

Grounding DINO and MM-Grounding-DINO through `transformers` (`DetectorBackend.GROUNDING_DINO`). Both share the
processor, the caption handling and the output format; `GroundingDinoVariant` picks the detector from the checkpoint's
`model_type` (`grounding-dino` or `mm-grounding-dino`). LLMDet checkpoints are MM-Grounding-DINO models.

A prompt becomes a caption: its text queries are joined into `"car . suv . traffic cone ."`.
The caption is tokenized and query spans are mapped to token positions once per prompt, then reused while the same
prompt is detected. Each predicted box takes the query whose tokens have the highest probability, so query ids (and
their classes) come back without phrase matching.

Grounding DINO fuses text and image early, so text features cannot be cached per query and the caption is encoded
together with every batch. The caption must fit in `max_text_len` (256) tokens.
Grounding DINO splits phrases at `.` and `?`, so text queries containing them are rejected.

`transformers` ties the box heads of all decoder layers of MM-Grounding-DINO to the first one, which overwrites the
separate heads of the published MM-Grounding-DINO and LLMDet checkpoints and shifts their boxes.
`MMGroundingDinoWithSeparateBoxHeads` keeps them apart when the checkpoint sets `decoder_bbox_embed_share: false`,
and raises `RuntimeError` if `transformers` no longer declares that tie in the form it drops, instead of silently
loading shifted boxes.

## Components

| Component | Description |
| --------- | ----------- |
| [core.py](./core.py) | `GroundingDetector`: shared caption handling, inference and post-processing. |
| [detector/](./detector/README.md) | `GroundingDinoDetector` and `MMGroundingDinoDetector`. |
| [variant.py](./variant.py) | `GroundingDinoVariant`: architecture of a checkpoint and the detector that loads it. |
| [separate_box_heads.py](./separate_box_heads.py) | `MMGroundingDinoWithSeparateBoxHeads`: MM-Grounding-DINO keeping per-layer box heads. |
| [active_caption.py](./active_caption.py) | `ActiveCaption`: tokenized caption of the prompt currently detected, reused across batches. |
| [caption.py](./caption.py) | `GroundingCaption`: caption text joined from text queries and per-query character spans. |
| [tokenized_caption.py](./tokenized_caption.py) | `TokenizedCaption`: token tensors, query-to-token mask and per-query probabilities. |
| [character_span.py](./character_span.py) | `CharacterSpan`: half-open character range. |
