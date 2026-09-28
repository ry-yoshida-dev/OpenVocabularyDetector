# omdet_turbo

## Overview

OmDet-Turbo through `transformers.OmDetTurboForObjectDetection` (`DetectorBackend.OMDET_TURBO`). Its vision backbone
needs `timm` (`pip install "open-vocabulary-detector[omdet-turbo]"`).

Every text query is encoded as one class, and the prompt as the task sentence `"Detect car, suv, person."`. Both are
tokenized once per prompt and reused while the same prompt is detected; the model caches the text embeddings of
recently seen classes and tasks itself. Each predicted box takes its most probable query. Queries longer than 77 CLIP
tokens are rejected; a longer task sentence is truncated, like in the `transformers` processor.

## Components

| Component | Description |
| --------- | ----------- |
| [detector.py](./detector.py) | `OmDetTurboDetector`. |
| [prompt_encoding.py](./prompt_encoding.py) | `PromptEncoding`: class and task tokens of the prompt currently detected. |
