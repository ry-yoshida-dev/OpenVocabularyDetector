# omdet_turbo

## Overview

OmDet-Turbo through `transformers` (`DetectorBackend.OMDET_TURBO`), with text queries. Its vision backbone needs the `omdet-turbo` extra (`timm`).

## Components

| Component | Description |
| --------- | ----------- |
| [detector.py](./detector.py) | `OmDetTurboDetector`. |
| [prompt_encoding.py](./prompt_encoding.py) | `PromptEncoding`, the tokenized current prompt. |
