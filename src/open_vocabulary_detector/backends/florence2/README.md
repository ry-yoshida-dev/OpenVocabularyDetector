# florence2

## Overview

Florence-2 through `transformers` (`DetectorBackend.FLORENCE2`), using its `<OPEN_VOCABULARY_DETECTION>` task. Every text query is generated separately, and every box has confidence `1.0`.

## Components

| Component | Description |
| --------- | ----------- |
| [detector.py](./detector.py) | `Florence2Detector`. |
| [location_parser.py](./location_parser.py) | `LocationParser`, which reads generated location tokens into boxes. |
| [token_generator.py](./token_generator.py) | `TokenGenerator`, the typed protocol of `generate`. |
