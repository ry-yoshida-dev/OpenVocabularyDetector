# detector

## Overview

Concrete Grounding DINO family detectors. Each only loads its model architecture; everything else comes from
`GroundingDetector` in [core.py](../core.py).

## Components

| Component | Description |
| --------- | ----------- |
| [original.py](./original.py) | `GroundingDinoDetector`: `GroundingDinoForObjectDetection`. |
| [mm.py](./mm.py) | `MMGroundingDinoDetector`: MM-Grounding-DINO and LLMDet through `MMGroundingDinoWithSeparateBoxHeads`. |
