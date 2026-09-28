# detector

## Overview

Concrete OWL detectors. Each loads its model and processor, exposes its text encoder and sets its `ImageFitting`;
everything else comes from `OwlDetector` in [core.py](../core.py).

## Components

| Component | Description |
| --------- | ----------- |
| [v1.py](./v1.py) | `OwlViTDetector`: OWL-ViT, images stretched to the input without padding. |
| [v2.py](./v2.py) | `Owlv2Detector`: OWLv2, images padded to a square. |
