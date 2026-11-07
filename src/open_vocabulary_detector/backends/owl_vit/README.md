# owl_vit

## Overview

OWL-ViT and OWLv2 through `transformers` (`DetectorBackend.OWL_VIT`), with text and visual queries, also mixed. The checkpoint's `model_type` selects the version.

## Components

| Component | Description |
| --------- | ----------- |
| [core.py](./core.py) | `OwlDetector`, the shared base of both versions. |
| [detector/](./detector/README.md) | `OwlViTDetector` (v1) and `Owlv2Detector` (v2). |
| [image_fitting.py](./image_fitting.py) | `ImageFitting`, how an image fits the model input. |
| [version.py](./version.py) | `OwlVersion`, the version of a checkpoint. |
| [box_query_selector.py](./box_query_selector.py) | `OwlBoxQuerySelector`, which picks the embedding representing a reference box. |
