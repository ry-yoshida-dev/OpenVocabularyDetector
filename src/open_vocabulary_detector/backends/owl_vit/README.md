# owl_vit

## Overview

OWL-ViT and OWLv2 through `transformers` (`DetectorBackend.OWL_VIT`). Both share their heads and query handling;
`OwlVersion` picks the detector from the checkpoint's `model_type` (`owlvit` or `owlv2`).

Text queries are embedded by the text encoder (at most 16 tokens each). Visual queries are embedded from their
reference boxes: for each box, the patch whose predicted box overlaps it best and looks least like background is taken
(image-guided detection), and the embeddings of one visual query are averaged. Both kinds may be mixed in one prompt,
even within one class. Text embeddings are cached per text query and reference embeddings per reference.

OWLv2 pads each image at the bottom and right to a square before resizing it, so its boxes are relative to that
square; `ImageFitting` rescales them to the image and maps reference boxes into the square.

## Components

| Component | Description |
| --------- | ----------- |
| [core.py](./core.py) | `OwlDetector`: shared text and visual query embedding, inference and box rescaling. |
| [detector/](./detector/README.md) | `OwlViTDetector` (v1) and `Owlv2Detector` (v2). |
| [image_fitting.py](./image_fitting.py) | `ImageFitting`: how an image fits the model input, and box mapping between the two. |
| [version.py](./version.py) | `OwlVersion`: version of a checkpoint and the detector that loads it. |
| [box_query_selector.py](./box_query_selector.py) | `OwlBoxQuerySelector`: picks the patch embedding representing a reference box. |
