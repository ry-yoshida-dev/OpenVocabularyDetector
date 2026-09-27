# owl_vit

## Overview

OWL-ViT (v1) through `transformers.OwlViTForObjectDetection` (`OVDBackend.OWL_VIT`).

Text queries are embedded by the text encoder (at most 16 tokens each for v1). Visual queries are embedded from their
reference boxes: for each box, the patch whose predicted box overlaps it best and looks least like background is taken
(image-guided detection), and the embeddings of one visual query are averaged. Both kinds may be mixed in one prompt,
even within one class. Text embeddings are cached per text query and reference embeddings per reference.

## Components

| Component | Description |
| --------- | ----------- |
| [detector.py](./detector.py) | `OwlViTDetector`. |
| [box_query_selector.py](./box_query_selector.py) | `OwlViTBoxQuerySelector`: picks the patch embedding representing a reference box. |
