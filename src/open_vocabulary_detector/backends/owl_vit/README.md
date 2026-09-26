# owl_vit

## Overview

OWL-ViT (v1) through `transformers.OwlViTForObjectDetection` (`OVDBackend.OWL_VIT`).

Classes are queried by their names through the text encoder (at most 16 tokens for v1). In a `TextVisualPrompt`,
classes shown in reference boxes are queried by image instead: for each box, the patch whose predicted box overlaps
it best and looks least like background is taken (image-guided detection), and the embeddings of the same class are
averaged. Text embeddings are cached per class name and reference embeddings per reference.

## Components

| Component | Description |
| --------- | ----------- |
| [detector.py](./detector.py) | `OwlViTDetector`. |
| [box_query_selector.py](./box_query_selector.py) | `OwlViTBoxQuerySelector`: picks the patch embedding representing a reference box. |
