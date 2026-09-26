# caches

## Overview

Concrete `EmbeddingCache`s, one per cached element.

## Components

| Component | Description |
| --------- | ----------- |
| [class_name.py](./class_name.py) | `ClassNameEmbeddingCache`: text embedding per class name, kept in a dict. |
| [visual_reference.py](./visual_reference.py) | `VisualReferenceEmbeddingCache`: box embeddings per `VisualReference`, held weakly and dropped with the reference. |
