# caches

## Overview

Concrete `EmbeddingCache`s, one per cached element.

## Components

| Component | Description |
| --------- | ----------- |
| [text.py](./text.py) | `TextEmbeddingCache`: text embedding per text query, kept in a dict. |
| [visual.py](./visual.py) | `VisualEmbeddingCache`: box embeddings per `VisualReference`, held weakly and dropped with the reference. |
