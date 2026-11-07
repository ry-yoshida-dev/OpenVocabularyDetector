# cache

## Overview

Query embeddings of OWL-ViT, OWLv2 and YOLOE, cached per text query and per visual reference, so prompts that share queries do not recompute them.

## Components

| Component | Description |
| --------- | ----------- |
| [core.py](./core.py) | `EmbeddingCache`, embeddings computed per key and kept for later prompts. |
| [store.py](./store.py) | `QueryEmbeddingStore`, which assembles the query embeddings of a prompt from the caches. |

## Examples

```python
store = QueryEmbeddingStore(embed_texts=embed_texts, embed_references=embed_references)
query_embeddings = store.embed(prompt)
```
