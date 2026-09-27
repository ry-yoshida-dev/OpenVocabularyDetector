# cache

## Overview

Query embeddings of OWL-ViT and YOLOE, cached at the granularity they depend on. A text embedding depends only on
its text query and a visual embedding only on its reference, so each is computed once per text query or reference;
`QueryEmbeddingStore` assembles the per-query embeddings of a prompt from these caches. Every cache derives from
`EmbeddingCache` in [core.py](./core.py) and differs only in how it stores entries ([caches/](./caches/README.md)).

## Components

| Component | Description |
| --------- | ----------- |
| [core.py](./core.py) | `EmbeddingCache`: abstract base computing missing keys in one call and returning values in key order. |
| [store.py](./store.py) | `QueryEmbeddingStore`: per-query embeddings of a prompt with text queries, visual queries or both, in query-id order; averages the reference boxes of each visual query. |
| [caches/](./caches/README.md) | Concrete caches per text query and per visual reference. |

## Examples

```python
store = QueryEmbeddingStore(embed_texts=embed_texts, embed_references=embed_references)
query_embeddings = store.embed(prompt)
```
