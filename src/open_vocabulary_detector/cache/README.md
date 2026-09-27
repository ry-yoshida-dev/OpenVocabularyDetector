# cache

## Overview

Query embeddings of OWL-ViT and YOLOE, cached at the granularity they depend on. A text embedding depends only on
its text query and a visual embedding only on its reference, so each is computed once per text query or reference;
`QueryEmbeddingStore` assembles the per-query embeddings of a prompt from these caches: text embeddings are kept in a
`dict`, reference embeddings in a `WeakKeyDictionary` so they are dropped together with their reference.

## Components

| Component | Description |
| --------- | ----------- |
| [core.py](./core.py) | `EmbeddingCache`: computes missing keys in one call, stores values in a given mapping and returns them in key order. |
| [store.py](./store.py) | `QueryEmbeddingStore`: per-query embeddings of a prompt with text queries, visual queries or both, in query-id order; averages the reference boxes of each visual query. |

## Examples

```python
store = QueryEmbeddingStore(embed_texts=embed_texts, embed_references=embed_references)
query_embeddings = store.embed(prompt)
```
