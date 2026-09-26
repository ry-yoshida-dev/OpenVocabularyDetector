# cache

## Overview

Query embeddings of OWL-ViT and YOLOE, cached at the granularity they depend on. A text embedding depends only on
its class name and a visual embedding only on its reference, so each is computed once per class name or reference;
`QueryEmbeddingStore` assembles the per-class embeddings of a prompt from these caches. Every cache derives from
`EmbeddingCache` in [core.py](./core.py) and differs only in how it stores entries ([caches/](./caches/README.md)).

## Components

| Component | Description |
| --------- | ----------- |
| [core.py](./core.py) | `EmbeddingCache`: abstract base computing missing keys in one call and returning values in key order. |
| [store.py](./store.py) | `QueryEmbeddingStore`: per-class query embeddings of a text or text-visual prompt. |
| [reference_embeddings.py](./reference_embeddings.py) | `ReferenceEmbeddings`: embeddings and class ids extracted from one visual reference. |
| [class_embedding_averager.py](./class_embedding_averager.py) | `ClassEmbeddingAverager`: merges embeddings of the same class into one. |
| [caches/](./caches/README.md) | Concrete caches per class name and per visual reference. |

## Examples

```python
store = QueryEmbeddingStore(embed_class_names=embed_class_names, embed_references=embed_references)
query_embeddings = store.embed(prompt)
```
