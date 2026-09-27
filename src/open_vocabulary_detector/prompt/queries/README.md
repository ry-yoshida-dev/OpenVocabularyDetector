# queries

## Overview

What a model scores boxes against. A `Prompt` lists the queries of each class; every query is scored on its own and
the best one decides the class of a box. Code dispatches on the query type with `match`.

A `VisualQuery` holds one or more reference images whose box embeddings are averaged into one query: use it for
several photos of the same appearance. Appearances that differ (a sedan and a van) belong in separate queries.

## Components

| Component | Description |
| --------- | ----------- |
| [core.py](./core.py) | `PromptQuery`: type alias of every query type. |
| [text.py](./text.py) | `TextQuery`: the phrase the model reads. |
| [visual.py](./visual.py) | `VisualQuery`: reference images averaged into one query. |

## Examples

```python
match detection.matched_query:
    case TextQuery(text=text):
        print("matched phrase", text)
    case VisualQuery(references=references):
        print("matched", len(references), "reference images")
```
