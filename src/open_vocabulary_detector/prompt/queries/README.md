# queries

## Overview

Query types a model scores boxes against: text phrases and reference images.

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
