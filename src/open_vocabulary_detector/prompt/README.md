# prompt

## Overview

What to detect. A `Prompt` maps each output class name to the queries the model scores for it: several text phrases, visual queries, or both, e.g. `"car"` by `"car"`, `"suv"` and photos of a van. Each detection reports its class and the query that matched.

## Components

| Component | Description |
| --------- | ----------- |
| [core.py](./core.py) | `Prompt`, the classes to detect and their queries. |
| [kind.py](./kind.py) | `PromptKind`: `TEXT` or `VISUAL` query. |
| [visual_reference.py](./visual_reference.py) | `VisualReference`, a reference image with XYXY boxes. |
| [queries/](./queries/README.md) | Query types: `TextQuery`, `VisualQuery` and their alias `PromptQuery`. |

## Examples

```python
Prompt.from_class_names(("person", "dog"))

prompt = Prompt.from_texts({"car": ("car", "suv", "taxi"), "person": ("person",)})
prompt.query_texts  # ("car", "suv", "taxi", "person")

van_references = (VisualReference(image=van_image_1, boxes=boxes_1), VisualReference(image=van_image_2, boxes=boxes_2))
Prompt(
    {
        "car": (TextQuery("car"), TextQuery("suv"), VisualQuery(van_references)),
        "my mug": (VisualQuery((mug_reference,)),),
    }
)
```
