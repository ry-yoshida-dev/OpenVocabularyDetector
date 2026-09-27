# prompt

## Overview

What to detect. A `Prompt` maps each output class name (class id = key order) to the queries the model scores for
it. A class may be queried by several text phrases, several visual queries, or both at once, e.g. `"car"` by
`"car"`, `"suv"` and photos of a van. The model scores every query, each box keeps its best one, and the detection
is reported under the class of that query together with the query itself (`Detection.matched_query`).

A detector accepts a prompt when `OVDBackend.supported_prompt_kinds` covers every query kind it uses
(`Prompt.kinds`): text only for Grounding DINO and YOLO-World, text and visual (also mixed) for OWL-ViT and YOLOE.

## Components

| Component | Description |
| --------- | ----------- |
| [core.py](./core.py) | `Prompt`: class names, queries in query-id order and the class of each query, with validation. |
| [kind.py](./kind.py) | `PromptKind`: `TEXT` or `VISUAL` query. |
| [visual_reference.py](./visual_reference.py) | `VisualReference`: reference image with XYXY boxes, compared by identity. |
| [queries/](./queries/README.md) | Query types: `TextQuery`, `VisualQuery` and their alias `PromptQuery`. |

## Examples

```python
Prompt.from_class_names(("person", "dog"))

prompt = Prompt.from_texts({"car": ("car", "suv", "taxi"), "person": ("person",)})
prompt.query_texts  # ("car", "suv", "taxi", "person")

van_references = (VisualReference(image=van_image_1, boxes=boxes_1), VisualReference(image=van_image_2, boxes=boxes_2))
Prompt({
    "car": (TextQuery("car"), TextQuery("suv"), VisualQuery(van_references)),
    "my mug": (VisualQuery((mug_reference,)),),
})
```
