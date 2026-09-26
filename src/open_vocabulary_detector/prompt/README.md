# prompt

## Overview

What to detect. Every prompt derives from `Prompt` in [core.py](./core.py), which holds the class names
(class id = index), and each concrete prompt in [prompts/](./prompts/README.md) adds only the inputs its kind needs,
so no backend receives inputs it ignores. A detector accepts the kinds listed in `OVDBackend.supported_prompt_kinds`.

## Components

| Component | Description |
| --------- | ----------- |
| [core.py](./core.py) | `Prompt`: abstract base holding validated class names and the prompt `kind`. |
| [kind.py](./kind.py) | `PromptKind`: `TEXT` or `TEXT_VISUAL`. |
| [visual_reference.py](./visual_reference.py) | `VisualReference`: reference image with XYXY boxes labelled by class id, compared by identity. |
| [prompts/](./prompts/README.md) | Concrete prompts: `TextPrompt` and `TextVisualPrompt`. |

## Examples

```python
reference = VisualReference(image=reference_image, boxes=boxes, class_ids=np.array([1]))
TextVisualPrompt(class_names=("person", "my mug"), visual_references=(reference,))
```
