# prompts

## Overview

Concrete prompts deriving from `Prompt`, one per `PromptKind`.

| Prompt | Kind | Inputs | Backends |
| ------ | ---- | ------ | -------- |
| `TextPrompt` | `TEXT` | Class names | All |
| `TextVisualPrompt` | `TEXT_VISUAL` | Class names plus `VisualReference`s; referenced classes are queried by image, the others by name | OWL-ViT, YOLOE |

## Components

| Component | Description |
| --------- | ----------- |
| [text.py](./text.py) | `TextPrompt`: classes queried by their names. |
| [text_visual.py](./text_visual.py) | `TextVisualPrompt`: classes queried by reference boxes where given, by name otherwise. |

## Examples

```python
TextPrompt(class_names=("person", "a photo of a traffic cone"))
TextVisualPrompt(class_names=("person", "my mug"), visual_references=(reference,))
```
