# open_vocabulary_detector

## Overview

Model-independent interface, prompt, settings and result types for open-vocabulary detection, the YAML presets of
every supported model, and the backends implementing the interface.

## Components

| Component | Description |
| --------- | ----------- |
| [detector.py](./detector.py) | `OpenVocabularyDetector`: runtime-independent base with `detect`, `detect_images`, prompt kind validation and shared post-processing. |
| [runtime/](./runtime/README.md) | `TorchRuntime` and `ClassPrediction`: device, precision, model preparation and output conversion for PyTorch-backed detectors. |
| [cache/](./cache/README.md) | Query embeddings of OWL-ViT and YOLOE, cached per class name and per visual reference. |
| [array_types.py](./array_types.py) | NumPy array aliases (`FloatArray`, `IntArray`, `BoolArray`). |
| [settings.py](./settings.py) | `OVDSettings`: backend, weights, thresholds, batch size, device and precision; `build()` loads the detector. |
| [backend.py](./backend.py) | `OVDBackend`: detector family and the prompt kinds it supports. |
| [options/](./options/README.md) | `Device` and `DetectionThresholds`: options composing `OVDSettings`. |
| [config/](./config/README.md) | YAML presets of every backend, buildable into `OVDSettings`. |
| [prompt/](./prompt/README.md) | `Prompt` base, `PromptKind`, `VisualReference`, and the concrete prompts `TextPrompt` and `TextVisualPrompt`. |
| [result/](./result/README.md) | `DetectionResult`, `Detection`, `ImageSize`. |
| [backends/](./backends/README.md) | Grounding DINO, OWL-ViT and (optional) Ultralytics detectors. |

## Examples

```python
detector = settings.build()

result = detector.detect(image, TextPrompt(class_names=("cat", "dog")))
results = detector.detect_images(images, TextPrompt(class_names=("cat", "dog")))
```
