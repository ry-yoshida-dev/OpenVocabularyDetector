# open_vocabulary_detector

## Overview

Model-independent interface, prompt, settings and result types for open-vocabulary detection, the YAML presets of
every supported model, and the backends implementing the interface.

## Components

| Component | Description |
| --------- | ----------- |
| [detector.py](./detector.py) | `OpenVocabularyDetector`: runtime-independent base with `detect`, `detect_images`, prompt kind validation and shared post-processing. |
| [runtime/](./runtime/README.md) | `TorchRuntime` and `QueryPrediction`: device, precision, model preparation and output conversion for PyTorch-backed detectors. |
| [cache/](./cache/README.md) | Query embeddings of OWL-ViT, OWLv2 and YOLOE, cached per text query and per visual reference. |
| [array_types.py](./array_types.py) | NumPy array aliases (`FloatArray`, `IntArray`, `BoolArray`). |
| [settings.py](./settings.py) | `DetectorSettings`: backend, weights, thresholds, batch size, device and precision; `build()` loads the detector. |
| [options/](./options/README.md) | `DetectorBackend`, `Device` and `DetectionThresholds`: options composing `DetectorSettings`. |
| [config/](./config/README.md) | YAML presets of every backend, buildable into `DetectorSettings`. |
| [prompt/](./prompt/README.md) | `Prompt` mapping each class name to its text and visual queries, the query types and `VisualReference`. |
| [result/](./result/README.md) | `DetectionResult`, `Detection`, `ImageSize`. |
| [backends/](./backends/README.md) | Grounding DINO / MM-Grounding-DINO, OWL-ViT / OWLv2, OmDet-Turbo, Florence-2 and (optional) Ultralytics detectors. |

## Examples

```python
detector = settings.build()

result = detector.detect(image, Prompt.from_class_names(("cat", "dog")))
results = detector.detect_images(images, Prompt.from_texts({"car": ("car", "suv", "taxi")}))
```
