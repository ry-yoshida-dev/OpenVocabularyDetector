# open_vocabulary_detector

## Overview

Model-independent interface, prompt, settings and result types for open-vocabulary detection, the YAML presets of every supported model, and the backends implementing the interface.

## Components

| Component | Description |
| --------- | ----------- |
| [detector.py](./detector.py) | `OpenVocabularyDetector`, the runtime-independent base with `detect` and `detect_images`. |
| [runtime/](./runtime/README.md) | Execution of PyTorch-backed models. |
| [cache/](./cache/README.md) | Query embedding caches. |
| [array_types.py](./array_types.py) | NumPy array aliases (`FloatArray`, `IntArray`, `BoolArray`). |
| [settings.py](./settings.py) | `DetectorSettings`, everything needed to load and run a detector; `build()` loads it. |
| [options/](./options/README.md) | Options composing `DetectorSettings`. |
| [config/](./config/README.md) | YAML presets of every backend, buildable into `DetectorSettings`. |
| [prompt/](./prompt/README.md) | `Prompt` and its query types. |
| [result/](./result/README.md) | `DetectionResult`, `Detection`, `ImageSize`. |
| [backends/](./backends/README.md) | Concrete detectors, one sub-package per backend. |

## Examples

```python
detector = settings.build()

result = detector.detect(image, Prompt.from_class_names(("cat", "dog")))
results = detector.detect_images(images, Prompt.from_texts({"car": ("car", "suv", "taxi")}))
```
