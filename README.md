# OpenVocabularyDetector

## Overview

`open_vocabulary_detector` puts open-vocabulary object detectors behind one interface, so that models can be swapped and compared by changing configuration only. Application code builds a `Prompt` that maps each class name to text phrases (`TextQuery`), reference images (`VisualQuery`) or both, calls one `OpenVocabularyDetector`, and reads one `DetectionResult` type with [Geometry](https://github.com/ry-yoshida-dev/Geometry) boxes in absolute XYXY pixels and confidences in `[0, 1]`. Which model runs is decided by a `DetectorSettings`, normally built from a YAML [preset](src/open_vocabulary_detector/config/README.md).

| Backend | Models | Library | Text queries | Visual queries | Presets |
| ------- | ------ | ------- | :----------: | :------------: | ------- |
| `grounding_dino` | Grounding DINO, MM-Grounding-DINO, LLMDet | `transformers` | yes | - | `original_tiny`, `original_base`, `mm_*`, `llmdet_tiny` ... `llmdet_large` |
| `owl_vit` | OWL-ViT, OWLv2 | `transformers` | yes | yes | `v1_base_patch32`, `v1_base_patch16`, `v1_large_patch14`, `v2_base_patch16[_ensemble]`, `v2_large_patch14[_ensemble]` |
| `omdet_turbo` | OmDet-Turbo | `transformers` + `timm` (optional) | yes | - | `swin_tiny` |
| `florence2` | Florence-2 | `transformers` | yes | - | `base`, `base_ft`, `large`, `large_ft` |
| `yolo_world` | YOLO-World | Ultralytics (optional, AGPL-3.0) | yes | - | `s`, `m`, `l`, `x` |
| `yoloe` | YOLOE | Ultralytics (optional, AGPL-3.0) | yes | yes | `26n` ... `26x`, `11s` ... `11l`, `v8s` ... `v8l` |

Backends with visual queries accept both kinds mixed in one prompt, even within one class. Florence-2 is generative and gives no scores: every box has confidence `1.0`, and inference time grows with the number of queries.

## Installation

```bash
pip install .
```

OmDet-Turbo needs `timm`, and YOLO-World and YOLOE need Ultralytics, which is licensed under AGPL-3.0; install these extras only when needed and, for Ultralytics, when its license terms are acceptable for your use:

```bash
pip install ".[omdet-turbo]"
pip install ".[ultralytics]"
```

## Usage

Build settings from a preset with [DictConfigHandler](https://github.com/ry-yoshida-dev/DictConfigHandler), which is not a dependency of this package (`pip install "dictconfig-handler @ git+https://github.com/ry-yoshida-dev/DictConfigHandler.git"`), then run any model with the same code:

```python
from dictconfig_handler import DictConfigHandler
from omegaconf import OmegaConf

from open_vocabulary_detector import DetectorSettings, Prompt

preset_path = "src/open_vocabulary_detector/config/yoloe/26s.yaml"
settings = DictConfigHandler(cfg=OmegaConf.load(preset_path)).build_dataclass(DetectorSettings, key="detector")
detector = settings.build()

results = detector.detect_images(images, Prompt.from_class_names(("person", "bus", "traffic light")))
for detection in results[0].filter_by_confidence(0.5):
    print(detection.class_name, detection.confidence, detection.box.value)
```

Give a class several phrases; `detection.matched_query` tells which one matched, and `nms_iou_threshold` merges boxes found by different queries of one class:

```python
prompt = Prompt.from_texts({"car": ("car", "suv", "van", "taxi"), "person": ("person",)})
for detection in detector.detect(image, prompt):
    print(detection.class_name, detection.matched_query, detection.confidence)
```

Mix text and reference images (OWL-ViT, OWLv2 or YOLOE):

```python
from open_vocabulary_detector import Prompt, TextQuery, VisualQuery, VisualReference

van_references = (VisualReference(image=van_image_1, boxes=boxes_1), VisualReference(image=van_image_2, boxes=boxes_2))
prompt = Prompt({"car": (TextQuery("car"), TextQuery("suv"), VisualQuery(van_references))})
result = detector.detect(image, prompt)
```

See [examples/](examples/README.md) for a command-line script over a directory.

## Development

Type checking the Ultralytics backends requires the `ultralytics` extra:

```bash
pip install -e ".[dev,ultralytics]"
```
