# OpenVocabularyDetector

## Overview

`open_vocabulary_detector` puts Grounding DINO, OWL-ViT, YOLO-World and YOLOE behind one interface so that models can
be swapped and compared by changing configuration only. Application code builds a prompt, calls one
`OpenVocabularyDetector`, and reads one `DetectionResult` type; which model runs is decided by a `DetectorSettings`,
normally built from a YAML preset.

- **Configuration is data**: every model-specific value (weights, recommended thresholds, batch size) lives in a YAML
  preset under [config/](src/open_vocabulary_detector/config/README.md), one folder per backend. The same
  `DetectorSettings` fields apply to every model.
- **Code knows only what cannot be data**: `DetectorBackend.supported_prompt_kinds` states which query kinds a family
  accepts. A detector rejects a prompt using an unsupported kind before running the model.
- **Classes are separate from queries**: a `Prompt` maps each class name (class id = key order) to its queries, text
  phrases (`TextQuery`), reference images (`VisualQuery`) or both, e.g. `"car"` by `"car"`, `"suv"` and photos of a
  van. The model scores every query, each box keeps its best query, class-wise NMS merges overlapping boxes of one
  class, and every detection reports its class together with the query that matched.
- **Results are comparable**: boxes are [Geometry](https://github.com/ry-yoshida-dev/Geometry) `Boxes2D` (absolute
  XYXY pixels), confidences are in `[0, 1]`, and every backend applies the confidence threshold, optional class-wise
  NMS and sorting in the same way.

| Query | Input | Grounding DINO | OWL-ViT | YOLO-World | YOLOE |
| ----- | ----- | :------------: | :-----: | :--------: | :---: |
| `TextQuery` | Text phrase | yes | yes | yes | yes |
| `VisualQuery` | Reference images with boxes, averaged into one query | - | yes | - | yes |

OWL-ViT and YOLOE accept both kinds mixed in one prompt, even within one class.

| Backend | Library | Presets |
| ------- | ------- | ------- |
| `grounding_dino` | Hugging Face `transformers` | `tiny`, `base` |
| `owl_vit` | Hugging Face `transformers` | `base_patch32`, `base_patch16`, `large_patch14` |
| `yolo_world` | Ultralytics (optional, AGPL-3.0) | `s`, `m`, `l`, `x` |
| `yoloe` | Ultralytics (optional, AGPL-3.0) | `26n` ... `26x`, `11s` ... `11l`, `v8s` ... `v8l` |

Images are processed in mini-batches. OWL-ViT and YOLOE cache query embeddings per text query and per visual
reference, so prompts that share queries do not recompute them. Weights are downloaded on first use.

For module details, see [src/open_vocabulary_detector/README.md](src/open_vocabulary_detector/README.md).

## Installation

```bash
pip install -e .
```

Grounding DINO and OWL-ViT need only the core dependencies. YOLO-World and YOLOE are backed by Ultralytics, which is
licensed under AGPL-3.0 and is therefore an optional extra; install it only when its license terms are acceptable for
your use:

```bash
pip install -e ".[ultralytics]"
```

For development (type checking the Ultralytics backends also requires the `ultralytics` extra):

```bash
pip install -e ".[dev,ultralytics]"
```

## Examples

Build settings from a preset with [DictConfigHandler](https://github.com/ry-yoshida-dev/DictConfigHandler), then run
any model with the same code. DictConfigHandler (which brings in OmegaConf) is not a dependency of this package;
install it separately:

```bash
pip install "dictconfig-handler @ git+https://github.com/ry-yoshida-dev/DictConfigHandler.git"
```

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

Several phrases for one class: boxes are reported as `"car"`, and `detection.matched_query` tells which query matched.
Enable `nms_iou_threshold` so that boxes found by different queries of one class are merged.

```python
prompt = Prompt.from_texts({"car": ("car", "suv", "van", "taxi"), "person": ("person",)})
for detection in detector.detect(image, prompt):
    print(detection.class_name, detection.matched_query, detection.confidence)
```

Text and reference images for one class (OWL-ViT or YOLOE). The reference images of one `VisualQuery` are averaged
into one query, so group photos of the same appearance and give different appearances separate queries.

```python
import numpy as np
from geometry import Box2DFormat, Boxes2D

from open_vocabulary_detector import Prompt, TextQuery, VisualQuery, VisualReference


def reference(image, xyxy):
    return VisualReference(
        image=image, boxes=Boxes2D.register(value=np.array(xyxy, dtype=np.float64), box2d_format=Box2DFormat.XYXY)
    )


prompt = Prompt({
    "car": (
        TextQuery("car"),
        TextQuery("suv"),
        VisualQuery((reference(van_image_1, [[120, 80, 260, 240]]), reference(van_image_2, [[30, 40, 300, 200]]))),
    ),
    "my mug": (VisualQuery((reference(mug_image, [[10, 10, 90, 120]]),)),),
})
result = detector.detect(image, prompt)
```

Text and visual scores come from different distributions, so check the confidence threshold on real data when mixing
them.

Command-line example over a directory:

```bash
python examples/detect_directory.py images/ --backend grounding_dino --weights IDEA-Research/grounding-dino-tiny --classes person "car:car,suv,taxi"
```
