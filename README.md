# OpenVocabularyDetector

## Overview

`open_vocabulary_detector` puts Grounding DINO, OWL-ViT, YOLO-World and YOLOE behind one interface so that models can
be swapped and compared by changing configuration only. Application code builds a prompt, calls one
`OpenVocabularyDetector`, and reads one `DetectionResult` type; which model runs is decided by an `OVDSettings`,
normally built from a YAML preset.

- **Configuration is data**: every model-specific value (weights, recommended thresholds, batch size) lives in a YAML
  preset under [config/](src/open_vocabulary_detector/config/README.md), one folder per backend. The same
  `OVDSettings` fields apply to every model.
- **Code knows only what cannot be data**: `OVDBackend.supported_prompt_kinds` states which prompt kinds a family
  accepts. A detector rejects a prompt of an unsupported kind before running the model.
- **Prompts carry only what their kind needs**: every prompt holds the class names (class id = index), and each kind
  adds its own inputs.
- **Results are comparable**: boxes are [Geometry](https://github.com/ry-yoshida-dev/Geometry) `Boxes2D` (absolute
  XYXY pixels), confidences are in `[0, 1]`, and every backend applies the confidence threshold, optional class-wise
  NMS and sorting in the same way.

| Prompt | Inputs | Grounding DINO | OWL-ViT | YOLO-World | YOLOE |
| ------ | ------ | :------------: | :-----: | :--------: | :---: |
| `TextPrompt` | Class names | yes | yes | yes | yes |
| `TextVisualPrompt` | Class names plus reference images with boxes | - | yes | - | yes |

| Backend | Library | Presets |
| ------- | ------- | ------- |
| `grounding_dino` | Hugging Face `transformers` | `tiny`, `base` |
| `owl_vit` | Hugging Face `transformers` | `base_patch32`, `base_patch16`, `large_patch14` |
| `yolo_world` | Ultralytics | `s`, `m`, `l`, `x` |
| `yoloe` | Ultralytics | `26n` ... `26x`, `11s` ... `11l`, `v8s` ... `v8l` |

Images are processed in mini-batches. OWL-ViT and YOLOE cache query embeddings per class name and per visual
reference, so prompts that share classes do not recompute them. Weights are downloaded on first use.

For module details, see [src/open_vocabulary_detector/README.md](src/open_vocabulary_detector/README.md).

## Installation

```bash
pip install -e ".[dev]"
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

from open_vocabulary_detector import OVDSettings, TextPrompt

preset_path = "src/open_vocabulary_detector/config/yoloe/26s.yaml"
settings = DictConfigHandler(cfg=OmegaConf.load(preset_path)).build_dataclass(OVDSettings, key="detector")
detector = settings.build()

results = detector.detect_images(images, TextPrompt(class_names=("person", "bus", "traffic light")))
for detection in results[0].filter_by_confidence(0.5):
    print(detection.class_name, detection.confidence, detection.box.value)
```

Visual prompt (OWL-ViT or YOLOE): `"my mug"` is queried by a box in a reference image, `"person"` by its name.

```python
import numpy as np
from geometry import Box2DFormat, Boxes2D

from open_vocabulary_detector import TextVisualPrompt, VisualReference

reference = VisualReference(
    image=reference_image,
    boxes=Boxes2D.register(value=np.array([[120, 80, 260, 240]], dtype=np.float64), box2d_format=Box2DFormat.XYXY),
    class_ids=np.array([1]),
)
result = detector.detect(image, TextVisualPrompt(class_names=("person", "my mug"), visual_references=(reference,)))
```

Command-line example over a directory:

```bash
python examples/detect_directory.py images/ --backend grounding_dino --weights IDEA-Research/grounding-dino-tiny --classes person bus
```
