# config

## Overview

Ready-made YAML presets of every supported model, one folder per `OVDBackend` value, named by model size.

Each preset's `detector` section maps one-to-one onto the fields of
[`OVDSettings`](../settings.py) (enums written by value, `thresholds` as a nested section, `null` disabling
NMS), so it can be built with any dataclass builder such as
`DictConfigHandler.build_dataclass(OVDSettings, key="detector")`. Switching models means switching preset files.

| Folder | Presets |
| ------ | ------- |
| [grounding_dino/](./grounding_dino) | `tiny`, `base` |
| [owl_vit/](./owl_vit) | `base_patch32`, `base_patch16`, `large_patch14` |
| [yolo_world/](./yolo_world) | YOLO-World v2 `s`, `m`, `l`, `x` |
| [yoloe/](./yoloe) | YOLOE text/visual-prompt `26n`-`26x`, `11s`-`11l`, `v8s`-`v8l` (no prompt-free `-pf` weights) |

## Examples

```yaml
detector:
  backend: yoloe
  weights_path: yoloe-26s-seg.pt
  batch_size: 16
  device: auto
  is_half_precision_enabled: false
  thresholds:
    confidence_threshold: 0.25
    nms_iou_threshold: 0.7
```

A preset is a starting point: copy it into an application config to change the device, precision or thresholds, or
point `weights_path` at fine-tuned or local weights of the same backend.
