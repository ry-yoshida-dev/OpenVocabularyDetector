# config

## Overview

Ready-made YAML presets of every supported model, one folder per `DetectorBackend` value, named by model variant and size. Each preset's `detector` section maps onto the fields of [`DetectorSettings`](../settings.py); copy a preset into an application config to change the device, precision, thresholds or weights.

## Components

| Component | Presets |
| --------- | ------- |
| [grounding_dino/](./grounding_dino) | Grounding DINO `original_tiny`, `original_base`; MM-Grounding-DINO `mm_tiny_*`, `mm_base_*`, `mm_large_*`; LLMDet `llmdet_tiny`, `llmdet_base`, `llmdet_large` |
| [owl_vit/](./owl_vit) | OWL-ViT `v1_base_patch32`, `v1_base_patch16`, `v1_large_patch14`; OWLv2 `v2_base_patch16`, `v2_large_patch14` and their `_ensemble` weights |
| [omdet_turbo/](./omdet_turbo) | `swin_tiny` |
| [florence2/](./florence2) | `base`, `base_ft`, `large`, `large_ft` |
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
