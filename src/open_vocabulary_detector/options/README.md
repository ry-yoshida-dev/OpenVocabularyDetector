# options

## Overview

Individual options composing `DetectorSettings`, shared by every backend: which detector family runs, where it runs
and how its output is post-processed.

## Components

| Component | Description |
| --------- | ----------- |
| [backend.py](./backend.py) | `DetectorBackend`: detector family and the query kinds it supports. |
| [device.py](./device.py) | `Device`: `auto`, `cpu`, `cuda` or `mps`; each runtime resolves it to its own device. |
| [thresholds.py](./thresholds.py) | `DetectionThresholds`: confidence threshold and optional NMS IoU threshold. |

## Examples

```python
DetectorBackend.YOLOE.supported_prompt_kinds
DetectionThresholds(confidence_threshold=0.3, nms_iou_threshold=0.5)
Device.CUDA
```
