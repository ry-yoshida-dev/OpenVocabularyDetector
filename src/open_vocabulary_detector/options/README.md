# options

## Overview

Options composing `DetectorSettings`, shared by every backend.

## Components

| Component | Description |
| --------- | ----------- |
| [backend.py](./backend.py) | `DetectorBackend`: detector family and the query kinds it supports. |
| [device.py](./device.py) | `Device`: `auto`, `cpu`, `cuda` or `mps`. |
| [thresholds.py](./thresholds.py) | `DetectionThresholds`: confidence threshold and optional NMS IoU threshold. |

## Examples

```python
DetectorBackend.YOLOE.supported_prompt_kinds
DetectionThresholds(confidence_threshold=0.3, nms_iou_threshold=0.5)
Device.CUDA
```
