# options

## Overview

Individual options composing `OVDSettings`, shared by every backend: where the detector runs and how its output is
post-processed.

## Components

| Component | Description |
| --------- | ----------- |
| [device.py](./device.py) | `Device`: `auto`, `cpu`, `cuda` or `mps`; each runtime resolves it to its own device. |
| [thresholds.py](./thresholds.py) | `DetectionThresholds`: confidence threshold and optional NMS IoU threshold. |

## Examples

```python
DetectionThresholds(confidence_threshold=0.3, nms_iou_threshold=0.5)
Device.CUDA
```
