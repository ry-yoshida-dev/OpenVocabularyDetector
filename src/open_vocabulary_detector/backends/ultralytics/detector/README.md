# detector

## Overview

Concrete Ultralytics detectors. Each loads its model and applies a prompt to it; prediction and post-processing come
from `UltralyticsDetector` in [core.py](../core.py).

## Components

| Component | Description |
| --------- | ----------- |
| [yolo_world.py](./yolo_world.py) | `YoloWorldDetector`: text queries. |
| [yolo_e.py](./yolo_e.py) | `YoloEDetector`, including visual prompt embedding. |
