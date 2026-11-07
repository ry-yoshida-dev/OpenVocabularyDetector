# ultralytics

## Overview

YOLO-World (`DetectorBackend.YOLO_WORLD`, text queries) and YOLOE (`DetectorBackend.YOLOE`, text and visual queries, also mixed) through Ultralytics, which is licensed under AGPL-3.0 and needs the `ultralytics` extra.

## Components

| Component | Description |
| --------- | ----------- |
| [core.py](./core.py) | `UltralyticsDetector`, the shared base of both families. |
| [detector/](./detector/README.md) | `YoloWorldDetector` and `YoloEDetector`. |
| [installation.py](./installation.py) | Fails fast with an installation hint when Ultralytics is missing. |
