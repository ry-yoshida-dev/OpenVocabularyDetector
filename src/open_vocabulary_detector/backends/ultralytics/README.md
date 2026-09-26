# ultralytics

## Overview

YOLO-World (`OVDBackend.YOLO_WORLD`) and YOLOE (`OVDBackend.YOLOE`) through Ultralytics.

The class embeddings are stored inside the Ultralytics model, so a detector holds one active prompt and re-applies a
prompt only when it differs from the active one. Confidence thresholding and NMS run inside Ultralytics; NMS is
always class-wise like every other backend, and `nms_iou_threshold=None` keeps every box.

- YOLO-World (`TextPrompt`): applying a prompt sets the classes (CLIP text encoding, cached by Ultralytics).
- YOLOE (`TextPrompt`, `TextVisualPrompt`): applying a prompt computes text embeddings for name-queried classes and
  visual prompt embeddings for classes shown in reference boxes (averaged per class), then sets them on the model.

## Components

| Component | Description |
| --------- | ----------- |
| [detector.py](./detector.py) | `UltralyticsDetector`: shared prediction and prompt activation logic. |
| [yolo_world.py](./yolo_world.py) | `YoloWorldDetector`. |
| [yolo_e.py](./yolo_e.py) | `YoloEDetector`, including visual prompt embedding. |
| [detection_head.py](./detection_head.py) | `DetectionHead`: typed protocol for the class count of an Ultralytics head. |
