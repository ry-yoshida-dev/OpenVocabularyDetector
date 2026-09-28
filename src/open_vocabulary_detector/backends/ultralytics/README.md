# ultralytics

## Overview

YOLO-World (`DetectorBackend.YOLO_WORLD`) and YOLOE (`DetectorBackend.YOLOE`) through Ultralytics.

Ultralytics is licensed under AGPL-3.0 and is an optional dependency (`pip install "open-vocabulary-detector[ultralytics]"`).
Importing this package without it raises `ModuleNotFoundError` with that hint; the rest of `open_vocabulary_detector`
does not import it.

The query embeddings are stored inside the Ultralytics model as its "classes", so a detector holds one active prompt
and re-applies a prompt only when it differs from the active one. Confidence thresholding and query-wise NMS run
inside Ultralytics; the predicted query ids are then folded into output classes and NMS runs again per class, like
every other backend. `nms_iou_threshold=None` keeps every box.

Unlike the `transformers` backends, Ultralytics keeps at most 300 detections per image. End-to-end heads (YOLO26,
YOLOE-26) fix this limit inside the model, so it is left at the Ultralytics default for every model.

- YOLO-World (text queries): applying a prompt sets the text queries (CLIP text encoding, cached by Ultralytics).
- YOLOE (text and visual queries, also mixed): applying a prompt computes text embeddings for the text queries and
  visual prompt embeddings of every reference box (averaged per visual query), then sets them on the model. The
  visual prompt predictor holds its own copy of the network; it is built on the first visual query and reused.

## Components

| Component | Description |
| --------- | ----------- |
| [core.py](./core.py) | `UltralyticsDetector`: shared prediction and prompt activation logic. |
| [detector/](./detector/README.md) | `YoloWorldDetector` and `YoloEDetector`. |
| [installation.py](./installation.py) | Fails fast with an installation hint when Ultralytics is missing. |
