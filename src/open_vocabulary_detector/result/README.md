# result

## Overview

Model-independent detection output shared by every detector, with boxes as [Geometry](https://github.com/ry-yoshida-dev/Geometry) `Boxes2D` in absolute XYXY pixels.

## Components

| Component | Description |
| --------- | ----------- |
| [detection_result.py](./detection_result.py) | `DetectionResult`, the detections of one image, with filtering, sorting and NMS. |
| [detection.py](./detection.py) | `Detection`, one detected object, yielded by iterating a result. |
| [image_size.py](./image_size.py) | `ImageSize`: validated image width and height. |
| [raw_detections.py](./raw_detections.py) | `RawDetections`, the model output of a mini-batch before post-processing. |

## Examples

```python
result = detector.detect(image, Prompt.from_class_names(("person", "dog")))

confident = result.filter_by_confidence(0.5)
print(confident.boxes.area, confident.boxes.center)

for detection in result.sort_by_confidence():
    print(detection.class_name, detection.matched_query, detection.confidence, detection.box.value)
```
