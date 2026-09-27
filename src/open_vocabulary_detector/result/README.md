# result

## Overview

Model-independent detection output shared by every detector.
Boxes are stored as `geometry.Boxes2D` in absolute XYXY pixel coordinates, so downstream code can use the
[Geometry](https://github.com/ry-yoshida-dev/Geometry) API (`area`, `center`, `crop_slice`, `shapely`, format conversion).

## Components

| Component | Description |
| --------- | ----------- |
| [detection_result.py](./detection_result.py) | `DetectionResult`: boxes, confidences and matched query ids for one image with their `Prompt`; class ids derive from the queries. Filtering, sorting and (class-wise) NMS. |
| [detection.py](./detection.py) | `Detection`: one object (`geometry.Box2D`, confidence, class id, class name, matched query), yielded by iterating a result. |
| [image_size.py](./image_size.py) | `ImageSize`: validated image width and height. |
| [raw_detections.py](./raw_detections.py) | `RawDetections`: per-box boxes, confidences and best query ids of a mini-batch before post-processing, as NumPy arrays independent of the runtime. |

## Examples

```python
result = detector.detect(image, Prompt.from_class_names(("person", "dog")))

confident = result.filter_by_confidence(0.5)
print(confident.boxes.area, confident.boxes.center)

for detection in result.sort_by_confidence():
    print(detection.class_name, detection.matched_query, detection.confidence, detection.box.value)
```
