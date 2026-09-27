# examples

## Overview

Runnable scripts using the package.

## Components

| Component | Description |
| --------- | ----------- |
| [detect_directory.py](./detect_directory.py) | Detects classes in every image of a directory with any backend and weights; `name:query,query` gives a class several text queries, and a repeated class collects the queries of every argument. |

## Examples

```bash
python examples/detect_directory.py images/ --backend yoloe --weights yoloe-26s-seg.pt --classes person bicycle
python examples/detect_directory.py images/ --backend grounding_dino --weights IDEA-Research/grounding-dino-tiny \
    --classes "man riding a bicycle" bicycle "car:car,suv,van,taxi"
```
