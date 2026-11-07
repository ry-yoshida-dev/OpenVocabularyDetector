# runtime

## Overview

How a detector's model is executed. Every current backend runs on PyTorch through `TorchRuntime`.

## Components

| Component | Description |
| --------- | ----------- |
| [pytorch.py](./pytorch.py) | `TorchRuntime`, which runs PyTorch models on the requested device and precision. |
| [query_prediction.py](./query_prediction.py) | `QueryPrediction`, the best query of every predicted box, converted to `RawDetections`. |

## Examples

```python
runtime = TorchRuntime(settings)
runtime.prepare_model(model)
pixel_values = runtime.to_model_input(pixel_values)
```
