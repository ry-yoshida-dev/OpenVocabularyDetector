# runtime

## Overview

How a detector's model is executed. `OpenVocabularyDetector` itself is independent of any inference framework;
each detector holds the runtime its model needs. Every current backend runs on PyTorch through `TorchRuntime`;
an ONNX Runtime or TensorRT runtime would be added here and handed its model outputs to the shared
post-processing as `RawDetections`.

## Components

| Component | Description |
| --------- | ----------- |
| [pytorch.py](./pytorch.py) | `TorchRuntime`: resolves `Device` to a `torch.device`, picks the dtype, and prepares models and inputs. |
| [class_prediction.py](./class_prediction.py) | `ClassPrediction`: best confidence and class id of every query box from PyTorch outputs, converted to `RawDetections`. |

## Examples

```python
runtime = TorchRuntime(settings)
runtime.prepare_model(model)
pixel_values = runtime.to_model_input(pixel_values)
```
