"""
Fail fast with an installation hint when the optional Ultralytics dependency is missing.

Ultralytics is licensed under AGPL-3.0, so it is not a required dependency of this package.
"""

from importlib.util import find_spec

if find_spec("ultralytics") is None:
    raise ModuleNotFoundError(
        "The YOLO-World and YOLOE backends need Ultralytics (AGPL-3.0). "
        + "Install it with: pip install 'open-vocabulary-detector[ultralytics]'",
        name="ultralytics",
    )
