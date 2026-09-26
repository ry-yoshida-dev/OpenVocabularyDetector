"""
Detect a fixed set of classes in every image of a directory.

Usage
-----
python examples/detect_directory.py IMAGE_DIR --backend yoloe --weights yoloe-26s-seg.pt --classes person bus
"""

import argparse
from pathlib import Path

from PIL import Image

from open_vocabulary_detector import (
    DetectionResult,
    DetectionThresholds,
    Device,
    OpenVocabularyDetector,
    OVDBackend,
    OVDSettings,
    Prompt,
    TextPrompt,
)

IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def main() -> None:
    """
    Parse arguments, run detection and print the detections of each image.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_directory", type=Path)
    parser.add_argument("--backend", type=OVDBackend, choices=list(OVDBackend), required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou-threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=Device, choices=list(Device), default=Device.AUTO)
    parser.add_argument("--half", action="store_true")
    arguments: argparse.Namespace = parser.parse_args()

    image_directory: Path = arguments.image_directory
    image_paths: list[Path] = sorted(
        path for path in image_directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"no images found in {image_directory}")

    detector: OpenVocabularyDetector = OVDSettings(
        backend=arguments.backend,
        weights_path=arguments.weights,
        thresholds=DetectionThresholds(
            confidence_threshold=arguments.confidence_threshold, nms_iou_threshold=arguments.nms_iou_threshold
        ),
        batch_size=arguments.batch_size,
        device=arguments.device,
        is_half_precision_enabled=arguments.half,
    ).build()
    class_names: list[str] = arguments.classes
    prompt: Prompt = TextPrompt(class_names=tuple(class_names))

    images: list[Image.Image] = [Image.open(path) for path in image_paths]
    results: list[DetectionResult] = detector.detect_images(images, prompt)

    for path, result in zip(image_paths, results, strict=True):
        print(f"{path.name}: {len(result)} detections")
        for detection in result:
            print(f"  {detection.class_name:20s} {detection.confidence:.3f} {detection.box.value.round(1).tolist()}")


if __name__ == "__main__":
    main()
