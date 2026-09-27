"""
Detect a fixed set of classes in every image of a directory.

Usage
-----
python examples/detect_directory.py IMAGE_DIR --backend yoloe --weights yoloe-26s-seg.pt --classes person bus

A class written as ``name:query,query`` (e.g. ``car:car,suv,taxi``) is queried by each phrase and reported as ``name``.
"""

import argparse
from pathlib import Path

from PIL import Image

from open_vocabulary_detector import (
    DetectionResult,
    DetectionThresholds,
    DetectorBackend,
    DetectorSettings,
    Device,
    OpenVocabularyDetector,
    Prompt,
    PromptQuery,
    TextQuery,
    VisualQuery,
)

IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
CLASS_QUERY_SEPARATOR: str = ":"
QUERY_SEPARATOR: str = ","


def parse_prompt(class_arguments: list[str]) -> Prompt:
    """
    Build a text prompt from ``name`` or ``name:query,query`` arguments.

    A class given more than once collects the phrases of every argument, e.g. ``car:car,suv car:van``.

    Parameters
    ----------
    class_arguments : list[str]
        Command-line class arguments.

    Returns
    -------
    Prompt
        Classes named ``name``, each queried by its listed phrases or by its name.
    """
    class_texts: dict[str, list[str]] = {}
    for argument in class_arguments:
        class_name, _, phrases = argument.partition(CLASS_QUERY_SEPARATOR)
        stripped_name: str = class_name.strip()
        class_phrases: list[str] = phrases.split(QUERY_SEPARATOR) if phrases else [stripped_name]
        class_texts.setdefault(stripped_name, []).extend(class_phrases)
    return Prompt.from_texts(class_texts)


def describe_query(query: PromptQuery) -> str:
    """
    Short label of the query that matched a detection.

    Parameters
    ----------
    query : PromptQuery
        Matched query.

    Returns
    -------
    str
        The phrase of a text query, or the reference count of a visual query.
    """
    match query:
        case TextQuery(text=text):
            return text
        case VisualQuery(references=references):
            return f"<{len(references)} reference images>"


def main() -> None:
    """
    Parse arguments, run detection and print the detections of each image.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_directory", type=Path)
    parser.add_argument("--backend", type=DetectorBackend, choices=list(DetectorBackend), required=True)
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

    detector: OpenVocabularyDetector = DetectorSettings(
        backend=arguments.backend,
        weights_path=arguments.weights,
        thresholds=DetectionThresholds(
            confidence_threshold=arguments.confidence_threshold, nms_iou_threshold=arguments.nms_iou_threshold
        ),
        batch_size=arguments.batch_size,
        device=arguments.device,
        is_half_precision_enabled=arguments.half,
    ).build()
    prompt: Prompt = parse_prompt(arguments.classes)

    images: list[Image.Image] = [Image.open(path) for path in image_paths]
    results: list[DetectionResult] = detector.detect_images(images, prompt)

    for path, result in zip(image_paths, results, strict=True):
        print(f"{path.name}: {len(result)} detections")
        for detection in result:
            print(
                f"  {detection.class_name:20s} {describe_query(detection.matched_query):20s} {detection.confidence:.3f} "
                + f"{detection.box.value.round(1).tolist()}"
            )


if __name__ == "__main__":
    main()
