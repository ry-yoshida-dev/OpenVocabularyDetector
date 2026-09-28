from dataclasses import fields
from pathlib import Path

import pytest
import yaml

import open_vocabulary_detector
from open_vocabulary_detector import DetectionThresholds, DetectorBackend, DetectorSettings, Device, PromptKind

PRESET_DIRECTORY: Path = Path(open_vocabulary_detector.__file__).parent / "config"


def load_preset_section(path: Path) -> dict[str, object]:
    document: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    section: object = document["detector"]
    assert isinstance(section, dict)
    return {str(key): value for key, value in section.items()}


def test_every_backend_has_presets() -> None:
    preset_backends: set[str] = {path.parent.name for path in PRESET_DIRECTORY.glob("*/*.yaml")}
    assert preset_backends == {backend.value for backend in DetectorBackend}


@pytest.mark.parametrize("path", sorted(PRESET_DIRECTORY.glob("*/*.yaml")), ids=lambda path: path.stem)
def test_preset_matches_settings_schema(path: Path) -> None:
    section: dict[str, object] = load_preset_section(path)
    assert set(section) == {field.name for field in fields(DetectorSettings)}
    assert section["backend"] == path.parent.name
    assert section["device"] in {device.value for device in Device}
    thresholds: object = section["thresholds"]
    assert isinstance(thresholds, dict)
    assert set(thresholds) == {field.name for field in fields(DetectionThresholds)}


def test_backend_capabilities() -> None:
    text_and_visual: frozenset[PromptKind] = frozenset({PromptKind.TEXT, PromptKind.VISUAL})
    assert DetectorBackend.GROUNDING_DINO.supported_prompt_kinds == frozenset({PromptKind.TEXT})
    assert DetectorBackend.OWL_VIT.supported_prompt_kinds == text_and_visual
    assert DetectorBackend.OMDET_TURBO.supported_prompt_kinds == frozenset({PromptKind.TEXT})
    assert DetectorBackend.FLORENCE2.supported_prompt_kinds == frozenset({PromptKind.TEXT})
    assert DetectorBackend.YOLOE.supported_prompt_kinds == text_and_visual
    assert DetectorBackend.YOLO_WORLD.supported_prompt_kinds == frozenset({PromptKind.TEXT})


def test_invalid_settings_raise() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        DetectorSettings(
            backend=DetectorBackend.YOLOE,
            weights_path="yoloe-26s-seg.pt",
            thresholds=DetectionThresholds(confidence_threshold=0.25, nms_iou_threshold=0.7),
            batch_size=0,
        )
    with pytest.raises(ValueError, match="confidence_threshold"):
        DetectionThresholds(confidence_threshold=1.5, nms_iou_threshold=None)
    with pytest.raises(ValueError, match="weights_path"):
        DetectorSettings(
            backend=DetectorBackend.YOLOE,
            weights_path=" ",
            thresholds=DetectionThresholds(confidence_threshold=0.25, nms_iou_threshold=None),
        )
