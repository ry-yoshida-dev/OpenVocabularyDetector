import json
from pathlib import Path

import pytest

from open_vocabulary_detector.backends.grounding_dino import GroundingDinoVariant
from open_vocabulary_detector.backends.owl_vit import OwlVersion


def write_checkpoint_config(directory: Path, model_type: str) -> str:
    (directory / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    return str(directory)


@pytest.mark.parametrize(("model_type", "version"), [("owlvit", OwlVersion.V1), ("owlv2", OwlVersion.V2)])
def test_owl_version_follows_model_type(tmp_path: Path, model_type: str, version: OwlVersion) -> None:
    assert OwlVersion.of_checkpoint(write_checkpoint_config(tmp_path, model_type)) is version


@pytest.mark.parametrize(
    ("model_type", "variant"),
    [("grounding-dino", GroundingDinoVariant.ORIGINAL), ("mm-grounding-dino", GroundingDinoVariant.MM)],
)
def test_grounding_dino_variant_follows_model_type(
    tmp_path: Path, model_type: str, variant: GroundingDinoVariant
) -> None:
    assert GroundingDinoVariant.of_checkpoint(write_checkpoint_config(tmp_path, model_type)) is variant


def test_checkpoint_of_another_architecture_is_rejected(tmp_path: Path) -> None:
    weights_path: str = write_checkpoint_config(tmp_path, "owlv2")
    with pytest.raises(ValueError, match="not one of"):
        GroundingDinoVariant.of_checkpoint(weights_path)
