from pathlib import Path

import pytest
import torch
from transformers import MMGroundingDinoConfig

from open_vocabulary_detector.backends.grounding_dino import MMGroundingDinoWithSeparateBoxHeads

FIRST_BOX_HEAD_WEIGHT: str = "bbox_embed.0.layers.0.weight"
SECOND_BOX_HEAD_WEIGHT: str = "bbox_embed.1.layers.0.weight"


def build_tiny_config(is_box_head_shared: bool) -> MMGroundingDinoConfig:
    return MMGroundingDinoConfig.from_dict(
        {
            "backbone_config": {
                "model_type": "swin",
                "embed_dim": 8,
                "depths": [1, 1, 1, 1],
                "num_heads": [1, 1, 1, 1],
                "out_features": ["stage2", "stage3", "stage4"],
            },
            "text_config": {
                "model_type": "bert",
                "hidden_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "intermediate_size": 32,
            },
            "d_model": 32,
            "encoder_layers": 1,
            "decoder_layers": 2,
            "encoder_ffn_dim": 32,
            "decoder_ffn_dim": 32,
            "encoder_attention_heads": 2,
            "decoder_attention_heads": 2,
            "num_queries": 4,
            "decoder_bbox_embed_share": is_box_head_shared,
        }
    )


@pytest.mark.parametrize("is_box_head_shared", [False, True])
def test_box_heads_follow_checkpoint_sharing(tmp_path: Path, is_box_head_shared: bool) -> None:
    source: MMGroundingDinoWithSeparateBoxHeads = MMGroundingDinoWithSeparateBoxHeads(
        build_tiny_config(is_box_head_shared)
    )
    with torch.no_grad():
        source.get_parameter(SECOND_BOX_HEAD_WEIGHT).add_(1.0)
    source.save_pretrained(tmp_path)
    loaded: MMGroundingDinoWithSeparateBoxHeads = MMGroundingDinoWithSeparateBoxHeads.from_pretrained(tmp_path)
    first_weight: torch.Tensor = loaded.get_parameter(FIRST_BOX_HEAD_WEIGHT)
    second_weight: torch.Tensor = loaded.get_parameter(SECOND_BOX_HEAD_WEIGHT)
    assert (first_weight.data_ptr() == second_weight.data_ptr()) is is_box_head_shared
    assert torch.equal(first_weight, second_weight) is is_box_head_shared
