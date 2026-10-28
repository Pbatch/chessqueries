"""The pieces shared by the browser exporters (chessqueries.models.browser)."""

from typing import Any

import timm
import torch

from chessqueries.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from chessqueries.models.browser import RGBANormalize, bake_pos_embed


def test_rgba_normalize_matches_training_normalization():
    rgba = torch.randint(0, 256, (1, 8, 8, 4), dtype=torch.uint8)
    rgb = rgba[..., :3].permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    expected = (rgb - mean) / std
    for image in (rgba, rgba.int(), rgba.float()):
        out = RGBANormalize()(image)
        assert out.dtype == torch.float32 and out.shape == (1, 3, 8, 8)
        torch.testing.assert_close(out, expected)


def test_bake_pos_embed_matches_timm_resampling():
    encoder: Any = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=0,
        dynamic_img_size=True,
    ).eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        expected = encoder.forward_features(x)
        assert bake_pos_embed(encoder, 128) == 8
        assert encoder.pos_embed.shape[1] == encoder.num_prefix_tokens + 64
        assert not encoder.pos_embed.requires_grad
        torch.testing.assert_close(encoder.forward_features(x), expected)


def test_bake_pos_embed_without_pos_embed():
    """RoPE-only encoders (DINOv3) just get the new grid size."""
    encoder = timm.create_model("vit_small_patch16_dinov3", pretrained=False, num_classes=0)
    assert encoder.pos_embed is None
    assert bake_pos_embed(encoder, 448) == 28
    assert encoder.patch_embed.grid_size == (28, 28)
