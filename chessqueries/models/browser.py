"""Pieces shared by the browser exporters, scripts/export_onnx.py and scripts/export_litert.py.

Imports only torch and timm: the LiteRT exporter runs outside the project environment,
without the data and config packages."""

from __future__ import annotations

from typing import Any

import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import resample_abs_pos_embed
from torch import nn


class RGBANormalize(nn.Module):
    """Canvas pixels to model input: RGBA [0, 255] NHWC (``ImageData.data``) in,
    ImageNet-normalized RGB NCHW float32 out.

    The input is cast to float before anything else, so an integer input feeds only a CAST,
    the one op whose input type export_litert's set_input_uint8 can change."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", 255.0 * torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", 255.0 * torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = image.float()[..., :3].permute(0, 3, 1, 2)
        return (x - self.mean) / self.std


def bake_pos_embed(encoder: Any, resolution: int) -> int:
    """Resample a timm encoder's position embedding to the export resolution up front, and
    return the patch grid size.

    timm resamples it on every forward with antialiased bicubic interpolation, which neither
    ONNX nor LiteRT has; at the baked size the resample is skipped. Encoders with only RoPE
    (DINOv3) have no position embedding, and just get the new grid size."""
    patch_embed = encoder.patch_embed
    grid = resolution // patch_embed.patch_size[0]
    if getattr(encoder, "pos_embed", None) is not None:
        encoder.pos_embed = nn.Parameter(
            resample_abs_pos_embed(
                encoder.pos_embed,
                new_size=[grid, grid],
                old_size=list(patch_embed.grid_size),
                num_prefix_tokens=0 if encoder.no_embed_class else encoder.num_prefix_tokens,
            ),
            requires_grad=False,
        )
    patch_embed.grid_size = (grid, grid)
    return grid
