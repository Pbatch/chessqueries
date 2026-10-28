"""Training augmentation on the GPU, a batch at a time.

The CPU presets in `transforms` make four float32 passes over every 644px image in a loader
worker, which caps a few CPUs at ~150 img/s, below what an A100 trains ViT-S at. Here workers only
decode and resize (`square_uint8`), and `GpuAugment` does the rest on the GPU: the rotation,
perspective and affine of a geometric preset compose into one homography per image, applied with
a single bilinear `grid_sample`, then colour jitter and normalisation run as batch tensor ops.

The random draws match the CPU presets (same probabilities, ranges and perspective corner
jitter, black fill), with two deliberate differences: one bilinear resample instead of three
(two of them nearest), and the colour jitter order is drawn per batch rather than per image.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from chessqueries.data.transforms import _GEOMETRIC_PRESETS, IMAGENET_MEAN, IMAGENET_STD, Augment

ROTATION_P = 0.9
PERSPECTIVE_P = 0.7
TRANSLATE = 0.08
JITTER = 0.2  # brightness, contrast and saturation factors are drawn from [1 - JITTER, 1 + JITTER]
GRAY = (0.299, 0.587, 0.114)


def square_uint8(resolution: int) -> transforms.Resize:
    """Worker-side transform for GPU augmentation: a plain (stretching) resize, kept uint8"""
    return transforms.Resize((resolution, resolution), antialias=True)


def _about_centre(m: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Conjugate (B, 3, 3) transforms so they act about the image centre"""
    cx, cy = (width - 1) / 2, (height - 1) / 2
    to_centre = torch.tensor([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=m.dtype, device=m.device)
    back = torch.tensor([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=m.dtype, device=m.device)
    return back @ m @ to_centre


def _homography(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """(B, 3, 3) maps taking each set of 4 `src` points to `dst`, both (B, 4, 2)"""
    batch = src.shape[0]
    a = src.new_zeros(batch, 8, 8)
    x, y, u, v = src[..., 0], src[..., 1], dst[..., 0], dst[..., 1]
    a[:, 0::2, 0], a[:, 0::2, 1], a[:, 0::2, 2] = x, y, 1
    a[:, 0::2, 6], a[:, 0::2, 7] = -u * x, -u * y
    a[:, 1::2, 3], a[:, 1::2, 4], a[:, 1::2, 5] = x, y, 1
    a[:, 1::2, 6], a[:, 1::2, 7] = -v * x, -v * y
    b = torch.stack([u, v], dim=2).reshape(batch, 8)
    h = torch.linalg.solve(a, b)
    return torch.cat([h, h.new_ones(batch, 1)], dim=1).reshape(batch, 3, 3)


def sample_homographies(
    batch: int, width: int, height: int, preset: Augment, device
) -> torch.Tensor:
    """(B, 3, 3) forward maps, input pixel -> output pixel: rotation, then perspective, then affine"""
    params = _GEOMETRIC_PRESETS[preset]
    eye = torch.eye(3, device=device).expand(batch, 3, 3)

    def chance(p: float) -> torch.Tensor:
        return (torch.rand(batch, device=device) < p)[:, None, None]

    # Rotation by up to +-degrees, applied with probability ROTATION_P
    angle = torch.deg2rad((torch.rand(batch, device=device) * 2 - 1) * params["rotation"])
    cos, sin = torch.cos(angle), torch.sin(angle)
    rotation = torch.zeros(batch, 3, 3, device=device)
    rotation[:, 0, 0], rotation[:, 0, 1], rotation[:, 1, 0], rotation[:, 1, 1] = cos, -sin, sin, cos
    rotation[:, 2, 2] = 1
    rotation = torch.where(chance(ROTATION_P), _about_centre(rotation, width, height), eye)

    # Perspective: each corner pulled inwards by up to distortion * half the side, as torchvision draws it
    reach = torch.tensor(
        [params["perspective"] * (width // 2), params["perspective"] * (height // 2)], device=device
    )
    corners = torch.tensor(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=torch.float32,
        device=device,
    ).expand(batch, 4, 2)
    inward = torch.tensor([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=torch.float32, device=device)
    moved = corners + inward * torch.rand(batch, 4, 2, device=device) * reach
    perspective = torch.where(chance(PERSPECTIVE_P), _homography(corners, moved), eye)

    # Affine: shift by up to TRANSLATE of each side, scale within the preset's range, always applied
    low, high = params["scale"]
    scale = low + torch.rand(batch, device=device) * (high - low)
    shift = (
        (torch.rand(batch, 2, device=device) * 2 - 1)
        * TRANSLATE
        * torch.tensor([width, height], device=device)
    )
    affine = torch.zeros(batch, 3, 3, device=device)
    affine[:, 0, 0], affine[:, 1, 1], affine[:, 2, 2] = scale, scale, 1
    affine = _about_centre(affine, width, height)
    affine[:, :2, 2] += shift

    return affine @ perspective @ rotation


def warp(images: torch.Tensor, homographies: torch.Tensor) -> torch.Tensor:
    """Apply forward pixel maps to (B, C, H, W) images; pixels from outside the image are black"""
    batch, _, height, width = images.shape
    ys, xs = torch.meshgrid(
        torch.arange(height, device=images.device, dtype=torch.float32),
        torch.arange(width, device=images.device, dtype=torch.float32),
        indexing="ij",
    )
    out = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1).reshape(1, -1, 3)  # (1, H*W, 3)
    # Each output pixel samples the input where the inverse map sends it
    src = out @ torch.linalg.inv(homographies).transpose(1, 2)
    src = src[..., :2] / src[..., 2:].clamp_min(1e-8)
    grid = torch.stack([src[..., 0] / (width - 1), src[..., 1] / (height - 1)], dim=-1) * 2 - 1
    return F.grid_sample(
        images,
        grid.reshape(batch, height, width, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


def colour_jitter(images: torch.Tensor) -> torch.Tensor:
    """torchvision's brightness/contrast/saturation jitter on (B, 3, H, W) images in [0, 1]"""
    batch = images.shape[0]
    gray_weights = torch.tensor(GRAY, device=images.device).view(1, 3, 1, 1)

    def factor() -> torch.Tensor:
        return 1 - JITTER + torch.rand(batch, 1, 1, 1, device=images.device) * 2 * JITTER

    for step in torch.randperm(3).tolist():
        if step == 0:
            images = images * factor()
        elif step == 1:
            mean = (images * gray_weights).sum(1, keepdim=True).mean(dim=(2, 3), keepdim=True)
            images = mean + (images - mean) * factor()
        else:
            gray = (images * gray_weights).sum(1, keepdim=True)
            images = gray + (images - gray) * factor()
        images = images.clamp(0, 1)
    return images


class GpuAugment(nn.Module):
    """uint8 (B, 3, H, W) batches -> augmented (when training) and ImageNet-normalised float32"""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, augment: Augment | str = Augment.GEOMETRIC_MILD) -> None:
        super().__init__()
        self.augment = Augment(augment)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor, train: bool) -> torch.Tensor:
        x = images.float() / 255
        if train and self.augment in _GEOMETRIC_PRESETS:
            batch, _, height, width = x.shape
            x = warp(x, sample_homographies(batch, width, height, self.augment, x.device))
        if train and self.augment is not Augment.NONE:
            x = colour_jitter(x)
        return (x - self.mean.to(x.device)) / self.std.to(x.device)
