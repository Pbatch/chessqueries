"""GPU batch augmentation: the warp maths, and that it matches the CPU presets it replaces."""

from unittest.mock import Mock

import pytest
import torch

from chessqueries.data.gpu_augment import (
    GpuAugment,
    _homography,
    colour_jitter,
    sample_homographies,
    square_uint8,
    warp,
)
from chessqueries.data.transforms import Augment, build_transform
from chessqueries.train.lit import JointDataModule


def _image(size: int = 96, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randint(0, 256, (3, size, size), dtype=torch.uint8)


def test_homography_maps_the_four_points() -> None:
    src = torch.tensor([[[0.0, 0], [99, 0], [99, 79], [0, 79]]])
    dst = torch.tensor([[[5.0, 3], [90, 8], [97, 70], [2, 75]]])
    h = _homography(src, dst)
    mapped = torch.cat([src, torch.ones(1, 4, 1)], dim=2) @ h.transpose(1, 2)
    assert torch.allclose(mapped[..., :2] / mapped[..., 2:], dst, atol=1e-3)


def test_identity_warp_keeps_the_image() -> None:
    x = _image().float()[None] / 255
    assert torch.allclose(warp(x, torch.eye(3)[None]), x, atol=1e-5)


def test_eval_only_normalises_like_the_cpu_pipeline() -> None:
    raw = _image(200)
    gpu = GpuAugment(Augment.GEOMETRIC_MILD)(square_uint8(96)(raw)[None], train=False)[0]
    cpu = build_transform(96, train=False)(raw.float())
    # The only difference is resizing uint8 rather than float
    assert (gpu - cpu).abs().mean() < 0.02


def test_none_preset_does_not_augment_in_training() -> None:
    x = _image()[None]
    aug = GpuAugment(Augment.NONE)
    assert torch.equal(aug(x, train=True), aug(x, train=False))


def _black_share(images: torch.Tensor) -> float:
    return float((images.abs().sum(1) < 1e-6).float().mean())


@pytest.mark.parametrize("preset", [Augment.GEOMETRIC_MILD, Augment.GEOMETRIC])
def test_geometric_coverage_matches_the_cpu_preset(preset: Augment) -> None:
    """Over many draws, the share of black fill should match torchvision's version"""
    torch.manual_seed(0)
    white = torch.ones(400, 3, 64, 64)
    gpu = _black_share(warp(white, sample_homographies(400, 64, 64, preset, "cpu")))
    cpu_tf = build_transform(64, train=True, augment=preset).transforms
    geometric = [
        t
        for t in cpu_tf
        if type(t).__name__ in {"RandomApply", "RandomPerspective", "RandomAffine"}
    ]
    cpu_images = []
    for image in white:
        for t in geometric:
            image = t(image)
        cpu_images.append(image)
    cpu = _black_share(torch.stack(cpu_images))
    assert cpu > 0.02
    assert abs(gpu - cpu) / cpu < 0.2, (gpu, cpu)


def test_colour_jitter_stays_in_range_and_varies() -> None:
    torch.manual_seed(0)
    x = torch.rand(64, 3, 16, 16)
    out = colour_jitter(x)
    assert out.min() >= 0 and out.max() <= 1
    # Brightness alone moves the mean by up to 20%
    ratio = out.mean(dim=(1, 2, 3)) / x.mean(dim=(1, 2, 3))
    assert ratio.min() < 0.95 and ratio.max() > 1.05


def test_datamodule_hook_augments_only_in_training() -> None:
    dm = JointDataModule(
        (), resolution=32, batch_size=2, workers=0, augment=Augment.GEOMETRIC_MILD, gpu_augment=True
    )
    x = torch.full((2, 3, 32, 32), 128, dtype=torch.uint8)
    y = torch.zeros(2, 64, dtype=torch.long)
    dm.trainer = Mock(training=False)
    evaluated, labels = dm.on_after_batch_transfer((x, y), 0)
    assert evaluated.dtype == torch.float32 and labels is y
    assert torch.allclose(evaluated[0], evaluated[1])
    torch.manual_seed(0)
    dm.trainer = Mock(training=True)
    trained, _ = dm.on_after_batch_transfer((x, y), 0)
    assert not torch.allclose(trained, evaluated)
