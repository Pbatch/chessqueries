"""Export a ChessQueries safetensors checkpoint to a browser-ready ONNX model.

The graph takes raw canvas pixels so the browser does no preprocessing beyond the
resize: input ``image`` is float32 ``(1, R, R, 4)`` RGBA in [0, 255] (the layout of
``ImageData.data``), and the graph drops alpha, applies ImageNet normalization and
transposes to NCHW. Output ``probs`` is float32 ``(1, 64, 13)`` softmax
probabilities, squares in FEN order (a8..h1), classes in ``Piece`` order.

Weights are always stored as FP16. ``--compute fp16`` runs the model in FP16 (WebGPU
needs the ``shader-f16`` feature); ``--compute fp32`` casts the weights back to FP32
in the graph, which onnxruntime folds away at load, for GPUs without ``shader-f16``.

With --check-images it then runs the exported model on onnxruntime's CUDA provider
and compares its argmax against PyTorch on the same inputs.

Usage:
    # onnxruntime-gpu 1.22 is the CUDA 12 build, matching the project's torch wheels
    uv run --no-project --python .venv/bin/python --with onnx --with onnxruntime-gpu==1.22.0 \
        scripts/export_onnx.py checkpoints/lite-ft-chesscam/lite-ft-chesscam-last.safetensors \
        ../CameraChessWeb/public/chessqueries_lite_fp16.onnx \
        --check-images 'data/chesscam/images/test/*.jpg'
    # Add --compute fp32 (and write chessqueries_lite_fp32.onnx) for the shader-f16-free model
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.io import ImageReadMode, read_image

from chessqueries.models.browser import RGBANormalize, bake_pos_embed
from chessqueries.models.checkpoint import load_safetensors_model, read_safetensors_metadata


class BrowserModel(nn.Module):
    """RGBA [0, 255] NHWC in, per-square softmax out; preprocessing lives in the graph."""

    def __init__(self, model: nn.Module, dtype: torch.dtype) -> None:
        super().__init__()
        self.model = model
        self.dtype = dtype
        self.normalize = RGBANormalize()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.normalize(image).to(self.dtype)
        return F.softmax(self.model(x).float(), dim=-1)


def store_weights_fp16(path: Path) -> None:
    """Store an FP32 graph's float weights as FP16, each cast back to FP32 in the graph."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(path))
    graph = model.graph
    casts = []
    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            continue
        name = init.name
        half = numpy_helper.from_array(
            numpy_helper.to_array(init).astype(np.float16), f"{name}_fp16"
        )
        init.CopyFrom(half)
        casts.append(helper.make_node("Cast", [half.name], [name], to=TensorProto.FLOAT))
    nodes = casts + list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes)
    onnx.save(model, str(path))


def load_rgba(path: str, resolution: int) -> torch.Tensor:
    """(1, R, R, 4) float32 RGBA, square-resized like the browser's canvas drawImage."""
    rgb = read_image(path, mode=ImageReadMode.RGB).float()
    rgb = F.interpolate(
        rgb[None],
        size=(resolution, resolution),
        mode="bilinear",
        antialias=True,
        align_corners=False,
    )
    alpha = torch.full_like(rgb[:, :1], 255.0)
    return torch.cat([rgb, alpha], dim=1).permute(0, 2, 3, 1).contiguous()


def check(model: nn.Module, onnx_path: Path, image_glob: str, resolution: int, limit: int) -> None:
    import onnxruntime as ort

    ort.preload_dlls()
    session = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider"])
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise SystemExit(f"CUDAExecutionProvider not active: {session.get_providers()}")
    paths = sorted(glob.glob(image_glob))[:limit]
    if not paths:
        raise SystemExit(f"no images match {image_glob!r}")
    square_agree = board_agree = 0
    max_diff = 0.0
    for path in paths:
        image = load_rgba(path, resolution)
        with torch.no_grad():
            expected = model(image.cuda()).cpu()
        actual = torch.from_numpy(session.run(None, {"image": image.numpy()})[0])
        same = expected.argmax(-1) == actual.argmax(-1)
        square_agree += int(same.sum())
        board_agree += int(same.all())
        max_diff = max(max_diff, float((expected - actual).abs().max()))
    n = len(paths)
    print(
        f"{n} images: square agreement {square_agree / (64 * n):.4%}, "
        f"board agreement {board_agree / n:.2%}, max |prob diff| {max_diff:.4f}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--compute", choices=["fp16", "fp32"], default="fp16")
    p.add_argument(
        "--check-images", default=None, help="glob of images to compare ONNX vs PyTorch on"
    )
    p.add_argument("--check-limit", type=int, default=200)
    args = p.parse_args()

    resolution = int(read_safetensors_metadata(args.checkpoint)["resolution"])
    recognizer = load_safetensors_model(args.checkpoint, device="cuda")
    bake_pos_embed(recognizer.encoder, resolution)
    dtype = torch.float16 if args.compute == "fp16" else torch.float32
    model = BrowserModel(recognizer.to(dtype), dtype).cuda().eval()
    dummy = torch.zeros(1, resolution, resolution, 4, device="cuda")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # The decoder's fused MultiheadAttention kernel has no ONNX op
    torch.backends.mha.set_fastpath_enabled(False)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            str(args.output),
            dynamo=False,
            opset_version=args.opset,
            input_names=["image"],
            output_names=["probs"],
            do_constant_folding=True,
        )
    if args.compute == "fp32":
        store_weights_fp16(args.output)
    print(
        f"-> {args.output} ({args.output.stat().st_size / 1e6:.1f} MB, {resolution}px, {args.compute} compute)"
    )

    if args.check_images:
        check(model, args.output, args.check_images, resolution, args.check_limit)


if __name__ == "__main__":
    main()
