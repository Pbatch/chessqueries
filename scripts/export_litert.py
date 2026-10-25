"""Export a ChessQueries safetensors checkpoint to a LiteRT (.tflite) model for LiteRT.js.

Same input/output contract as scripts/export_onnx.py: input float32 ``(1, R, R, 4)``
RGBA in [0, 255]; output float32 ``(1, 64, 13)`` softmax probabilities, squares in FEN
order (a8..h1). Weights are stored as FP16.

Attention is rewritten (numerically identical) so the whole graph runs on the LiteRT
WebGPU delegate, which rejects tensors above 4D and matmuls of two constants or with a
constant first input. The rewrite is checked against the original model before export.

litert-torch pins its own torch, so this runs outside the project venv:
    uv run --no-project --python 3.12 --with litert-torch==0.9.4 --with timm==1.0.27 \
        --with safetensors python scripts/export_litert.py \
        checkpoints/lite-ft-chesscam/lite-ft-chesscam-last.safetensors \
        ../CameraChessWeb/public/chessqueries_lite.tflite
"""
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from timm.layers import resample_abs_pos_embed
from torch import nn
from torch.nn import functional as F

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
# Skip chessqueries.models.__init__, which pulls in the LLM baselines and their dependencies
models_pkg = types.ModuleType("chessqueries.models")
models_pkg.__path__ = [str(repo / "chessqueries" / "models")]
sys.modules["chessqueries.models"] = models_pkg
from chessqueries.models.checkpoint import load_safetensors_model, read_safetensors_metadata  # noqa: E402

import litert_torch  # noqa: E402
from ai_edge_litert.interpreter import Interpreter  # noqa: E402
from litert_torch.generative.quantize import quant_recipes  # noqa: E402

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class BrowserModel(nn.Module):
    """Same contract as scripts/export_onnx.py: RGBA [0, 255] NHWC in, per-square softmax out."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", 255.0 * torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", 255.0 * torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = image[..., :3].permute(0, 3, 1, 2)
        x = (x - self.mean) / self.std
        return F.softmax(self.model(x), dim=-1)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, scale: float) -> torch.Tensor:
    """Multi-head attention on (B, N, C) inputs, one channel slice per head.

    Avoids the reshape/permute head split of timm and PyTorch: the LiteRT WebGPU delegate
    rejects their 5D tensors, and scrambles tokens when it fuses a 4D permute chain."""
    c = q.shape[-1]
    d = c // heads
    outs = []
    for i in range(heads):
        s = slice(i * d, (i + 1) * d)
        scores = torch.bmm(q[..., s] * scale, k[..., s].transpose(1, 2))
        outs.append(torch.bmm(torch.softmax(scores, dim=-1), v[..., s]))
    return torch.cat(outs, dim=-1)


class EncoderAttention(nn.Module):
    """Drop-in for timm's Attention (DINOv2: fused qkv, no qk-norm)."""

    def __init__(self, attn: nn.Module) -> None:
        super().__init__()
        assert isinstance(attn.q_norm, nn.Identity) and isinstance(attn.k_norm, nn.Identity)
        self.qkv, self.proj = attn.qkv, attn.proj
        self.heads, self.scale = attn.num_heads, attn.scale

    def forward(self, x: torch.Tensor, attn_mask=None, is_causal=False) -> torch.Tensor:
        c = x.shape[-1]
        qkv = self.qkv(x)
        out = attention(qkv[..., :c], qkv[..., c:2 * c], qkv[..., 2 * c:], self.heads, self.scale)
        return self.proj(out)


class DecoderAttention(nn.Module):
    """Drop-in for a batch_first nn.MultiheadAttention without masks."""

    batch_first = True

    def __init__(self, mha: nn.MultiheadAttention) -> None:
        super().__init__()
        assert mha.batch_first and mha._qkv_same_embed_dim
        c = mha.embed_dim
        self.q, self.k, self.v = (nn.Linear(c, c) for _ in range(3))
        for i, lin in enumerate((self.q, self.k, self.v)):
            lin.weight = nn.Parameter(mha.in_proj_weight[i * c:(i + 1) * c].detach())
            lin.bias = nn.Parameter(mha.in_proj_bias[i * c:(i + 1) * c].detach())
        self.out_proj = mha.out_proj
        self.heads = mha.num_heads
        self.scale = (c // mha.num_heads) ** -0.5

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False,
                is_causal=False):
        assert attn_mask is None and key_padding_mask is None
        out = attention(self.q(query), self.k(key), self.v(value), self.heads, self.scale)
        return self.out_proj(out), None


class PrecomputedFirstLayer(nn.Module):
    """The first decoder layer with its self-attention block precomputed.

    Its input is the learned square queries, so self-attention + norm1 is a constant;
    left in the graph, it is a matmul of two constants, which the GPU delegate rejects."""

    def __init__(self, layer: nn.TransformerDecoderLayer, queries: torch.Tensor) -> None:
        super().__init__()
        assert not layer.norm_first
        self.layer = layer
        self.self_attn = layer.self_attn  # TransformerDecoder reads .self_attn.batch_first
        mha = layer.multihead_attn
        with torch.no_grad():
            x = layer.norm1(queries + layer._sa_block(queries, None, None))
            self.register_buffer("x", x)
            # Cross-attention queries are constant too: keep them as the matmul's second
            # input (scores^T = K Q^T), the only constant position the delegate accepts
            n, c = x.shape[1], x.shape[2]
            d = c // mha.heads
            q = (mha.q(x) * mha.scale).reshape(n, mha.heads, d).permute(1, 2, 0)
            self.register_buffer("q_t", q.contiguous())  # (heads, d, n)

    def cross_attention(self, memory: torch.Tensor) -> torch.Tensor:
        mha = self.layer.multihead_attn
        m, c = memory.shape[1], memory.shape[2]
        heads, d, n = self.q_t.shape
        k = mha.k(memory).reshape(m, heads, d).permute(1, 0, 2)  # (heads, m, d)
        v = mha.v(memory).reshape(m, heads, d).permute(1, 0, 2)
        attn = torch.softmax(torch.bmm(k, self.q_t), dim=1).transpose(1, 2)  # (heads, n, m)
        out = torch.bmm(attn, v).permute(1, 0, 2).reshape(1, n, c)
        return mha.out_proj(out)

    def forward(self, tgt, memory, **kwargs):
        layer = self.layer
        x = layer.norm2(self.x + self.cross_attention(memory))
        return layer.norm3(x + layer._ff_block(x))


def rewrite_attention(recognizer: nn.Module) -> None:
    for block in recognizer.encoder.blocks:
        block.attn = EncoderAttention(block.attn)
    for layer in recognizer.decoder.layers:
        layer.self_attn = DecoderAttention(layer.self_attn)
        layer.multihead_attn = DecoderAttention(layer.multihead_attn)
    layers = recognizer.decoder.layers
    layers[0] = PrecomputedFirstLayer(layers[0], recognizer.queries(1))


def build_model(ckpt: Path) -> tuple[BrowserModel, int]:
    """The browser wrapper around the checkpoint, attention rewritten for the GPU delegate."""
    resolution = int(read_safetensors_metadata(ckpt)["resolution"])
    recognizer = load_safetensors_model(ckpt, device="cpu").float()
    encoder = recognizer.encoder
    grid = resolution // encoder.patch_embed.patch_size[0]
    encoder.pos_embed = nn.Parameter(resample_abs_pos_embed(
        encoder.pos_embed, new_size=(grid, grid), old_size=encoder.patch_embed.grid_size,
        num_prefix_tokens=0 if encoder.no_embed_class else encoder.num_prefix_tokens,
    ), requires_grad=False)
    encoder.patch_embed.grid_size = (grid, grid)
    torch.backends.mha.set_fastpath_enabled(False)
    model = BrowserModel(recognizer).eval()

    sample = torch.rand(1, resolution, resolution, 4) * 255
    with torch.no_grad():
        reference = model(sample).numpy()
        rewrite_attention(recognizer)
        rewritten = model(sample).numpy()
    print("attention rewrite: max |prob diff|", float(np.abs(reference - rewritten).max()))
    return model, resolution


def main() -> None:
    ckpt, out = Path(sys.argv[1]), Path(sys.argv[2])
    model, resolution = build_model(ckpt)
    sample = (torch.rand(1, resolution, resolution, 4) * 255,)
    t = time.time()
    edge = litert_torch.convert(model, sample, quant_config=quant_recipes.full_fp16_recipe())
    edge.export(str(out))
    print(f"-> {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t:.0f}s")

    with torch.no_grad():
        expected = model(*sample).numpy()
    interpreter = Interpreter(model_path=str(out))
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], sample[0].numpy())
    interpreter.invoke()
    actual = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    print("max |prob diff| vs torch:", float(np.abs(expected - actual).max()),
          "argmax agree:", float((expected.argmax(-1) == actual.argmax(-1)).mean()))


if __name__ == "__main__":
    main()
