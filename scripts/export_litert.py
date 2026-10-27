"""Export a ChessQueries safetensors checkpoint to a LiteRT (.tflite) model for LiteRT.js.

Input uint8 ``(1, R, R, 4)`` RGBA, canvas ``ImageData.data`` as is (a quarter of the
upload of float32, and no conversion in the browser); output float32 ``(1, 64, 13)``
softmax probabilities, squares in FEN order (a8..h1). Weights are stored as FP16.

litert-torch cannot convert a uint8 input, so the model is converted with an int32 input
that is cast to float first thing, and the input's type is then rewritten to uint8 in the
.tflite (see set_input_uint8). CAST from uint8 runs fully on the WebGPU delegate.

A second output, float32 ``(1, 64, 2)``, is where each square query looks: the (x, y)
centroid, in [0, 1] input coordinates, of its last-layer cross-attention over the patch
grid (averaged over heads). The centroids are pulled towards the board centre, but their
rank and file directions give the board orientation the model assumed.

The model is rewritten (numerically identical, see ChessQueries2D) so the whole graph runs
correctly on the LiteRT WebGPU delegate; the rewrite is checked against the original
model before export.

litert-torch pins its own torch, so this runs outside the project venv:
    uv run --no-project --python 3.12 --with litert-torch==0.9.4 --with timm==1.0.27 \
        --with torchao==0.17.0 --with safetensors --with tflite==2.18.0 python scripts/export_litert.py \
        checkpoints/lite-ft-chesscam-v2/lite-ft-chesscam-v2-last.safetensors \
        ../CameraChessWeb/public/chessqueries_lite.tflite
"""
import math
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
import tflite  # noqa: E402
from ai_edge_litert.interpreter import Interpreter  # noqa: E402
from litert_torch.backend.composite import StableHLOCompositeBuilder  # noqa: E402
from litert_torch.generative.quantize import quant_recipes  # noqa: E402

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
# The WebGPU delegate (LiteRT.js 2.5.3 and 2.6.0) fails to compile a model with more than 7
# attention composites ("STABLEHLO_COMPOSITE failed to prepare"), so only the first 7 encoder
# blocks use fused_attention
MAX_FUSED_ATTENTION = 7
# The delegate's fp16 tops out at 65504. LayerNorm inputs' sums of squares and attention scores
# are kept below this on the export sample, leaving 16x headroom for other images
FP16_LIMIT = 4096.0


class BrowserModel(nn.Module):
    """RGBA NHWC in (int32 here, uint8 once exported), per-square softmax out."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", 255.0 * torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", 255.0 * torch.tensor(STD).view(1, 3, 1, 1))

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        # Cast before anything else, so the input feeds only a CAST, the one op whose input
        # type set_input_uint8 can change
        x = image.float()[..., :3].permute(0, 3, 1, 2)
        return (x - self.mean) / self.std

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, centroids = self.model(self.normalize(image))
        # Batch dimension added last: the delegate mislays out 3D tensors made from linear outputs
        return F.softmax(logits, dim=-1).reshape(1, 64, -1), centroids.reshape(1, 64, 2)


class ReferenceModel(nn.Module):
    """The original model, with the centroids from its last decoder layer's attention weights."""

    def __init__(self, recognizer: nn.Module, grid_xy: torch.Tensor) -> None:
        super().__init__()
        self.recognizer = recognizer
        self.register_buffer("grid_xy", grid_xy)
        self.num_prefix_tokens = recognizer.encoder.num_prefix_tokens
        last = recognizer.decoder.layers[-1]
        self.weights: list[torch.Tensor] = []

        def mha(x, mem, attn_mask, key_padding_mask, is_causal=False):
            out, w = last.multihead_attn(x, mem, mem, need_weights=True, average_attn_weights=True)
            self.weights.append(w)
            return last.dropout2(out)
        last._mha_block = mha

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.weights.clear()
        logits = self.recognizer(x)
        w = self.weights[0][0, :, self.num_prefix_tokens:]
        return logits[0], (w @ self.grid_xy) / w.sum(-1, keepdim=True)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, scale: float,
              need_weights: bool = False, shrink: float = 1.0) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Multi-head attention on 2D (N, C) queries and (M, C) keys/values, all heads at once on
    3D (heads, tokens, d) tensors, so no tensor goes above 3D (the delegate rejects
    timm/PyTorch's 5D split). About 20% less GPU time than one matmul per head channel slice.
    With need_weights, also returns the (N, M) attention weights averaged over heads.

    With shrink, the scores are computed divided by it, for scores beyond fp16's range (DINOv3's
    first block reaches millions), then grown back after subtracting each row's max. Scores far
    below the max are clamped first, to a gap whose exp is 0 in fp16 anyway: grown back to -inf
    instead, they make the delegate's fp16 softmax NaN."""
    n, c = q.shape
    m = k.shape[0]
    d = c // heads
    qh = (q * (scale / shrink)).reshape(n, heads, d).transpose(0, 1)  # (H, N, d)
    kh = k.reshape(m, heads, d).permute(1, 2, 0)  # (H, d, M)
    vh = v.reshape(m, heads, d).transpose(0, 1)  # (H, M, d)
    scores = torch.matmul(qh, kh)  # (H, N, M)
    if shrink != 1.0:
        scores = (scores - scores.amax(-1, keepdim=True)).clamp(min=-FP16_LIMIT / shrink) * shrink
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, vh).transpose(0, 1).reshape(n, c)
    return (out, probs.mean(0)) if need_weights else out


def fused_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, scale: float) -> torch.Tensor:
    """Multi-head attention on 2D (N, C) queries and (M, C) keys/values, as one
    odml.scaled_dot_product_attention composite, which the WebGPU delegate runs as a single
    fused op without writing out the (heads, N, M) scores. About 30% less GPU time than
    attention() for an encoder block, about 15% less for the whole model with 7 fused blocks.
    With no mask the composite is not causal there, but litert-torch's
    scaled_dot_product_attention_with_hlfb treats no mask as causal, so it is built here
    around ordinary attention.

    q is scaled before the composite, which gets scale 1, to keep the fused kernel's fp16
    scores small (this is the form that was checked against the CPU in the browser).

    Caveat, seen with LiteRT.js 2.5.3: compiling a model with these composites for WebGPU
    after a CPU (wasm) model has run on the same page can fail to compile, or compile and
    give wrong boards. Compiled first on a fresh page, as the app does, it is correct."""
    n, c = q.shape
    m = k.shape[0]
    d = c // heads
    q = q * scale
    builder = StableHLOCompositeBuilder(name="odml.scaled_dot_product_attention", attr={"scale": 1.0})
    qh, kh, vh = builder.mark_inputs(q.reshape(1, n, heads, d), k.reshape(1, m, heads, d), v.reshape(1, m, heads, d))
    out = F.scaled_dot_product_attention(qh.transpose(1, 2), kh.transpose(1, 2), vh.transpose(1, 2), scale=1.0)
    return builder.mark_outputs(out.transpose(1, 2)).reshape(n, c)


def layer_scale(block: nn.Module, i: int, x: torch.Tensor, factor: float = 1.0) -> torch.Tensor:
    """A block's i-th LayerScale, times `factor`: a module in timm's VisionTransformer, a gamma in its Eva"""
    gamma = getattr(block, f"gamma_{i}", None)
    if gamma is not None:
        return (gamma * factor) * x
    out = getattr(block, f"ls{i}", nn.Identity())(x)
    return out if factor == 1.0 else out * factor


def power_of_two_above(ratio: float) -> float:
    """The smallest power of two at least `ratio`, or 1 for ratios up to 1"""
    return 2.0 ** max(0, math.ceil(math.log2(ratio))) if ratio > 0 else 1.0


class InProjection(nn.Module):
    """The q/k/v projections of an nn.MultiheadAttention as separate linear maps."""

    def __init__(self, mha: nn.MultiheadAttention) -> None:
        super().__init__()
        assert mha.batch_first and mha._qkv_same_embed_dim
        c = mha.embed_dim
        for i, name in enumerate("qkv"):
            self.register_buffer(f"{name}_weight", mha.in_proj_weight[i * c:(i + 1) * c].detach().clone())
            self.register_buffer(f"{name}_bias", mha.in_proj_bias[i * c:(i + 1) * c].detach().clone())
        self.out_proj = mha.out_proj
        self.heads = mha.num_heads
        self.scale = (c // mha.num_heads) ** -0.5

    def project(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, getattr(self, f"{name}_weight"), getattr(self, f"{name}_bias"))

    def forward(self, query: torch.Tensor, memory: torch.Tensor,
                need_weights: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        out = attention(self.project("q", query), self.project("k", memory), self.project("v", memory),
                        self.heads, self.scale, need_weights)
        if need_weights:
            return self.out_proj(out[0]), out[1]
        return self.out_proj(out)


class ChessQueries2D(nn.Module):
    """ChessQueriesModel for batch size 1 with every token tensor kept 2D, (tokens, channels).

    The LiteRT WebGPU delegate lays out a linear layer's output on 3D (1, N, C) tensors
    differently from the ops that consume it, and scrambles all but the first token.
    Numerically identical to the original (checked in build_model). Also:
    - attention is split into heads on 3D tensors instead of 5D reshapes, except in the
      first MAX_FUSED_ATTENTION encoder blocks, where it is a fused attention composite
      (see fused_attention);
    - the first decoder layer's self-attention is precomputed: its input is the learned
      square queries, so it is a matmul of two constants, which the delegate rejects;
    - that layer's cross-attention queries are constant too, so its scores are folded
      into one ordinary linear map of the memory;
    - its residual add has a constant (64, C) operand, which the delegate lays out wrongly
      (wrong boards); the add is done on a single (1, 64 * C) row instead."""

    def __init__(self, recognizer: nn.Module, grid_xy: torch.Tensor) -> None:
        super().__init__()
        self.encoder = recognizer.encoder
        # Patch centres in [0, 1] input coordinates, (patches, 2)
        self.register_buffer("grid_xy", grid_xy)
        self.norm, self.head = recognizer.norm, recognizer.head
        self.decoder_layers = recognizer.decoder.layers
        assert recognizer.decoder.norm is None and not any(l.norm_first for l in self.decoder_layers)
        for block in self.encoder.blocks:
            attn = block.attn
            assert isinstance(attn.q_norm, nn.Identity) and isinstance(attn.k_norm, nn.Identity)
            assert isinstance(getattr(attn, "norm", nn.Identity()), nn.Identity)
        self.rope = getattr(self.encoder, "rope", None) is not None
        if self.rope:
            self.register_rope_tables(grid_xy.shape[0])
        # Set by fit_fp16: the residual stream is divided by residual_scale, and the attention
        # scores of the blocks in score_shrink are computed shrunk (see attention)
        self.residual_scale = 1.0
        self.score_shrink: dict[int, float] = {}
        self.self_attn = nn.ModuleList(InProjection(l.self_attn) for l in self.decoder_layers)
        self.cross_attn = nn.ModuleList(InProjection(l.multihead_attn) for l in self.decoder_layers)

        first, cross = self.decoder_layers[0], self.cross_attn[0]
        with torch.no_grad():
            queries = recognizer.queries(1)[0]  # (64, C)
            x = first.norm1(queries + self.self_attn[0](queries, queries))
            self.register_buffer("first_x", x)
            # Each head's scores K q^T: (memory Wk^T + bk) q^T = memory (q Wk)^T + q bk
            d = x.shape[-1] // cross.heads
            q = cross.project("q", x) * cross.scale
            weights, biases = [], []
            for i in range(cross.heads):
                s = slice(i * d, (i + 1) * d)
                weights.append(q[:, s] @ cross.k_weight[s])
                biases.append(q[:, s] @ cross.k_bias[s])
            self.register_buffer("first_scores_weight", torch.cat(weights))  # (heads * 64, C)
            self.register_buffer("first_scores_bias", torch.cat(biases))

    def register_rope_tables(self, patches: int) -> None:
        """DINOv3 (timm's Eva) rotates q and k by angles that depend only on the patch position,
        so for a fixed input size the sin and cos are constant (tokens, C) tables, tiled over the
        heads. Prefix tokens are not rotated: cos 1, sin 0."""
        enc = self.encoder
        assert enc.blocks[0].attn.rotate_half and not getattr(enc, "rope_mixed", False)
        grid = int(round(patches ** 0.5))
        heads = enc.blocks[0].attn.num_heads
        with torch.no_grad():
            sin, cos = enc.rope.get_embed(shape=(grid, grid)).chunk(2, -1)  # (patches, d) each
        prefix = enc.num_prefix_tokens
        self.register_buffer("rope_sin", F.pad(sin.repeat(1, heads), (0, 0, prefix, 0)))
        self.register_buffer("rope_cos", F.pad(cos.repeat(1, heads), (0, 0, prefix, 0), value=1.0))

    def rotate(self, t: torch.Tensor, heads: int) -> torch.Tensor:
        """Rotate-half RoPE on (N, C) queries or keys, split into heads on a 3D view"""
        n, c = t.shape
        half = c // heads // 2
        th = t.reshape(n, heads, 2 * half)
        rotated = torch.cat([-th[..., half:], th[..., :half]], dim=-1)
        # As rows: the delegate lays out a constant (N, C) operand of a mul wrongly, and on a 3D
        # view with the table broadcast over the heads too
        out = t.reshape(1, -1) * self.rope_cos.reshape(1, -1) + rotated.reshape(1, -1) * self.rope_sin.reshape(1, -1)
        return out.reshape(n, c)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encoder
        if not self.rope:
            return enc.norm_pre(enc._pos_embed(enc.patch_embed(x)))[0]  # (N, C)
        # Eva: the patches come out NHWC; the CLS and register tokens go in front, unrotated
        patches = enc.patch_embed(x)
        patches = patches.reshape(-1, patches.shape[-1])
        prefix = torch.cat([enc.cls_token[0], enc.reg_token[0]])
        return enc.norm_pre(torch.cat([prefix, patches]))

    def layer_norm(self, norm: nn.LayerNorm, h: torch.Tensor) -> torch.Tensor:
        """`norm` on the shrunk residual stream, with its eps shrunk to match, so the result is unchanged"""
        if self.residual_scale == 1.0:
            return norm(h)
        return F.layer_norm(h, norm.normalized_shape, norm.weight, norm.bias, norm.eps / self.residual_scale ** 2)

    def qkv(self, block: nn.Module, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn = block.attn
        c = h.shape[-1]
        x = self.layer_norm(block.norm1, h)
        if getattr(attn, "q_bias", None) is not None:
            qkv = F.linear(x, attn.qkv.weight, torch.cat((attn.q_bias, attn.k_bias, attn.v_bias)))
        else:
            qkv = attn.qkv(x)
        q, k, v = qkv[:, :c], qkv[:, c:2 * c], qkv[:, 2 * c:]
        if self.rope:
            q, k = self.rotate(q, attn.num_heads), self.rotate(k, attn.num_heads)
        return q, k, v

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encoder
        unscaled = [i for i in range(len(enc.blocks)) if i not in self.score_shrink]
        fused = set(unscaled[:MAX_FUSED_ATTENTION])
        shrink = 1.0 / self.residual_scale
        h = self.embed(x) * shrink if self.residual_scale != 1.0 else self.embed(x)
        for i, block in enumerate(enc.blocks):
            attn = block.attn
            q, k, v = self.qkv(block, h)
            if i in fused:
                out = fused_attention(q, k, v, attn.num_heads, attn.scale)
            else:
                out = attention(q, k, v, attn.num_heads, attn.scale, shrink=self.score_shrink.get(i, 1.0))
            h = h + layer_scale(block, 1, attn.proj(out), shrink)
            h = h + layer_scale(block, 2, block.mlp(self.layer_norm(block.norm2, h)), shrink)
        return self.layer_norm(enc.norm, h)

    @torch.no_grad()
    def fit_fp16(self, x: torch.Tensor) -> None:
        """Choose residual_scale and score_shrink so that, on `x`, LayerNorm inputs' sums of squares
        and attention scores stay below FP16_LIMIT. DINOv2 needs neither; DINOv3 needs both"""
        assert self.residual_scale == 1.0 and not self.score_shrink
        enc = self.encoder

        def sum_squares(h: torch.Tensor) -> float:
            return float(((h - h.mean(-1, keepdim=True)) ** 2).sum(-1).max())

        squares, shrink = [], {}
        h = self.embed(x)
        for i, block in enumerate(enc.blocks):
            attn = block.attn
            squares.append(sum_squares(h))
            q, k, v = self.qkv(block, h)
            n, c = q.shape
            heads, d = attn.num_heads, c // attn.num_heads
            scores = (q * attn.scale).reshape(n, heads, d).transpose(0, 1) @ k.reshape(n, heads, d).permute(1, 2, 0)
            ratio = float(scores.abs().max()) / FP16_LIMIT
            if ratio > 1:
                shrink[i] = power_of_two_above(ratio)
            h = h + layer_scale(block, 1, attn.proj(attention(q, k, v, heads, attn.scale)))
            squares.append(sum_squares(h))
            h = h + layer_scale(block, 2, block.mlp(block.norm2(h)))
        squares.append(sum_squares(h))
        # Sums of squares shrink with the square of the scale
        self.residual_scale = power_of_two_above((max(squares) / FP16_LIMIT) ** 0.5)
        self.score_shrink = shrink
        print(f"fp16 ranges: max LayerNorm sum of squares {max(squares):.3g}, residual scale {self.residual_scale:g}, "
              f"score shrink by block {shrink}")

    def first_cross_attention(self, memory: torch.Tensor) -> torch.Tensor:
        cross = self.cross_attn[0]
        v = cross.project("v", memory)
        scores_t = F.linear(memory, self.first_scores_weight, self.first_scores_bias)  # (M, heads * 64)
        m = memory.shape[0]
        d = v.shape[-1] // cross.heads
        n = scores_t.shape[-1] // cross.heads
        scores = scores_t.reshape(m, cross.heads, n).permute(1, 2, 0)  # (H, 64, M)
        vh = v.reshape(m, cross.heads, d).transpose(0, 1)  # (H, M, d)
        out = torch.matmul(torch.softmax(scores, dim=-1), vh).transpose(0, 1).reshape(n, -1)
        return cross.out_proj(out)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        memory = self.encode(x)
        last = len(self.decoder_layers) - 1
        assert last > 0
        for i, layer in enumerate(self.decoder_layers):
            if i == 0:
                # As one row: the delegate lays out a constant (64, C) operand of an add wrongly
                attended = self.first_cross_attention(memory)
                h = layer.norm2((attended.reshape(1, -1) + self.first_x.reshape(1, -1)).reshape(attended.shape))
            else:
                h = layer.norm1(h + self.self_attn[i](h, h))
                if i == last:
                    out, weights = self.cross_attn[i](h, memory, need_weights=True)
                else:
                    out = self.cross_attn[i](h, memory)
                h = layer.norm2(h + out)
            h = layer.norm3(h + layer.linear2(layer.activation(layer.linear1(h))))
        # Prefix (CLS/register) tokens have no position, so they are left out and the rest renormalised
        w = weights[:, self.encoder.num_prefix_tokens:]
        centroids = torch.matmul(w, self.grid_xy) / w.sum(-1, keepdim=True)  # (64, 2)
        return self.head(self.norm(h)), centroids  # (64, 13), (64, 2)


def random_image(resolution: int) -> torch.Tensor:
    """Pixel values that fit uint8, as int32, which litert-torch can convert."""
    return torch.randint(0, 256, (1, resolution, resolution, 4), dtype=torch.int32)


def set_input_uint8(path: Path) -> None:
    """Rewrite the converted model's int32 input as uint8, in place in the flatbuffer."""
    buf = bytearray(path.read_bytes())
    model = tflite.Model.GetRootAsModel(buf, 0)
    graph = model.Subgraphs(0)
    index = int(graph.InputsAsNumpy()[0])
    consumers = []
    for i in range(graph.OperatorsLength()):
        op = graph.Operators(i)
        if index in op.InputsAsNumpy():
            code = model.OperatorCodes(op.OpcodeIndex())
            consumers.append(max(code.BuiltinCode(), code.DeprecatedBuiltinCode()))
    assert consumers == [tflite.BuiltinOperator.CAST], f"input feeds ops {consumers}, not one CAST"
    tensor = graph.Tensors(index)
    assert tensor.Type() == tflite.TensorType.INT32
    # The type field is a single byte stored in the tensor table (present, as INT32 is not the default)
    buf[tensor._tab.Pos + tensor._tab.Offset(6)] = tflite.TensorType.UINT8
    path.write_bytes(buf)


def build_model(ckpt: Path) -> tuple[BrowserModel, int]:
    """The browser wrapper around the checkpoint, attention rewritten for the GPU delegate."""
    resolution = int(read_safetensors_metadata(ckpt)["resolution"])
    recognizer = load_safetensors_model(ckpt, device="cpu").float()
    encoder = recognizer.encoder
    grid = resolution // encoder.patch_embed.patch_size[0]
    if encoder.pos_embed is not None:  # DINOv3 has only RoPE
        encoder.pos_embed = nn.Parameter(resample_abs_pos_embed(
            encoder.pos_embed, new_size=(grid, grid), old_size=encoder.patch_embed.grid_size,
            num_prefix_tokens=0 if encoder.no_embed_class else encoder.num_prefix_tokens,
        ), requires_grad=False)
    encoder.patch_embed.grid_size = (grid, grid)
    torch.backends.mha.set_fastpath_enabled(False)
    centres = (torch.arange(grid, dtype=torch.float32) + 0.5) / grid
    ys, xs = torch.meshgrid(centres, centres, indexing="ij")
    grid_xy = torch.stack([xs.flatten(), ys.flatten()], dim=-1)  # row-major, as the patch tokens
    model = BrowserModel(ChessQueries2D(recognizer, grid_xy)).eval()
    reference = BrowserModel(ReferenceModel(recognizer, grid_xy)).eval()

    sample = random_image(resolution)
    model.model.fit_fp16(model.normalize(sample))
    with torch.no_grad():
        diffs = [float((r - m).abs().max()) for r, m in zip(reference(sample), model(sample))]
    print("2D rewrite: max |prob diff| %.2e, max |centroid diff| %.2e" % tuple(diffs))
    assert max(diffs) < 1e-4
    return model, resolution


def main() -> None:
    ckpt, out = Path(sys.argv[1]), Path(sys.argv[2])
    model, resolution = build_model(ckpt)
    sample = (random_image(resolution),)
    t = time.time()
    edge = litert_torch.convert(model, sample, quant_config=quant_recipes.full_fp16_recipe())
    edge.export(str(out))
    set_input_uint8(out)
    print(f"-> {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t:.0f}s")

    with torch.no_grad():
        expected = [t.numpy() for t in model(*sample)]
    interpreter = Interpreter(model_path=str(out))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    assert input_details["dtype"] == np.uint8
    interpreter.set_tensor(input_details["index"], sample[0].numpy().astype(np.uint8))
    interpreter.invoke()
    # Output order is up to the converter, so match the outputs by shape
    actual = {tuple(d["shape"]): interpreter.get_tensor(d["index"]) for d in interpreter.get_output_details()}
    probs, centroids = actual[expected[0].shape], actual[expected[1].shape]
    print("max |prob diff| vs torch:", float(np.abs(expected[0] - probs).max()),
          "argmax agree:", float((expected[0].argmax(-1) == probs.argmax(-1)).mean()),
          "max |centroid diff|:", float(np.abs(expected[1] - centroids).max()))


if __name__ == "__main__":
    main()
