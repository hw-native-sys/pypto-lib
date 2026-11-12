# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""V4.1 HC coefficient generation, stream collapse, residual expansion, and final collapse."""

import math
from pathlib import Path

import pypto.language as pl
import torch
import torch.nn.functional as F
from pypto.runtime import pto_isa_include_dir

from config import FLASH as M, T_DYN
from rmsnorm import NORM_T_TILE, rms_norm_tiled


# model config
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
NORM_EPS = M.rms_norm_eps

# tiling
MIX_PAD = 32
D_TILE = 2560
PRE_T_TILE = 8
POST_WORKER_TILE = 64


_KERNELS = Path(__file__).parent / "kernels"


@pl.jit.extern(core_type="mixed", dual_aiv_dispatch=True,
               aic_source=_KERNELS / "hc_pre.cpp", aiv_source=_KERNELS / "hc_pre.cpp",
               include_dirs=(pto_isa_include_dir(), str(_KERNELS)))
def _hc_pre(
    x_hc: pl.Tensor,
    incoming_pre_mix: pl.Tensor,
    function: pl.Tensor,
    output: pl.Out[pl.Tensor],
    partials: pl.Out[pl.Tensor],
    squares: pl.Out[pl.Tensor],
    scale: pl.Tensor,
    base: pl.Tensor,
    pre_mix: pl.Out[pl.Tensor],
    post_mix: pl.Out[pl.Tensor],
    residual_mix: pl.Out[pl.Tensor],
    tokens: pl.Scalar[pl.INDEX],
    padded_rows: pl.Scalar[pl.INDEX],
):
    """Project, compute raw-input RMS, collapse, and normalize HC coefficients."""
    ...


@pl.jit.inline
def hc_pre(
    x_hc: pl.Tensor,
    incoming_pre_mix: pl.Tensor,
    function: pl.Tensor,
    scale: pl.Tensor,
    base: pl.Tensor,
    output: pl.Tensor,
    pre_mix: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
):
    """Fuse raw BF16 collapse and coefficient generation in one mixed SPMD."""
    tokens = pl.tensor.dim(x_hc, 0)
    if tokens <= 64:
        row_tile = 64
    elif tokens <= 80:
        row_tile = 80
    elif tokens <= 96:
        row_tile = 96
    elif tokens <= 128:
        row_tile = 128
    elif tokens <= 512:
        row_tile = 128
    else:
        row_tile = 256
    owners = 27
    if tokens > 128:
        owners = 8
    padded_rows = ((tokens + row_tile - 1) // row_tile) * row_tile
    partial_rows = owners * padded_rows
    partials = pl.create_tensor([partial_rows, MIX_PAD], dtype=pl.FP32)
    squares = pl.create_tensor([owners, padded_rows], dtype=pl.FP32)
    workers = pl.system.available_cluster_count()
    with pl.spmd(workers, name_hint="hc_pre", sync_start=True) as ready:
        _hc_pre(
            x_hc, incoming_pre_mix, function, output, partials, squares,
            scale, base, pre_mix, post_mix, residual_mix, tokens, padded_rows,
        )
    return ready


@pl.jit.inline(auto_scope=False)
def hc_normalize(
    collapsed: pl.Tensor,
    norm_weight: pl.Tensor,
    output: pl.Tensor,
    active_rows: pl.Scalar[pl.INT32],
):
    """Normalize rounded BF16 HC rows and clear inactive physical rows."""
    tokens = pl.tensor.dim(collapsed, 0)
    active = pl.max(pl.min(pl.cast(active_rows, pl.INDEX), tokens), 0)
    blocks = (tokens + NORM_T_TILE - 1) // NORM_T_TILE
    workers = pl.min(blocks, pl.system.available_cluster_count() * 2)
    with pl.spmd(workers, name_hint="hc_normalize") as ready:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, blocks, workers):
            rms_norm_tiled(collapsed, norm_weight, output, block, tokens, output_fp32=False)
            first = block * NORM_T_TILE
            rows = pl.min(NORM_T_TILE, tokens - first)
            for local in pl.range(rows):
                row = first + local
                if row >= active:
                    output[row : row + 1, 0:D] = pl.full([1, D], dtype=pl.BF16, value=0.0)
    return ready


@pl.jit.extern(core_type="aiv", source=_KERNELS / "hc_collapse.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _hc_collapse(
    x: pl.Tensor,
    pre_mix: pl.Tensor,
    norm_weight: pl.Tensor,
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    collapsed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    rows: pl.Scalar[pl.INDEX],
    active_rows: pl.Scalar[pl.INDEX],
    mode: pl.Scalar[pl.INDEX],
):
    """Collapse streams; mode 1 normalizes, and mode 2 also stores raw BF16 rows."""
    ...


@pl.jit.inline
def hc_head(
    x_hc: pl.Tensor,
    pre_mix: pl.Tensor,
    output: pl.Tensor,
    worker_limit: pl.constexpr,
):
    """Collapse HC streams with the delayed pre-mix and BF16 rounding."""
    t_dim = pl.tensor.dim(x_hc, 0)
    input_view = pl.reshape(x_hc, [t_dim, HC_MULT, D])
    mix_view = pl.reshape(pre_mix, [t_dim, HC_MULT])
    output_view = pl.reshape(output, [t_dim, D])
    collapsed_view = pl.reshape(output, [t_dim, D])
    workers = pl.max(pl.min((t_dim + PRE_T_TILE - 1) // PRE_T_TILE, worker_limit), 1)
    with pl.spmd(workers, name_hint="hc_collapse", allow_early_resolve=True):
        _hc_collapse(input_view, mix_view, mix_view, output_view, collapsed_view, t_dim, t_dim, 0)
    return output


@pl.jit.inline
def zero_bf16_padding(x: pl.Tensor, valid_rows: pl.Scalar[pl.INT32]):
    """Zero padding rows beyond the local active token count."""
    tokens = pl.tensor.dim(x, 0)
    if tokens > 0 and valid_rows < tokens:
        for row in pl.spmd(tokens, name_hint="zero_bf16_padding"):
            if row >= valid_rows:
                x[row : row + 1, 0:D] = pl.full([1, D], dtype=pl.BF16, value=0.0)


@pl.jit.extern(core_type="aiv", source=_KERNELS / "hc_post.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _hc_post(
    sublayer: pl.Tensor, residual: pl.Tensor, post_mix: pl.Tensor, residual_mix: pl.Tensor,
    output: pl.Out[pl.Tensor], tokens: pl.Scalar[pl.INDEX],
) -> pl.Tensor:
    """Mix four residual streams with ordered FP32 register operations."""
    ...


@pl.jit.inline(auto_scope=False)
def hc_post_regscope_block(
    sublayer: pl.Tensor,
    residual_rows: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix_flat: pl.Tensor,
    output_rows: pl.Tensor,
    block: pl.Scalar[pl.INDEX],
    stream_tile: pl.constexpr,
):
    """Mix a hidden-width tile using register accumulators."""
    pair_block = block // (D // D_TILE)
    t = pair_block // (HC_MULT // stream_tile)
    out_base = (pair_block % (HC_MULT // stream_tile)) * stream_tile
    d0 = (block % (D // D_TILE)) * D_TILE
    row = t * HC_MULT
    x_tile = pl.tile.load(sublayer, [t, d0], [1, D_TILE], target_memory=pl.Mem.Vec)
    x_tile = pl.tile.reshape(x_tile, [D_TILE])
    post = pl.tile.load(post_mix, [t, 0], [1, 8], valid_shape=[1, HC_MULT], target_memory=pl.Mem.Vec)
    post = pl.tile.reshape(post, [8])
    mix = pl.tile.load(residual_mix_flat, [t, 0], [1, HC_MULT * HC_MULT], target_memory=pl.Mem.Vec)
    mix = pl.tile.reshape(mix, [HC_MULT * HC_MULT])
    residual = pl.tile.load(residual_rows, [row, d0], [HC_MULT, D_TILE], target_memory=pl.Mem.Vec)
    residual = pl.tile.reshape(residual, [HC_MULT * D_TILE])
    output = pl.tile.create([stream_tile * D_TILE], dtype=pl.FP32, target_memory=pl.Mem.Vec)
    with pl.reg_scope():
        mask = pl.reg.full_mask(lanes=64)
        for segment in pl.hardware_loop(D_TILE // 64):
            offset = segment * 64
            x = pl.reg.convert(pl.reg.load(x_tile, offset, lanes=64), dtype=pl.FP32)
            r0 = pl.reg.load(residual, offset, lanes=64)
            r1 = pl.reg.load(residual, D_TILE + offset, lanes=64)
            r2 = pl.reg.load(residual, 2 * D_TILE + offset, lanes=64)
            r3 = pl.reg.load(residual, 3 * D_TILE + offset, lanes=64)
            for stream in pl.unroll(stream_tile):
                h = out_base + stream
                c0 = pl.reg.broadcast(pl.reg.load(post, h, lanes=1), lanes=64, dtype=pl.FP32)
                c1 = pl.reg.broadcast(pl.reg.load(mix, h, lanes=1), lanes=64, dtype=pl.FP32)
                c2 = pl.reg.broadcast(pl.reg.load(mix, HC_MULT + h, lanes=1), lanes=64, dtype=pl.FP32)
                c3 = pl.reg.broadcast(pl.reg.load(mix, 2 * HC_MULT + h, lanes=1), lanes=64, dtype=pl.FP32)
                c4 = pl.reg.broadcast(pl.reg.load(mix, 3 * HC_MULT + h, lanes=1), lanes=64, dtype=pl.FP32)
                value = pl.reg.mul(x, c0, mask)
                value = pl.reg.add(value, pl.reg.mul(r0, c1, mask), mask)
                value = pl.reg.add(value, pl.reg.mul(r1, c2, mask), mask)
                value = pl.reg.add(value, pl.reg.mul(r2, c3, mask), mask)
                value = pl.reg.add(value, pl.reg.mul(r3, c4, mask), mask)
                written = pl.reg.store(output, stream * D_TILE + offset, value)
    output = pl.tile.reshape(output, [stream_tile, D_TILE])
    pl.tile.store(output, [row + out_base, d0], output_rows)
    return output_rows


@pl.jit.inline
def hc_post_regscope(
    sublayer: pl.Tensor,
    residual: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    output: pl.Tensor,
    deps: pl.Array[2, pl.TASK_ID],
):
    """Mix HC streams with an opt-in A5 RegScope implementation."""
    t_dim = pl.tensor.dim(sublayer, 0)
    residual_rows = pl.reshape(residual, [t_dim * HC_MULT, D])
    residual_mix_flat = pl.reshape(residual_mix, [t_dim, HC_MULT * HC_MULT])
    output_rows = pl.reshape(output, [t_dim * HC_MULT, D])
    total_work = t_dim * (D // D_TILE)
    full_work = total_work // POST_WORKER_TILE * POST_WORKER_TILE
    tail_work = (total_work - full_work) * HC_MULT
    workers = pl.min(total_work * HC_MULT, POST_WORKER_TILE)
    with pl.spmd(workers, name_hint="hc_post", deps=[deps]):
        worker = pl.tile.get_block_idx()
        for block in pl.pipeline(worker, full_work, POST_WORKER_TILE, stage=1):
            hc_post_regscope_block(sublayer, residual_rows, post_mix, residual_mix_flat, output_rows, block, HC_MULT)
        for tail in pl.range(worker, tail_work, POST_WORKER_TILE):
            tail_chunk = full_work + tail // HC_MULT
            tail_stream = (tail_chunk // (D // D_TILE)) * HC_MULT + tail % HC_MULT
            tail_block = tail_stream * (D // D_TILE) + tail_chunk % (D // D_TILE)
            hc_post_regscope_block(sublayer, residual_rows, post_mix, residual_mix_flat, output_rows, tail_block, 1)
    return output


@pl.jit.inline
def hc_post(
    sublayer: pl.Tensor,
    residual: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    output: pl.Tensor,
    deps: pl.Array[2, pl.TASK_ID],
):
    """Mix all four residual streams after the supplied producer dependencies."""
    tokens = pl.tensor.dim(sublayer, 0)
    sublayer_view = pl.reshape(sublayer, [tokens, D])
    residual_view = pl.reshape(residual, [tokens, HC_MULT, D])
    post_view = pl.reshape(post_mix, [tokens, HC_MULT])
    mix_view = pl.reshape(residual_mix, [tokens, HC_MULT, HC_MULT])
    output_view = pl.reshape(output, [tokens, HC_MULT, D])
    workers = pl.min(tokens * (D // D_TILE) * HC_MULT, POST_WORKER_TILE)
    with pl.spmd(workers, name_hint="hc_post", deps=[deps]):
        _hc_post(sublayer_view, residual_view, post_view, mix_view, output_view, tokens)
    return output


def _round_hf32(value: torch.Tensor) -> torch.Tensor:
    """Round finite FP32 values to E8M10, with halfway values away from zero."""
    bits = value.contiguous().view(torch.int32)
    rounded = ((bits + 4096) & -8192).view(torch.float32)
    return torch.where(torch.isfinite(value), rounded, value)


def golden_hc_mixes(
    x_hc: torch.Tensor,
    function: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    iterations: int = M.hc_sinkhorn_iters,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project one HC stream into pre, post, and doubly-stochastic residual mixes."""
    flat = x_hc.flatten(-2).float()
    inverse_rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + M.rms_norm_eps)
    projection_input = _round_hf32(flat)
    projection_weight = _round_hf32(function.float())
    mixes = F.linear(projection_input, projection_weight) * inverse_rms
    hc = M.hc_mult
    pre_raw, post_raw, residual_raw = mixes.split((hc, hc, hc * hc), dim=-1)
    pre = torch.sigmoid(pre_raw * scale[0] + base[:hc]) + M.hc_eps
    post = 2 * torch.sigmoid(post_raw * scale[1] + base[hc : 2 * hc])
    residual_base = base[2 * hc :].unflatten(-1, (hc, hc))
    residual = (residual_raw.unflatten(-1, (hc, hc)) * scale[2] + residual_base).softmax(dim=-1)
    residual = residual + M.hc_eps
    residual = residual / (residual.sum(dim=-2, keepdim=True) + M.hc_eps)
    for _ in range(iterations - 1):
        residual = residual / (residual.sum(dim=-1, keepdim=True) + M.hc_eps)
        residual = residual / (residual.sum(dim=-2, keepdim=True) + M.hc_eps)
    return pre, post, residual


def golden_hc_collapse(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Collapse four HC streams with the kernel's pairwise FP32 reduction."""
    products = x.float() * pre_mix.float().unsqueeze(-1)
    first_pair = products[..., 0, :] + products[..., 1, :]
    second_pair = products[..., 2, :] + products[..., 3, :]
    return (first_pair + second_pair).to(x.dtype)


def golden_hc_norm(collapsed: torch.Tensor, norm_weight: torch.Tensor) -> torch.Tensor:
    """Normalize BF16 HC rows with sequential 64-lane FP32 accumulators."""
    value = collapsed.float()
    width = value.shape[-1]
    square = value.square().reshape(*value.shape[:-1], width // 64, 64)
    partial = torch.zeros_like(square[..., 0, :])
    for chunk in range(width // 64):
        partial = partial + square[..., chunk, :]
    while partial.shape[-1] > 1:
        partial = partial[..., 0::2] + partial[..., 1::2]
    argument = partial * (1.0 / width) + NORM_EPS
    inverse = torch.ones_like(argument) / torch.sqrt(argument)
    return (value * inverse * norm_weight.float()).to(torch.bfloat16)


def golden_hc_head_norm(x: torch.Tensor, pre_mix: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Collapse four HC streams to BF16 and normalize the resulting rows."""
    collapsed = golden_hc_collapse(x, pre_mix).bfloat16()
    return golden_hc_norm(collapsed, weight)


def golden_hc_post(
    sublayer: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    residual_mix: torch.Tensor,
) -> torch.Tensor:
    """Expand a sublayer result and mix the four residual streams in FP32."""
    update = post_mix.unsqueeze(-1) * sublayer.unsqueeze(-2)
    skip = (residual_mix.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=-3)
    return update.float() + skip.float()


def golden_l2_hc_pre(tensors):
    """Compute the four outputs with the incoming mix independent of the next mix."""
    pre_mix, post_mix, residual_mix = golden_hc_mixes(
        tensors["x_hc"], tensors["function"], tensors["scale"], tensors["base"],
    )
    tensors["pre_mix"][:] = pre_mix
    tensors["post_mix"][:] = post_mix
    tensors["residual_mix"][:] = residual_mix
    collapsed = golden_hc_collapse(tensors["x_hc"], tensors["incoming_pre_mix"])
    tensors["output"][:] = collapsed.to(torch.bfloat16)


def golden_l2_hc_post(tensors):
    """Fill the expanded residual stream with FP32 accumulation and storage."""
    tensors["output"][:] = golden_hc_post(
        tensors["sublayer"], tensors["residual"], tensors["post_mix"], tensors["residual_mix"],
    )


def golden_l2_hc_head(tensors):
    """Fill the expected collapsed stream output."""
    tensors["output"][:] = golden_hc_collapse(tensors["x_hc"], tensors["pre_mix"]).to(torch.bfloat16)


def make_hc_inputs(tokens, seed, case):
    """Seeded HC fixtures with V4.1 layer-1 attention weight statistics."""
    gen = torch.Generator().manual_seed(seed)
    magnitudes = (0.5 + torch.arange(tokens) % 4).reshape(-1, 1, 1)
    x_hc = (torch.randn(tokens, HC_MULT, D, generator=gen) * magnitudes).bfloat16().float()

    # Per-row standard deviations and gate parameters from the V4.1 Flash 1017 checkpoint.
    function_std = torch.tensor([
        0.029376, 0.031490, 0.055490, 0.049310, 0.023323, 0.181760, 0.112393, 0.096347,
        0.033303, 0.034775, 0.035849, 0.039710, 0.033325, 0.030450, 0.034318, 0.033540,
        0.033511, 0.050113, 0.030742, 0.028955, 0.027775, 0.029948, 0.031218, 0.038125,
    ])
    scale = torch.tensor([0.070230, 0.019080, 0.172269])
    base = torch.tensor([
        -2.338990, 2.601740, -2.912670, -3.033931, -0.503342, -0.577662, -3.500769, -2.494710,
        2.750622, -2.458451, -3.305034, -3.223911, -2.937424, 2.832767, -0.933163, -2.659624,
        -2.655424, -1.275563, 2.152474, -3.270429, -2.550060, -2.554717, -2.587215, 2.604832,
    ])

    # Incoming gates use the preceding layer's FFN pre-gate weight statistics.
    incoming_std = torch.tensor([0.041462, 0.027009, 0.054705, 0.025310]) * math.sqrt(HC_DIM) * 0.031293
    incoming_base = torch.tensor([2.531060, -2.808948, -3.796850, -2.947976])
    incoming_logits = torch.randn(tokens, HC_MULT, generator=gen) * incoming_std + incoming_base
    values = {
        "x_hc": x_hc,
        "pre_mix": torch.sigmoid(incoming_logits) + HC_EPS,
        "hc_attn_fn": torch.randn(MIX_HC, HC_DIM, generator=gen) * function_std[:, None],
        "hc_attn_scale": scale,
        "hc_attn_base": base,
        "attn_norm_weight": (torch.randn(D, generator=gen) * 0.001745 + 0.021617).bfloat16(),
    }
    if case == "zero":
        values["x_hc"] = torch.zeros_like(x_hc)
    return values


@pl.jit
def l2_hc_pre(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    function: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    scale: pl.Tensor[[3], pl.FP32],
    base: pl.Tensor[[MIX_HC], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    pre_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    post_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    residual_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32]],
):
    """Validate the shared HcPre computation."""
    x_hc.bind_dynamic(0, T_DYN)
    incoming_pre_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    pre_mix.bind_dynamic(0, T_DYN)
    post_mix.bind_dynamic(0, T_DYN)
    residual_mix.bind_dynamic(0, T_DYN)
    hc_pre(x_hc, incoming_pre_mix, function, scale, base, output, pre_mix, post_mix, residual_mix)
    return output


def build_l2_hc_pre_specs(batch: int = 33, sequence: int = 1):
    """Build coefficient and independent incoming-mix fixtures."""
    from golden import TensorSpec

    tokens = batch * sequence
    inputs = make_hc_inputs(tokens, seed=3, case="normal")
    return [
        TensorSpec("x_hc", [tokens, HC_MULT, D], torch.float32, init_value=inputs["x_hc"]),
        TensorSpec("incoming_pre_mix", [tokens, HC_MULT], torch.float32, init_value=inputs["pre_mix"]),
        TensorSpec("function", [MIX_HC, HC_DIM], torch.float32, init_value=inputs["hc_attn_fn"]),
        TensorSpec("scale", [3], torch.float32, init_value=inputs["hc_attn_scale"]),
        TensorSpec("base", [MIX_HC], torch.float32, init_value=inputs["hc_attn_base"]),
        TensorSpec("output", [tokens, D], torch.bfloat16),
        TensorSpec("pre_mix", [tokens, HC_MULT], torch.float32),
        TensorSpec("post_mix", [tokens, HC_MULT], torch.float32),
        TensorSpec("residual_mix", [tokens, HC_MULT, HC_MULT], torch.float32),
    ]


@pl.jit
def l2_hc_post(
    sublayer: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    """Validate HC post mapping and residual mixing."""
    sublayer.bind_dynamic(0, T_DYN)
    residual.bind_dynamic(0, T_DYN)
    post_mix.bind_dynamic(0, T_DYN)
    residual_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    hc_post_deps = pl.array.create(2, pl.TASK_ID)
    hc_post(sublayer, residual, post_mix, residual_mix, output, deps=hc_post_deps)
    return output


@pl.jit
def l2_hc_post_regscope(
    sublayer: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    """Validate the opt-in A5 RegScope HC post implementation."""
    sublayer.bind_dynamic(0, T_DYN)
    residual.bind_dynamic(0, T_DYN)
    post_mix.bind_dynamic(0, T_DYN)
    residual_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    hc_post_deps = pl.array.create(2, pl.TASK_ID)
    hc_post_regscope(sublayer, residual, post_mix, residual_mix, output, deps=hc_post_deps)
    return output


def build_l2_hc_post_specs(batch: int = 33, sequence: int = 1):
    """Build BF16-valued residual streams and nonsymmetric mixing coefficients."""
    from golden import TensorSpec

    tokens = batch * sequence
    generator = torch.Generator().manual_seed(7)

    def init_sublayer():
        return torch.randn(tokens, D, generator=generator).bfloat16()

    def init_residual():
        return torch.randn(tokens, HC_MULT, D, generator=generator)

    def init_post_mix():
        return 2 * torch.sigmoid(torch.randn(tokens, HC_MULT, generator=generator))

    def init_residual_mix():
        mix = torch.rand(tokens, HC_MULT, HC_MULT, generator=generator) + HC_EPS
        for _ in range(HC_SINKHORN_ITER):
            mix = mix / (mix.sum(dim=-1, keepdim=True) + HC_EPS)
            mix = mix / (mix.sum(dim=-2, keepdim=True) + HC_EPS)
        return mix

    return [
        TensorSpec("sublayer", [tokens, D], torch.bfloat16, init_value=init_sublayer),
        TensorSpec("residual", [tokens, HC_MULT, D], torch.float32, init_value=init_residual),
        TensorSpec("post_mix", [tokens, HC_MULT], torch.float32, init_value=init_post_mix),
        TensorSpec("residual_mix", [tokens, HC_MULT, HC_MULT], torch.float32, init_value=init_residual_mix),
        TensorSpec("output", [tokens, HC_MULT, D], torch.float32),
    ]


@pl.jit
def l2_hc_head(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    """Validate the final HC stream collapse."""
    x_hc.bind_dynamic(0, T_DYN)
    pre_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    hc_head(x_hc, pre_mix, output, worker_limit=64)
    return output


def build_l2_hc_head_specs(batch: int = 33, sequence: int = 1):
    """Build deterministic inputs and output for stream-collapse validation."""
    from golden import TensorSpec

    tokens = batch * sequence
    generator = torch.Generator().manual_seed(3)

    def init_x_hc():
        return torch.randn(tokens, HC_MULT, D, generator=generator)

    def init_pre_mix():
        return torch.rand(tokens, HC_MULT, generator=generator)

    return [
        TensorSpec("x_hc", [tokens, HC_MULT, D], torch.float32, init_value=init_x_hc),
        TensorSpec("pre_mix", [tokens, HC_MULT], torch.float32, init_value=init_pre_mix),
        TensorSpec("output", [tokens, D], torch.bfloat16),
    ]


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 HC pre, post, and head validation")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--batch", type=int, help="explicit batch size; omitted: bounded correctness cases")
    parser.add_argument("--sequence", type=int)
    parser.add_argument("--op", choices=["pre", "post", "post-regscope", "head", "all"], default="all")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data")
    parser.add_argument("--runtime-dir")
    parser.add_argument("--enable-chip-swimlane", type=int, choices=[1, 2, 3, 4], default=0)
    args = parser.parse_args()
    if args.execution_mode and args.op in ("pre", "all"):
        parser.error("Fused HC pre requires A5 device execution")
    if args.op == "all" and (args.golden_data or args.runtime_dir):
        parser.error("replay requires selecting one --op")
    explicit_shape = args.batch is not None or args.sequence is not None
    args.batch = 33 if args.batch is None else args.batch
    args.sequence = 1 if args.sequence is None else args.sequence
    cases = [
        ("pre", l2_hc_pre, build_l2_hc_pre_specs, golden_l2_hc_pre),
        ("post", l2_hc_post, build_l2_hc_post_specs, golden_l2_hc_post),
        ("head", l2_hc_head, build_l2_hc_head_specs, golden_l2_hc_head),
    ]
    if args.op == "post-regscope":
        cases = [("post-regscope", l2_hc_post_regscope, build_l2_hc_post_specs, golden_l2_hc_post)]
    failures = []
    for name, fn, build_specs, golden_fn in cases:
        if args.op != "all" and args.op != name:
            continue
        print(f"[HC] {name}", flush=True)
        if name == "pre":
            compare_fn = {
                "pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
                "post_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
                "residual_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
            }
            compare_fn["output"] = ratio_allclose(atol=1e-4, rtol=1.0 / 128)
        elif name in ("post", "post-regscope"):
            compare_fn = {"output": ratio_allclose(atol=1e-6, rtol=1e-5)}
        else:
            compare_fn = {"output": ratio_allclose(atol=1e-4, rtol=1.0 / 128)}
        config = {"backend_type": args.backend_type, "execution_mode": args.execution_mode}
        config.update(device_id=args.device, enable_chip_swimlane=args.enable_chip_swimlane)
        shapes = [(args.batch, args.sequence)]
        if name == "pre" and not explicit_shape and not args.golden_data and not args.runtime_dir:
            shapes = [(rows, 1) for rows in (1, 65, 81, 97, 129, 513)]
        for batch, sequence in shapes:
            print(f"[HC] {name} rows={batch * sequence}", flush=True)
            result = run(
                fn=fn,
                specs=build_specs(batch, sequence),
                golden_fn=golden_fn,
                config=config,
                rtol=1e-3,
                atol=1e-3,
                compare_fn=compare_fn,
                compile_only=args.compile_only,
                save_data=args.save_data,
                golden_data=args.golden_data,
                runtime_dir=args.runtime_dir,
            )
            if not result.passed:
                failures.append(f"{name}[{batch * sequence}]: {result.error or 'validation failed'}")
    if failures:
        raise SystemExit("; ".join(failures))
