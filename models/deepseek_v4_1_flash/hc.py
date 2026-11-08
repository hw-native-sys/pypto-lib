# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 HC coefficient generation, stream collapse, residual expansion, and final collapse."""

import math
from pathlib import Path

import pypto.language as pl
import torch
import torch.nn.functional as F
from pypto.runtime import pto_isa_include_dir

from config import FLASH as M, SP_T_DYN, T_DYN


# model config
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
NORM_EPS = M.rms_norm_eps

# tiling
MIX_PAD = 32
HC_PAD = 8
T_TILE = 8
LINEAR_T_TILE = 16
HF32_LINEAR_T_TILE = 80  # Matches M_TILE in kernels/hc_linear.cpp.
SINKHORN_T_TILE = 6
SINKHORN_WORKER_TILE = 12
RMS_K_TILE = 1024
LINEAR_OK = 4
D_TILE = 2560
PRE_T_TILE = 2
HEAD_D_TILE = 5120
PRE_NORM_T_TILE = 8
POST_WORKER_TILE = 64


@pl.jit.extern(core_type="aic", source=Path(__file__).parent / "kernels" / "hc_linear.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _hc_linear(
    x: pl.Tensor,
    weight: pl.Tensor,
    partials: pl.Out[pl.Tensor],
    rows: pl.Scalar[pl.INDEX],
    padded_rows: pl.Scalar[pl.INDEX],
) -> pl.Tensor:
    """Project FP32 HC operands with nearest-away HF32 and FP32 accumulation."""
    ...


@pl.jit.extern(core_type="aiv", source=Path(__file__).parent / "kernels" / "hc_sinkhorn.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _hc_sinkhorn(
    partials: pl.Tensor, inv_rms: pl.Tensor, scale: pl.Tensor, base: pl.Tensor,
    pre_mix: pl.Out[pl.Tensor], post_mix: pl.Out[pl.Tensor], residual_mix: pl.Out[pl.Tensor],
    tokens: pl.Scalar[pl.INDEX], padded_rows: pl.Scalar[pl.INDEX],
    eps: pl.Scalar[pl.FP32], iterations: pl.Scalar[pl.INDEX],
) -> pl.Tensor:
    """Compute FP32 coefficients in a register-resident VF."""
    ...


@pl.jit.inline
def hc_sinkhorn(
    inv_rms: pl.Tensor, partials: pl.Tensor, scale: pl.Tensor, base: pl.Tensor,
    pre_mix: pl.Tensor, post_mix: pl.Tensor, residual_mix: pl.Tensor,
):
    """Normalize six independent HC matrices per vector worker."""
    tokens = pl.tensor.dim(pre_mix, 0)
    padded_rows = pl.tensor.dim(inv_rms, 0)
    blocks = (tokens + SINKHORN_T_TILE - 1) // SINKHORN_T_TILE
    workers = pl.min(blocks, SINKHORN_WORKER_TILE)
    with pl.spmd(workers, name_hint="mhc_sinkhorn") as ready:
        _hc_sinkhorn(partials, inv_rms, scale, base, pre_mix, post_mix, residual_mix,
                     tokens, padded_rows, HC_EPS, HC_SINKHORN_ITER)
    return ready


@pl.jit.inline
def hc_project(x_hc: pl.Tensor, function: pl.Tensor, rms_worker_limit: pl.constexpr):
    """Compute FP32 inverse RMS and split-K HC projection coefficients."""
    t_dim = pl.tensor.dim(x_hc, 0)
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])

    inv_rms = pl.create_tensor([t_linear, 1], dtype=pl.FP32)
    rms_blocks = (t_dim + T_TILE - 1) // T_TILE
    rms_workers = pl.max(pl.min(rms_blocks, rms_worker_limit), 1)
    for worker in pl.spmd(rms_workers, name_hint="mhc_rms"):
        for block in pl.range(worker, rms_blocks, rms_workers):
            t0 = block * T_TILE
            valid_rows = pl.min(T_TILE, t_dim - t0)
            sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.pipeline(HC_DIM // RMS_K_TILE, stage=4):
                k0 = kb * RMS_K_TILE
                rms_x_tile = pl.slice(
                    x_flat,
                    [T_TILE, RMS_K_TILE],
                    [t0, k0],
                    valid_shape=[valid_rows, RMS_K_TILE],
                )
                sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(rms_x_tile, rms_x_tile)), [1, T_TILE]))
            rms_arg = pl.add(pl.mul(sq_sum, HC_DIM_INV), NORM_EPS)
            inv_rms[t0 : t0 + T_TILE, 0:1] = pl.reshape(
                pl.rsqrt(rms_arg, high_precision=True), [T_TILE, 1]
            )

    mixes_partials = pl.create_tensor([LINEAR_OK * t_linear, MIX_PAD], dtype=pl.FP32)
    tasks = ((t_dim + HF32_LINEAR_T_TILE - 1) // HF32_LINEAR_T_TILE) * LINEAR_OK
    workers = pl.min(tasks, pl.system.available_cluster_count())
    with pl.spmd(workers, name_hint="mhc_linear"):
        _hc_linear(x_flat, function, mixes_partials, t_dim, t_linear)

    return inv_rms, mixes_partials


@pl.jit.inline
def hc_head(
    x_hc: pl.Tensor,
    pre_mix: pl.Tensor,
    output: pl.Tensor,
    worker_limit: pl.constexpr,
):
    """Collapse HC streams with the delayed pre-mix and BF16 rounding."""
    t_dim = pl.tensor.dim(x_hc, 0)
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])
    workers = pl.max(pl.min((t_dim + PRE_T_TILE - 1) // PRE_T_TILE, worker_limit), 1)
    rows_per_worker = (t_dim + workers - 1) // workers
    for block in pl.spmd(workers, name_hint="hc_collapse", allow_early_resolve=True):
        for t in pl.range(block * rows_per_worker, pl.min((block + 1) * rows_per_worker, t_dim)):
            pre_tile = pl.load(pre_mix, [t, 0], [1, HC_PAD], valid_shape=[1, HC_MULT])
            pre0 = pl.tile.read(pre_tile, [0, 0])
            pre1 = pl.tile.read(pre_tile, [0, 1])
            pre2 = pl.tile.read(pre_tile, [0, 2])
            pre3 = pl.tile.read(pre_tile, [0, 3])
            for d0 in pl.pipeline(0, D, HEAD_D_TILE, stage=2):
                x0 = pl.load(x_flat, [t, d0], [1, HEAD_D_TILE])
                x1 = pl.load(x_flat, [t, D + d0], [1, HEAD_D_TILE])
                x2 = pl.load(x_flat, [t, 2 * D + d0], [1, HEAD_D_TILE])
                x3 = pl.load(x_flat, [t, 3 * D + d0], [1, HEAD_D_TILE])
                y0 = pl.mul(x0, pre0)
                y1 = pl.mul(x1, pre1)
                y2 = pl.mul(x2, pre2)
                y3 = pl.mul(x3, pre3)
                y01 = pl.add(y0, y1)
                y23 = pl.add(y2, y3)
                y_tile = pl.add(y01, y23)
                y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
                pl.store(y_bf16, [t, d0], output)
    return output


@pl.jit.extern(core_type="aiv", source=Path(__file__).parent / "kernels" / "hc_collapse_norm.cpp",
               include_dirs=(pto_isa_include_dir(),))
def hc_collapse_norm(
    x: pl.Tensor,
    pre_mix: pl.Tensor,
    norm_weight: pl.Tensor,
    output: pl.Out[pl.Tensor],
    rows: pl.Scalar[pl.INDEX],
):
    """Collapse and normalize eight HC rows with BF16 rounding."""
    ...


@pl.jit.inline(auto_scope=False)
def hc_head_norm(
    x_hc: pl.Tensor,
    pre_mix: pl.Tensor,
    norm_weight: pl.Tensor[[D], pl.BF16],
    output: pl.Tensor,
    worker_limit: pl.constexpr,
):
    """Collapse eight HC rows and normalize their rounded BF16 values."""
    tokens = pl.tensor.dim(x_hc, 0)
    workers = pl.min((tokens + PRE_NORM_T_TILE - 1) // PRE_NORM_T_TILE, worker_limit)
    with pl.spmd(workers, name_hint="hc_collapse_norm") as ready:
        hc_collapse_norm(x_hc, pre_mix, norm_weight, output, tokens)
    return ready


@pl.jit.inline
def hc_pre(
    x_hc: pl.Tensor,
    incoming_pre_mix: pl.Tensor,
    function: pl.Tensor,
    scale: pl.Tensor,
    base: pl.Tensor,
    output: pl.Tensor,
    next_pre_mix: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    rms_worker_limit: pl.constexpr,
    head_worker_limit: pl.constexpr,
):
    """Generate delayed HC coefficients and collapse the incoming streams."""
    inv_rms, partials = hc_project(x_hc, function, rms_worker_limit=rms_worker_limit)
    mixes_ready = hc_sinkhorn(inv_rms, partials, scale, base, next_pre_mix, post_mix, residual_mix)
    hc_head(x_hc, incoming_pre_mix, output, worker_limit=head_worker_limit)
    return mixes_ready


@pl.jit.inline
def zero_bf16_padding(x: pl.Tensor, valid_rows: pl.Scalar[pl.INT32]):
    """Clear physical rows that belong to an empty or partial TP slab.

    A TP rank with no logical token still publishes one fixed-capacity slab so
    every rank executes the same distributed window protocol.  Clearing the
    unused rows makes that padding deterministic and prevents an empty owner
    from publishing uninitialized data.
    """
    tokens = pl.tensor.dim(x, 0)
    if tokens > 0 and valid_rows < tokens:
        for row in pl.spmd(tokens, name_hint="decode_sp_zero_padding"):
            if row >= valid_rows:
                x[row : row + 1, 0:D] = pl.full([1, D], dtype=pl.BF16, value=0.0)


@pl.jit.extern(core_type="aiv", source=Path(__file__).parent / "kernels" / "hc_post.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _hc_post(
    sublayer: pl.Tensor, residual: pl.Tensor, post_mix: pl.Tensor, residual_mix: pl.Tensor,
    output: pl.Out[pl.Tensor], tokens: pl.Scalar[pl.INDEX],
) -> pl.Tensor:
    """Mix four residual streams with ordered FP32 register operations."""
    ...


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
    workers = pl.min(tokens * (D // D_TILE) * HC_MULT, POST_WORKER_TILE)
    with pl.spmd(workers, name_hint="hc_post", deps=[deps]):
        _hc_post(sublayer, residual, post_mix, residual_mix, output, tokens)
    return output


@pl.jit.inline
def prefill_sp_post(
    sublayer: pl.Tensor[[SP_T_DYN, D], pl.BF16],
    residual: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    post_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[SP_T_DYN, HC_MULT, HC_MULT], pl.FP32],
    output: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    next_pre_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Run local mHC-post, writing active values and zero padding in one producer.

    A later pure-output clear is not an ordering edge for the post task. Each
    residual element is instead written exactly once, with hc_post's FP32
    accumulation order and FP32 output.
    """
    tokens = pl.tensor.dim(residual, 0)
    residual_flat = pl.reshape(residual, [tokens, HC_MULT * D])
    mix_flat = pl.reshape(residual_mix, [tokens, HC_MULT * HC_MULT])
    output_flat = pl.reshape(output, [tokens, HC_MULT * D])
    for worker in pl.spmd(64, name_hint="sp_mhc_post"):
        for block in pl.range(worker, tokens * HC_MULT, 64):
            row = block // HC_MULT
            out_h = block % HC_MULT
            for col in pl.pipeline(0, D, 1024, stage=2):
                value = pl.full([1, 1024], dtype=pl.FP32, value=0.0)
                if tp_rank * tokens + row < num_tokens:
                    hidden = pl.cast(sublayer[row:row + 1, col:col + 1024], pl.FP32)
                    value = pl.mul(hidden, pl.read(post_mix, [row, out_h]))
                    for in_h in pl.unroll(HC_MULT):
                        lane = residual_flat[row:row + 1, in_h * D + col:in_h * D + col + 1024]
                        value = pl.add(value, pl.mul(lane, pl.read(mix_flat, [row, in_h * HC_MULT + out_h])))
                output_flat[row:row + 1, out_h * D + col:out_h * D + col + 1024] = value
    active = pl.max(0, pl.min(tokens, num_tokens - tp_rank * tokens))
    if active < tokens:
        for worker in pl.spmd(32, name_hint="sp_mix_zero_padding"):
            for row in pl.range(active + worker, tokens, 32):
                zero_mix = pl.full([1, 8], dtype=pl.FP32, value=0.0)
                next_pre_mix[row:row + 1, :] = pl.set_validshape(zero_mix, 1, HC_MULT)
    return output, next_pre_mix


def _round_hf32(value: torch.Tensor) -> torch.Tensor:
    """Round finite FP32 values to E8M10, with halfway values away from zero."""
    bits = value.contiguous().view(torch.int32)
    rounded = ((bits + 4096) & -8192).view(torch.float32)
    return torch.where(torch.isfinite(value), rounded, value)


def golden_mhc_mixes(
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
    pre_mix, post_mix, residual_mix = golden_mhc_mixes(
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


HC_INPUT_NAMES = ("x_hc", "pre_mix", "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_weight")


def make_hc_inputs(tokens, seed, case):
    """mHC inputs with distinct per-token magnitudes so both RMS statistics stay observable."""
    gen = torch.Generator().manual_seed(seed)
    magnitudes = (0.5 + torch.arange(tokens) % 4).reshape(-1, 1, 1)
    x_hc = (torch.randn(tokens, HC_MULT, D, generator=gen) * magnitudes).bfloat16().float()
    values = {
        "x_hc": x_hc,
        "pre_mix": torch.sigmoid(torch.randn(tokens, HC_MULT, generator=gen)) + M.hc_eps,
        "hc_attn_fn": torch.randn(MIX_HC, HC_DIM, generator=gen) / math.sqrt(HC_DIM),
        "hc_attn_scale": torch.randn(3, generator=gen),
        "hc_attn_base": torch.randn(MIX_HC, generator=gen),
        "attn_norm_weight": (torch.randn(D, generator=gen) * 0.1 + 1).bfloat16(),
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
    hc_pre(x_hc, incoming_pre_mix, function, scale, base, output, pre_mix, post_mix, residual_mix, rms_worker_limit=64, head_worker_limit=64)
    return output


def build_l2_hc_pre_specs(batch: int = 33, sequence: int = 1):
    """Build coefficient and independent incoming-mix fixtures."""
    from golden import TensorSpec

    tokens = batch * sequence
    generator = torch.Generator().manual_seed(3)
    incoming_generator = torch.Generator().manual_seed(5)

    def init_x_hc():
        return torch.randn(tokens, HC_MULT, D, generator=generator)

    def init_incoming_pre_mix():
        return torch.sigmoid(torch.randn(tokens, HC_MULT, generator=incoming_generator)) + HC_EPS

    def init_function():
        return torch.randn(MIX_HC, HC_DIM, generator=generator) / math.sqrt(HC_DIM)

    def init_scale():
        return torch.randn(3, generator=generator)

    def init_base():
        return torch.randn(MIX_HC, generator=generator)

    return [
        TensorSpec("x_hc", [tokens, HC_MULT, D], torch.float32, init_value=init_x_hc),
        TensorSpec("incoming_pre_mix", [tokens, HC_MULT], torch.float32, init_value=init_incoming_pre_mix),
        TensorSpec("function", [MIX_HC, HC_DIM], torch.float32, init_value=init_function),
        TensorSpec("scale", [3], torch.float32, init_value=init_scale),
        TensorSpec("base", [MIX_HC], torch.float32, init_value=init_base),
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
    parser.add_argument("--batch", type=int, default=33)
    parser.add_argument("--sequence", type=int, default=1)
    parser.add_argument("--op", choices=["pre", "post", "head", "all"], default="all")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data")
    parser.add_argument("--runtime-dir")
    parser.add_argument("--enable-chip-swimlane", type=int, choices=[1, 2, 3, 4], default=0)
    args = parser.parse_args()
    if args.op == "all" and (args.golden_data or args.runtime_dir):
        parser.error("replay requires selecting one --op")
    cases = [
        ("pre", l2_hc_pre, build_l2_hc_pre_specs, golden_l2_hc_pre),
        ("post", l2_hc_post, build_l2_hc_post_specs, golden_l2_hc_post),
        ("head", l2_hc_head, build_l2_hc_head_specs, golden_l2_hc_head),
    ]
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
        elif name == "post":
            compare_fn = {"output": ratio_allclose(atol=1e-6, rtol=1e-5)}
        else:
            compare_fn = {"output": ratio_allclose(atol=1e-4, rtol=1.0 / 128)}
        config = {"backend_type": args.backend_type, "execution_mode": args.execution_mode}
        config.update(device_id=args.device, enable_chip_swimlane=args.enable_chip_swimlane)
        result = run(
            fn=fn,
            specs=build_specs(args.batch, args.sequence),
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
            failures.append(f"{name}: {result.error or 'validation failed'}")
    if failures:
        raise SystemExit("; ".join(failures))
