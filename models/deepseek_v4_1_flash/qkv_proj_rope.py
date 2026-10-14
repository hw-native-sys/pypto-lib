# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared V4.1 Q/KV projection, normalization, and RoPE stages."""

import pypto.language as pl
import torch

from rmsnorm import rms_norm
from config import (
    FLASH as M,
    T_DYN,
    TP_SIZE,
    ORI_BLOCKS_DYN,
)


from quant import mxfp8_linear
from rmsnorm import golden_rms_norm


# model config
LOCAL_H = M.num_attention_heads // TP_SIZE
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank

# Shared projection tiling.
M_TILE = 16
MX_M_TILE = 32
N_TILE = 128
K_TILE = 256


_PREFILL_WORKERS = 64
_PREFILL_PROJECTION_K_TILE = 32


@pl.jit.inline
def prefill_c1a_publish_window(
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.FP8E8M0],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    blocks = pl.tensor.dim(cache, 0)
    cache_rows = blocks * 128
    flat = pl.reshape(cache, [cache_rows, HEAD_DIM])
    scale_flat = pl.reshape(scales, [cache_rows, HEAD_DIM // 32])
    with pl.spmd(num_tokens, name_hint="c1a_cache_publish", deps=[cache_ready]) as publish_tid:
        t = pl.tile.get_block_idx()
        slot_i64 = pl.read(slots, [t])
        if slot_i64 >= 0:
            slot = pl.cast(slot_i64, pl.INDEX)
            source = pl.slice(kv, [1, HEAD_DIM * 2], [t, 0], valid_shape=[1, HEAD_DIM])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 1, HEAD_DIM * 2)
            value = pl.reshape(pl.cast(source, pl.FP32), [HEAD_DIM // 16, 32])
            amax = pl.maximum(pl.row_max(pl.abs(value)), 1e-4)
            raw = pl.mul(amax, 1.0 / 448.0)
            bits = pl.reinterpret_view(raw, pl.INT32)
            exponent = pl.shrs(pl.add(bits, 8388607), 23)
            scale = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
            payload = pl.cast(pl.row_expand_div(value, scale), pl.FP8E4M3FN, mode="rint")
            flat[slot:slot + 1, :] = pl.set_validshape(pl.reshape(payload, [1, HEAD_DIM * 2]), 1, HEAD_DIM)
            signed_exponent = pl.sub(exponent, pl.mul(pl.shrs(exponent, 7), 256))
            codes = pl.cast(signed_exponent, pl.INT8)
            encoded = pl.reinterpret_view(pl.reinterpret_view(codes, pl.UINT8), pl.FP8E8M0)
            encoded_row = pl.reshape(encoded, [1, HEAD_DIM // 16])
            encoded_valid = pl.set_validshape(encoded_row, 1, HEAD_DIM // 32)
            scale_flat[slot:slot + 1, :] = encoded_valid
    return cache, scales


def make_mx_projection(width, output_width, output_dtype=pl.BF16, *, name_hint="attention_mx_projection"):
    """Specialize an MXFP8 projection without expanding weights in HBM."""
    fp32_output = output_dtype == pl.FP32

    @pl.jit.inline
    def project(
        x: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.FP8E4M3FN],
        scale: pl.Tensor[[width // 32, output_width], pl.FP8E8M0, pl.MX_B_NN],
        output: pl.Tensor[[T_DYN, output_width], output_dtype],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        for mt in pl.parallel((num_tokens + MX_M_TILE - 1) // MX_M_TILE):
            t0 = mt * MX_M_TILE
            for block in pl.spmd(output_width // N_TILE, name_hint=name_hint):
                n0 = block * N_TILE
                rows = pl.min(MX_M_TILE, num_tokens - t0)
                first = pl.load(x, [t0, 0], [MX_M_TILE, K_TILE], valid_shape=[rows, K_TILE])
                first = pl.set_validshape(pl.fillpad(first, pad_value=pl.PadValue.zero), MX_M_TILE, K_TILE)
                firstq_values = pl.reshape(pl.cast(first, pl.FP32), [MX_M_TILE * (K_TILE // 32), 32])
                firstq_reduce_tmp = pl.create_tile([MX_M_TILE * (K_TILE // 32), 32], dtype=pl.FP32)
                firstq_maximum = pl.maximum(pl.row_max(pl.abs(firstq_values), tmp_tile=firstq_reduce_tmp), 1e-4)
                firstq_bits = pl.reinterpret_view(pl.mul(firstq_maximum, 1.0 / 448.0), pl.INT32)
                firstq_exponent = pl.shrs(pl.add(firstq_bits, 8388607), 23)
                firstq_scale = pl.reinterpret_view(pl.shls(firstq_exponent, 23), pl.FP32)
                firstq_quantized = pl.cast(
                    pl.row_expand_div(firstq_values, firstq_scale), pl.FP8E4M3FN, mode="rint"
                )
                firstq_payload = pl.reshape(firstq_quantized, [MX_M_TILE, K_TILE])
                firstq_signed_exponent = pl.sub(firstq_exponent, pl.mul(pl.shrs(firstq_exponent, 7), 256))
                firstq_codes = pl.reinterpret_view(pl.cast(firstq_signed_exponent, pl.INT8), pl.UINT8)
                firstq_flat = pl.reshape(firstq_codes, [1, MX_M_TILE * (K_TILE // 32)])
                firstq_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                firstq_packed = pl.tmov_x2zz(
                    firstq_flat, firstq_tmp, group_axis=1, dst_rows=MX_M_TILE, dst_cols=8
                )
                a0 = firstq_payload
                sa0 = pl.reinterpret_view(firstq_packed, pl.FP8E8M0)
                b0 = pl.load(weight, [0, n0], [K_TILE, N_TILE])
                sb0 = pl.load(scale, [0, n0], [K_TILE // 32, N_TILE])
                acc = pl.matmul_mx(a0, sa0, b0, sb0)
                for kb in pl.range(1, width // K_TILE):
                    k0 = kb * K_TILE
                    values = pl.load(x, [t0, k0], [MX_M_TILE, K_TILE], valid_shape=[rows, K_TILE])
                    values = pl.set_validshape(pl.fillpad(values, pad_value=pl.PadValue.zero), MX_M_TILE, K_TILE)
                    nextq_values = pl.reshape(pl.cast(values, pl.FP32), [MX_M_TILE * (K_TILE // 32), 32])
                    nextq_reduce_tmp = pl.create_tile([MX_M_TILE * (K_TILE // 32), 32], dtype=pl.FP32)
                    nextq_maximum = pl.maximum(
                        pl.row_max(pl.abs(nextq_values), tmp_tile=nextq_reduce_tmp), 1e-4
                    )
                    nextq_bits = pl.reinterpret_view(pl.mul(nextq_maximum, 1.0 / 448.0), pl.INT32)
                    nextq_exponent = pl.shrs(pl.add(nextq_bits, 8388607), 23)
                    nextq_scale = pl.reinterpret_view(pl.shls(nextq_exponent, 23), pl.FP32)
                    nextq_quantized = pl.cast(
                        pl.row_expand_div(nextq_values, nextq_scale), pl.FP8E4M3FN, mode="rint"
                    )
                    nextq_payload = pl.reshape(nextq_quantized, [MX_M_TILE, K_TILE])
                    nextq_signed_exponent = pl.sub(
                        nextq_exponent, pl.mul(pl.shrs(nextq_exponent, 7), 256)
                    )
                    nextq_codes = pl.reinterpret_view(pl.cast(nextq_signed_exponent, pl.INT8), pl.UINT8)
                    nextq_flat = pl.reshape(nextq_codes, [1, MX_M_TILE * (K_TILE // 32)])
                    nextq_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                    nextq_packed = pl.tmov_x2zz(
                        nextq_flat, nextq_tmp, group_axis=1, dst_rows=MX_M_TILE, dst_cols=8
                    )
                    a = nextq_payload
                    sa = pl.reinterpret_view(nextq_packed, pl.FP8E8M0)
                    b = pl.load(weight, [k0, n0], [K_TILE, N_TILE])
                    sb = pl.load(scale, [k0 // 32, n0], [K_TILE // 32, N_TILE])
                    acc = pl.matmul_mx_acc(acc, a, sa, b, sb)
                if fp32_output:
                    output = pl.store(pl.set_validshape(pl.mul(acc, 1.0), rows, N_TILE), [t0, n0], output)
                else:
                    value = pl.cast(acc, target_type=pl.BF16, mode="rint")
                    output = pl.store(pl.set_validshape(value, rows, N_TILE), [t0, n0], output)
        return output

    return project


def _make_rope_block(heads, inverse, head_dim, rope_dim):
    sign = -1.0 if inverse else 1.0
    nope_dim = head_dim - rope_dim

    @pl.jit.inline
    def rotate_block(
        x: pl.Tensor[[T_DYN, heads * head_dim], pl.BF16],
        cos: pl.Tensor[[T_DYN, rope_dim // 2], pl.FP32],
        sin: pl.Tensor[[T_DYN, rope_dim // 2], pl.FP32],
        output: pl.Tensor[[T_DYN, heads * head_dim], pl.BF16],
        block: pl.Scalar[pl.INT32],
    ):
        t = block // heads
        h = block % heads
        base = h * head_dim
        output[t : t + 1, base : base + nope_dim] = x[t : t + 1, base : base + nope_dim]
        tail = pl.cast(x[t : t + 1, base + nope_dim : base + head_dim], pl.FP32)
        even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
        odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
        c = cos[t : t + 1, :]
        s = pl.mul(sin[t : t + 1, :], sign)
        re = pl.sub(pl.mul(even, c), pl.mul(odd, s))
        im = pl.add(pl.mul(even, s), pl.mul(odd, c))
        rotated = pl.full([1, rope_dim], dtype=pl.FP32, value=0.0)
        rotated = pl.tensor.scatter(re, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
        rotated = pl.tensor.scatter(im, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
        output[t : t + 1, base + nope_dim : base + head_dim] = pl.cast(
            rotated, pl.BF16, mode="rint"
        )
        return output

    return rotate_block


def make_rope(
    heads,
    inverse=False,
    *,
    head_dim=HEAD_DIM,
    rope_dim=ROPE_DIM,
    max_workers=None,
    name_hint="attention_rope",
):
    """Specialize adjacent-pair RoPE with direct or capped strided scheduling."""
    rotate_block = _make_rope_block(heads, inverse, head_dim, rope_dim)
    worker_limit = 2**31 - 1 if max_workers is None else max_workers

    @pl.jit.inline
    def rotate(
        x: pl.Tensor[[T_DYN, heads * head_dim], pl.BF16],
        cos: pl.Tensor[[T_DYN, rope_dim // 2], pl.FP32],
        sin: pl.Tensor[[T_DYN, rope_dim // 2], pl.FP32],
        output: pl.Tensor[[T_DYN, heads * head_dim], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        blocks = num_tokens * heads
        workers = pl.min(blocks, worker_limit)
        for worker in pl.spmd(workers, name_hint=name_hint):
            for block in pl.range(worker, blocks, workers):
                rotate_block(x, cos, sin, output, block)
        return output

    return rotate


def make_rope_with_deps(
    heads,
    inverse=False,
    *,
    head_dim=HEAD_DIM,
    rope_dim=ROPE_DIM,
    max_workers=None,
    name_hint="attention_rope",
):
    """Specialize adjacent-pair RoPE with an explicit producer dependency."""
    rotate_block = _make_rope_block(heads, inverse, head_dim, rope_dim)
    worker_limit = 2**31 - 1 if max_workers is None else max_workers

    @pl.jit.inline
    def rotate(
        x: pl.Tensor[[T_DYN, heads * head_dim], pl.BF16],
        cos: pl.Tensor[[T_DYN, rope_dim // 2], pl.FP32],
        sin: pl.Tensor[[T_DYN, rope_dim // 2], pl.FP32],
        output: pl.Tensor[[T_DYN, heads * head_dim], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
        ready: pl.Scalar[pl.TASK_ID],
    ):
        blocks = num_tokens * heads
        workers = pl.min(blocks, worker_limit)
        with pl.spmd(workers, name_hint=name_hint, deps=[ready]) as rope_tid:
            worker = pl.tile.get_block_idx()
            for block in pl.range(worker, blocks, workers):
                rotate_block(x, cos, sin, output, block)
        return rope_tid

    return rotate


def _make_bf16_projection_block(width, output_width):
    n_tile = min(N_TILE, output_width)
    k_tile = min(K_TILE, width)

    @pl.jit.inline
    def project_block(
        source: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.BF16],
        output: pl.Tensor[[T_DYN, output_width], pl.BF16],
        task: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        output_tiles = output_width // n_tile
        row = task // output_tiles * M_TILE
        col = task % output_tiles * n_tile
        rows = pl.min(M_TILE, num_tokens - row)
        acc = pl.create_tensor([M_TILE, n_tile], dtype=pl.FP32)
        for k in pl.range(0, width, k_tile):
            a = pl.slice(source, [M_TILE, k_tile], [row, k], valid_shape=[rows, k_tile])
            b = pl.slice(weight, [k_tile, n_tile], [k, col])
            acc = pl.matmul_acc(acc, a, b, init_cond=(k == 0))
        output[row : row + M_TILE, col : col + n_tile] = pl.set_validshape(
            pl.cast(acc, pl.BF16, mode="rint"), rows, n_tile
        )
        return output

    return project_block


def make_bf16_projection_staged(width, output_width):
    """Stage FP32 projection values before a separate BF16 narrowing task.

    The BF16 operand passes through Vector before Cube starts its result transfer
    (hw-native-sys/pypto#2829). Store FP32 values through Vector as well: direct
    Acc-to-GM stores were unstable in the concurrent A5 prefill workload.
    Accumulation and RNE rounding are unchanged.
    """
    n_tile = min(N_TILE, output_width)
    k_tile = min(K_TILE, width)

    @pl.jit.inline
    def project(
        source: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.BF16],
        output: pl.Tensor[[T_DYN, output_width], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        tokens = pl.tensor.dim(source, 0)
        accumulated = pl.create_tensor([tokens, output_width], dtype=pl.FP32)
        blocks = (num_tokens + M_TILE - 1) // M_TILE * (output_width // n_tile)
        with pl.spmd(blocks, name_hint="prefill_bf16_projection_cube") as cube_tid:
            block = pl.tile.get_block_idx()
            row = block // (output_width // n_tile) * M_TILE
            col = block % (output_width // n_tile) * n_tile
            rows = pl.min(M_TILE, num_tokens - row)
            acc = pl.create_tensor([M_TILE, n_tile], dtype=pl.FP32)
            for k in pl.range(0, width, k_tile):
                a = pl.slice(source, [M_TILE, k_tile], [row, k], valid_shape=[rows, k_tile])
                # Preserve BF16 values while establishing Vector-to-Cube startup.
                a = pl.cast(pl.cast(a, pl.FP32), pl.BF16, mode="rint")
                b = pl.slice(weight, [k_tile, n_tile], [k, col])
                acc = pl.matmul_acc(acc, a, b, init_cond=(k == 0))
            accumulated[row : row + M_TILE, col : col + n_tile] = pl.set_validshape(pl.mul(acc, 1.0), rows, n_tile)
        with pl.spmd(blocks, name_hint="prefill_bf16_projection_cast", deps=[cube_tid]):
            block = pl.tile.get_block_idx()
            row = block // (output_width // n_tile) * M_TILE
            col = block % (output_width // n_tile) * n_tile
            rows = pl.min(M_TILE, num_tokens - row)
            value = pl.slice(accumulated, [M_TILE, n_tile], [row, col], valid_shape=[rows, n_tile])
            output[row : row + M_TILE, col : col + n_tile] = pl.cast(value, pl.BF16, mode="rint")
        return output

    return project


def make_bf16_projection_with_deps(
    width,
    output_width,
    *,
    workers=32,
    name_hint="attention_bf16_projection",
):
    """Specialize a BF16 projection with an explicit producer dependency."""
    project_block = _make_bf16_projection_block(width, output_width)
    n_tile = min(N_TILE, output_width)

    @pl.jit.inline
    def project(
        source: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.BF16],
        output: pl.Tensor[[T_DYN, output_width], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
        ready: pl.Scalar[pl.TASK_ID],
    ):
        blocks = (num_tokens + M_TILE - 1) // M_TILE * (output_width // n_tile)
        with pl.spmd(workers, name_hint=name_hint, deps=[ready]) as projection_tid:
            worker = pl.tile.get_block_idx()
            for task in pl.range(worker, blocks, workers):
                project_block(source, weight, output, task, num_tokens)
        return projection_tid

    return project


def golden_rope_interleave(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply adjacent-pair rotary embedding to the final dimension."""
    value = x.float().unflatten(-1, (-1, 2))
    real, imag = value.unbind(dim=-1)
    if inverse:
        sin = -sin
    while cos.ndim < real.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    rotated = torch.stack((real * cos - imag * sin, imag * cos + real * sin), dim=-1)
    return rotated.flatten(-2).to(x.dtype)


def golden_qkv_proj_rope(
    x: torch.Tensor,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor | None,
    q_norm_weight: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor | None,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor | None,
    kv_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project normalized query latent and shared KV, then rotate their RoPE tails."""
    qr = golden_rms_norm(mxfp8_linear(x, wq_a, wq_a_scale), q_norm_weight)
    head_dim = wkv.shape[-1]
    num_heads = wq_b.shape[-1] // head_dim
    q = mxfp8_linear(qr, wq_b, wq_b_scale).unflatten(-1, (num_heads, head_dim))
    kv = golden_rms_norm(mxfp8_linear(x, wkv, wkv_scale), kv_norm_weight)
    rd = cos.shape[-1] * 2
    q = torch.cat((q[..., :-rd], golden_rope_interleave(q[..., -rd:], cos, sin)), dim=-1)
    kv = torch.cat((kv[..., :-rd], golden_rope_interleave(kv[..., -rd:], cos, sin)), dim=-1)
    return q, kv, qr


@pl.jit.inline
def _prefill_project_qa(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    output: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Preserve the scale-corrected group-32 Prefill SWA Q projection."""
    scale_storage = pl.tensor.view(scale, [Q_LORA // 16, D // 2], layout=pl.ND)
    for mt in pl.parallel((num_tokens + MX_M_TILE - 1) // MX_M_TILE):
        t0 = mt * MX_M_TILE
        for block in pl.spmd(Q_LORA // N_TILE, name_hint="prefill_attention_q_a"):
            n0 = block * N_TILE
            rows = pl.min(MX_M_TILE, num_tokens - t0)
            acc = pl.tile.full([MX_M_TILE, N_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(D // (2 * _PREFILL_PROJECTION_K_TILE)):
                raw = pl.load(scale_storage, [n0 // 16, kb * 32], [N_TILE // 16, 32])
                raw_u8 = pl.reinterpret_view(raw, pl.UINT8)
                codes = pl.ands(pl.cast(pl.reinterpret_view(raw_u8, pl.INT8), pl.INT32), 255)
                scale_pair = pl.reinterpret_view(pl.maximum(pl.shls(codes, 23), 4194304), pl.FP32)
                for half in pl.unroll(2):
                    k0 = (kb * 2 + half) * _PREFILL_PROJECTION_K_TILE
                    if half == 0:
                        gathered_scale = pl.tile.gather_mask(
                            scale_pair, mask_pattern=pl.tile.MaskPattern.P0101
                        )
                    else:
                        gathered_scale = pl.tile.gather_mask(
                            scale_pair, mask_pattern=pl.tile.MaskPattern.P1010
                        )
                    sb = pl.reshape(gathered_scale, [1, N_TILE])
                    source = pl.load(
                        x,
                        [t0, k0],
                        [MX_M_TILE, _PREFILL_PROJECTION_K_TILE],
                        valid_shape=[rows, _PREFILL_PROJECTION_K_TILE],
                    )
                    source = pl.set_validshape(
                        pl.fillpad(source, pad_value=pl.PadValue.zero),
                        MX_M_TILE,
                        _PREFILL_PROJECTION_K_TILE,
                    )
                    value = pl.cast(source, pl.FP32)
                    reduce_tmp = pl.create_tile([MX_M_TILE, _PREFILL_PROJECTION_K_TILE], dtype=pl.FP32)
                    maximum = pl.maximum(pl.row_max(pl.abs(value), tmp_tile=reduce_tmp), 1e-4)
                    bits = pl.reinterpret_view(pl.mul(maximum, 1.0 / 448.0), pl.INT32)
                    exponent = pl.shrs(pl.add(bits, 8388607), 23)
                    sa = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
                    payload = pl.cast(pl.row_expand_div(value, sa), pl.FP8E4M3FN, mode="rint")
                    a = pl.cast(payload, pl.BF16)
                    b = pl.cast(
                        pl.load(weight, [k0, n0], [_PREFILL_PROJECTION_K_TILE, N_TILE]), pl.BF16
                    )
                    part = pl.col_expand_mul(pl.row_expand_mul(pl.matmul(a, b), sa), sb)
                    acc = pl.add(acc, part)
            result = pl.set_validshape(pl.cast(acc, pl.BF16, mode="rint"), rows, N_TILE)
            output = pl.store(result, [t0, n0], output)
    return output


def _make_q_proj_qr(project, worker_limit, norm_name):
    @pl.jit.inline(auto_scope=False)
    def q_proj_qr(
        x: pl.Tensor[[T_DYN, D], pl.BF16],
        wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
        q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
        query_latent: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        tokens = pl.tensor.dim(x, 0)
        projected = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
        project(x, wq_a, wq_a_scale, projected, num_tokens)
        norm_blocks = (num_tokens + 7) // 8
        norm_workers = pl.min(norm_blocks, worker_limit)
        for norm_worker in pl.spmd(norm_workers, name_hint=norm_name):
            for norm_block in pl.range(norm_worker, norm_blocks, norm_workers):
                rms_norm(projected, q_norm_weight, query_latent, norm_block, num_tokens, width=Q_LORA)
        return query_latent

    return q_proj_qr


def _make_q_proj_rope(project, rotate):
    @pl.jit.inline(auto_scope=False)
    def q_proj_rope(
        query_latent: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        tokens = pl.tensor.dim(query_latent, 0)
        expanded = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
        project(query_latent, wq_b, wq_b_scale, expanded, num_tokens)
        rotate(expanded, rope_cos, rope_sin, query, num_tokens)
        return query

    return q_proj_rope


def _make_kv_proj_rope(project, rotate):
    @pl.jit.inline(auto_scope=False)
    def kv_proj_rope(
        x: pl.Tensor[[T_DYN, D], pl.BF16],
        wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
        rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        window_kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        tokens = pl.tensor.dim(x, 0)
        projected = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
        project(x, wkv, wkv_scale, projected, num_tokens)
        normalized = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
        norm_blocks = (num_tokens + 7) // 8
        for norm_worker in pl.spmd(norm_blocks, name_hint="attention_kv_norm"):
            rms_norm(projected, kv_norm_weight, normalized, norm_worker, num_tokens, width=HEAD_DIM)
        rotate(normalized, rope_cos, rope_sin, window_kv, num_tokens)
        return window_kv

    return kv_proj_rope


@pl.jit.inline(auto_scope=False)
def prefill_kv_proj_rope(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    projected: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    window_kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Run Prefill KV preprocessing with caller-owned dependency scratch."""
    tokens = pl.tensor.dim(x, 0)
    _project_kv(x, wkv, wkv_scale, projected, num_tokens)
    normalized = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    norm_blocks = (num_tokens + 7) // 8
    norm_workers = pl.min(norm_blocks, _PREFILL_WORKERS)
    for norm_worker in pl.spmd(norm_workers, name_hint="prefill_attention_kv_norm"):
        for norm_block in pl.range(norm_worker, norm_blocks, norm_workers):
            rms_norm(projected, kv_norm_weight, normalized, norm_block, num_tokens, width=HEAD_DIM)
    _prefill_rotate_kv(normalized, rope_cos, rope_sin, window_kv, num_tokens)
    return window_kv


@pl.jit.inline(auto_scope=False)
def qkv_proj_rope(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    query_latent: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    window_kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    q_proj_qr(x, wq_a, wq_a_scale, q_norm_weight, query_latent, num_tokens)
    q_proj_rope(query_latent, wq_b, wq_b_scale, rope_cos, rope_sin, query, num_tokens)
    kv_proj_rope(x, wkv, wkv_scale, kv_norm_weight, rope_cos, rope_sin, window_kv, num_tokens)
    return query_latent, query, window_kv


def make_qkv_proj_rope_with_deps(
    project_a,
    project_b,
    rotate_query,
    project_kv,
    rotate_kv,
):
    """Compose dependency-aware C1A decode primitives behind the shared stage boundary."""

    @pl.jit.inline(auto_scope=False)
    def qkv_proj_rope_with_deps(
        x: pl.Tensor[[T_DYN, D], pl.BF16],
        wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
        q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
        rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        query_latent: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
        query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
        window_kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
        ready: pl.Scalar[pl.TASK_ID],
    ):
        tokens = pl.tensor.dim(x, 0)
        projected_q = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
        qa_tid = project_a(x, wq_a, wq_a_scale, projected_q, num_tokens, ready)
        norm_blocks = (num_tokens + 7) // 8
        with pl.spmd(norm_blocks, name_hint="c1a_q_rmsnorm", deps=[qa_tid]) as qr_tid:
            norm_worker = pl.tile.get_block_idx()
            rms_norm(projected_q, q_norm_weight, query_latent, norm_worker, num_tokens, width=Q_LORA)
        expanded_q = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
        qb_tid = project_b(query_latent, wq_b, wq_b_scale, expanded_q, num_tokens, qr_tid)
        q_tid = rotate_query(expanded_q, rope_cos, rope_sin, query, num_tokens, qb_tid)
        projected_kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
        kv_tid = project_kv(x, wkv, wkv_scale, projected_kv, num_tokens, ready)
        normalized_kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
        norm_blocks = (num_tokens + 7) // 8
        with pl.spmd(norm_blocks, name_hint="c1a_kv_rmsnorm", deps=[kv_tid]) as kvn_tid:
            norm_worker = pl.tile.get_block_idx()
            rms_norm(projected_kv, kv_norm_weight, normalized_kv, norm_worker, num_tokens, width=HEAD_DIM)
        kvr_tid = rotate_kv(normalized_kv, rope_cos, rope_sin, window_kv, num_tokens, kvn_tid)
        return qr_tid, q_tid, kvr_tid

    return qkv_proj_rope_with_deps


_project_qa = make_mx_projection(D, Q_LORA, name_hint="attention_q_a")
_project_qb = make_mx_projection(Q_LORA, LOCAL_H * HEAD_DIM, name_hint="attention_q_b")
_project_kv = make_mx_projection(D, HEAD_DIM, name_hint="attention_kv")
_rotate_q = make_rope(LOCAL_H, name_hint="attention_q_rope")
_rotate_kv = make_rope(1, name_hint="attention_kv_rope")

_prefill_rotate_q = make_rope(
    LOCAL_H, max_workers=_PREFILL_WORKERS, name_hint="prefill_attention_q_rope"
)
_prefill_rotate_kv = make_rope(
    1, max_workers=_PREFILL_WORKERS, name_hint="prefill_attention_kv_rope"
)

q_proj_qr = _make_q_proj_qr(_project_qa, 2**31 - 1, "attention_q_norm")
q_proj_rope = _make_q_proj_rope(_project_qb, _rotate_q)
kv_proj_rope = _make_kv_proj_rope(_project_kv, _rotate_kv)
prefill_q_proj_qr = _make_q_proj_qr(_prefill_project_qa, _PREFILL_WORKERS, "prefill_attention_q_norm")
prefill_q_proj_rope = _make_q_proj_rope(_project_qb, _prefill_rotate_q)
