# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Continuous-batch decode sliding-window attention for encoder layers 0 and 1."""

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from models.deepseek_v4_1_flash.attention_tp import decode_tp_output_all_reduce
from models.deepseek_v4_1_flash.attention_common import AttentionGoldenResult, golden_swa_attention
from models.deepseek_v4_1_flash.config import (
    D,
    DECODE_MAX_TOKENS,
    FLASH,
    HEAD_DIM,
    LOCAL_H,
    LOCAL_O_GROUPS,
    LOCAL_O_WIDTH,
    NOPE_DIM,
    O_GROUP_IN,
    O_LORA,
    ORI_BLOCKS_DYN,
    Q_LORA,
    ROPE_DIM,
    T_DYN,
    TP_SIZE,
    WINDOW_CACHE_GROUP,
)


# Model configuration.
EPS = FLASH.rms_norm_eps
SOFTMAX_SCALE = HEAD_DIM ** -0.5

# Tiling. The mixed FP8 V2C pipe requires 32 physical rows on the pinned toolchain.
M_TILE = 16
MX_M_TILE = 32
N_TILE = 128
K_TILE = 256

def make_projection(width, output_width, output_dtype=pl.BF16):
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
            for block in pl.spmd(output_width // N_TILE, name_hint="swa_mx_projection"):
                n0 = block * N_TILE
                rows = pl.min(MX_M_TILE, num_tokens - t0)
                first = pl.load(x, [t0, 0], [MX_M_TILE, K_TILE], valid_shape=[rows, K_TILE])
                first = pl.set_validshape(pl.fillpad(first, pad_value=pl.PadValue.zero), MX_M_TILE, K_TILE)
                # s = 2**ceil(log2(max(amax, 1e-4) / 448)); native quant_mx uses a different rule.
                firstq_values = pl.reshape(pl.cast(first, pl.FP32), [MX_M_TILE * (K_TILE // 32), 32])
                firstq_reduce_tmp = pl.create_tile([MX_M_TILE * (K_TILE // 32), 32], dtype=pl.FP32)
                firstq_maximum = pl.maximum(pl.row_max(pl.abs(firstq_values), tmp_tile=firstq_reduce_tmp), 1e-4)
                firstq_bits = pl.reinterpret_view(pl.mul(firstq_maximum, 1.0 / 448.0), pl.INT32)
                firstq_exponent = pl.shrs(pl.add(firstq_bits, 8388607), 23)
                firstq_scale = pl.reinterpret_view(pl.shls(firstq_exponent, 23), pl.FP32)
                firstq_quantized = pl.cast(pl.row_expand_div(firstq_values, firstq_scale), pl.FP8E4M3FN, mode="rint")
                firstq_payload = pl.reshape(firstq_quantized, [MX_M_TILE, K_TILE])
                firstq_signed_exponent = pl.sub(firstq_exponent, pl.mul(pl.shrs(firstq_exponent, 7), 256))
                firstq_codes = pl.reinterpret_view(pl.cast(firstq_signed_exponent, pl.INT8), pl.UINT8)
                firstq_flat = pl.reshape(firstq_codes, [1, MX_M_TILE * (K_TILE // 32)])
                firstq_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                firstq_packed = pl.tmov_x2zz(firstq_flat, firstq_tmp, group_axis=1, dst_rows=MX_M_TILE, dst_cols=8)
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
                    nextq_maximum = pl.maximum(pl.row_max(pl.abs(nextq_values), tmp_tile=nextq_reduce_tmp), 1e-4)
                    nextq_bits = pl.reinterpret_view(pl.mul(nextq_maximum, 1.0 / 448.0), pl.INT32)
                    nextq_exponent = pl.shrs(pl.add(nextq_bits, 8388607), 23)
                    nextq_scale = pl.reinterpret_view(pl.shls(nextq_exponent, 23), pl.FP32)
                    nextq_quantized = pl.cast(pl.row_expand_div(nextq_values, nextq_scale), pl.FP8E4M3FN, mode="rint")
                    nextq_payload = pl.reshape(nextq_quantized, [MX_M_TILE, K_TILE])
                    nextq_signed_exponent = pl.sub(nextq_exponent, pl.mul(pl.shrs(nextq_exponent, 7), 256))
                    nextq_codes = pl.reinterpret_view(pl.cast(nextq_signed_exponent, pl.INT8), pl.UINT8)
                    nextq_flat = pl.reshape(nextq_codes, [1, MX_M_TILE * (K_TILE // 32)])
                    nextq_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                    nextq_packed = pl.tmov_x2zz(nextq_flat, nextq_tmp, group_axis=1, dst_rows=MX_M_TILE, dst_cols=8)
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


def make_norm(width):
    @pl.jit.inline
    def normalize(
        x: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width], pl.BF16],
        output: pl.Tensor[[T_DYN, width], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd((num_tokens + 7) // 8, name_hint="swa_rmsnorm"):
            t = block * 8
            rows = pl.min(8, num_tokens - t)
            source = pl.slice(x, [8, width], [t, 0], valid_shape=[rows, width])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 8, width)
            value = pl.cast(source, pl.FP32)
            inv = pl.rsqrt(pl.add(pl.mul(pl.row_sum(pl.mul(value, value)), 1.0 / width),
                                 EPS), high_precision=True)
            gamma = pl.reshape(pl.cast(weight[:], pl.FP32), [1, width])
            normalized = pl.col_expand_mul(pl.row_expand_mul(value, inv), gamma)
            output[t:t + 8, :] = pl.set_validshape(pl.cast(normalized, pl.BF16, mode="rint"), rows, width)
        return output

    return normalize


def make_rope(heads, inverse=False):
    sign = -1.0 if inverse else 1.0

    @pl.jit.inline
    def rotate(
        x: pl.Tensor[[T_DYN, heads * HEAD_DIM], pl.BF16],
        cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        output: pl.Tensor[[T_DYN, heads * HEAD_DIM], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(num_tokens * heads, name_hint="swa_rope"):
            t = block // heads
            h = block % heads
            base = h * HEAD_DIM
            output[t:t + 1, base:base + NOPE_DIM] = x[t:t + 1, base:base + NOPE_DIM]
            tail = pl.cast(x[t:t + 1, base + NOPE_DIM:base + HEAD_DIM], pl.FP32)
            even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            c = cos[t:t + 1, :]
            s = pl.mul(sin[t:t + 1, :], sign)
            re = pl.sub(pl.mul(even, c), pl.mul(odd, s))
            im = pl.add(pl.mul(even, s), pl.mul(odd, c))
            rotated = pl.full([1, ROPE_DIM], dtype=pl.FP32, value=0.0)
            rotated = pl.tensor.scatter(re, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
            rotated = pl.tensor.scatter(im, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
            output[t:t + 1, base + NOPE_DIM:base + HEAD_DIM] = pl.cast(rotated, pl.BF16, mode="rint")
        return output

    return rotate


@pl.jit.inline
def publish_window(
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
    with pl.spmd(num_tokens, name_hint="swa_cache_publish", deps=[cache_ready]) as publish_tid:
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
            scale_flat[slot:slot + 1, :] = pl.set_validshape(pl.reshape(encoded, [1, HEAD_DIM // 16]),
                                                         1, HEAD_DIM // 32)
    return cache, scales


@pl.jit.inline
def gather_window(
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.FP8E8M0],
    indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    selected: pl.Tensor[[T_DYN, 128, HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    blocks = pl.tensor.dim(cache, 0)
    cache_rows = blocks * 128
    flat = pl.reshape(cache, [cache_rows, HEAD_DIM])
    scale_flat = pl.reshape(scales, [cache_rows, HEAD_DIM // 32])
    tokens = pl.tensor.dim(selected, 0)
    selected_rows = tokens * 128
    gathered = pl.reshape(selected, [selected_rows, HEAD_DIM])
    with pl.spmd(num_tokens * 8, name_hint="swa_cache_gather") as gather_tid:
        block = pl.tile.get_block_idx()
        t = block // 8
        for i in pl.range(block % 8 * 16, block % 8 * 16 + 16):
            row_i32 = pl.read(indices, [t, i])
            dst = t * 128 + i
            if row_i32 >= 0:
                row = pl.cast(row_i32, pl.INDEX)
                value = pl.reshape(pl.cast(flat[row:row + 1, :], pl.FP32), [HEAD_DIM // 32, 32])
                scale_row = pl.slice(scale_flat, [1, 32], [row, 0], valid_shape=[1, HEAD_DIM // 32])
                raw_codes = pl.reinterpret_view(scale_row, pl.UINT8)
                signed_codes = pl.cast(pl.reinterpret_view(raw_codes, pl.INT8), pl.INT32)
                codes = pl.ands(signed_codes, 255)
                scale_values = pl.reinterpret_view(pl.maximum(pl.shls(codes, 23), 4194304), pl.FP32)
                scale = pl.reshape(scale_values[:, :HEAD_DIM // 32], [HEAD_DIM // 32, 1])
                decoded = pl.cast(pl.row_expand_mul(value, scale), pl.BF16, mode="rint")
                gathered[dst:dst + 1, :] = pl.reshape(decoded, [1, HEAD_DIM])
            else:
                gathered[dst:dst + 1, :] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
    return gather_tid


@pl.jit.inline
def attend_window(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    selected: pl.Tensor[[T_DYN, 128, HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    tokens = pl.tensor.dim(query, 0)
    head_rows = tokens * LOCAL_H
    cache_rows = tokens * 128
    qflat = pl.reshape(query, [head_rows, HEAD_DIM])
    kflat = pl.reshape(selected, [cache_rows, HEAD_DIM])
    oflat = pl.reshape(output, [head_rows, HEAD_DIM])
    for block in pl.spmd(num_tokens * (LOCAL_H // M_TILE), name_hint="swa_online_attention"):
        t = block // (LOCAL_H // M_TILE)
        h = block % (LOCAL_H // M_TILE) * M_TILE
        q0 = t * LOCAL_H + h
        maximum = pl.full([1, M_TILE], dtype=pl.FP32, value=-1e30)
        denominator = pl.full([1, M_TILE], dtype=pl.FP32, value=0.0)
        numerator = pl.full([M_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
        for part in pl.range(2):
            k0 = t * 128 + part * 64
            q = qflat[q0:q0 + M_TILE, :]
            kv = kflat[k0:k0 + 64, :]
            scores = pl.matmul(q, kv, b_trans=True)
            scores = pl.mul(scores, SOFTMAX_SCALE)
            idx = pl.cast(indices[t:t + 1, part * 64:part * 64 + 64], pl.FP32)
            valid = pl.minimum(pl.maximum(pl.add(idx, 1.0), 0.0), 1.0)
            bias = pl.mul(pl.sub(valid, 1.0), 1e30)
            scores = pl.col_expand_add(scores, bias)
            next_max = pl.maximum(maximum, pl.reshape(pl.row_max(scores), [1, M_TILE]))
            correction = pl.exp(pl.sub(maximum, next_max))
            probabilities = pl.col_expand_mul(
                pl.exp(pl.row_expand_sub(scores, pl.reshape(next_max, [M_TILE, 1]))), valid)
            denominator = pl.add(pl.mul(denominator, correction),
                                 pl.reshape(pl.row_sum(probabilities), [1, M_TILE]))
            weights = pl.cast(probabilities, pl.BF16, mode="rint")
            weighted = pl.matmul(weights, kv)
            numerator = pl.add(pl.row_expand_mul(numerator, pl.reshape(correction, [M_TILE, 1])), weighted)
            maximum = next_max
        sinks = pl.reshape(sink[h:h + M_TILE], [1, M_TILE])
        final_max = pl.maximum(maximum, sinks)
        correction = pl.exp(pl.sub(maximum, final_max))
        denominator = pl.add(pl.mul(denominator, correction), pl.exp(pl.sub(sinks, final_max)))
        result = pl.row_expand_mul(numerator, pl.reshape(pl.div(correction, denominator), [M_TILE, 1]))
        oflat[q0:q0 + M_TILE, :] = pl.cast(result, pl.BF16, mode="rint")
    return output


@pl.jit.inline
def grouped_output(
    x: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    weight: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    output: pl.Tensor[[T_DYN, LOCAL_O_WIDTH], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    for block in pl.spmd((num_tokens + M_TILE - 1) // M_TILE * (LOCAL_O_WIDTH // N_TILE),
                         name_hint="swa_grouped_output"):
        t0 = block // (LOCAL_O_WIDTH // N_TILE) * M_TILE
        n0 = block % (LOCAL_O_WIDTH // N_TILE) * N_TILE
        group = n0 // O_LORA
        local_n = n0 % O_LORA
        rows = pl.min(M_TILE, num_tokens - t0)
        acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.FP32)
        for kb in pl.range(O_GROUP_IN // K_TILE):
            k0 = kb * K_TILE
            a = pl.slice(x, [M_TILE, K_TILE], [t0, group * O_GROUP_IN + k0],
                         valid_shape=[rows, K_TILE])
            w = pl.reshape(weight[group:group + 1, local_n:local_n + N_TILE, k0:k0 + K_TILE],
                           [N_TILE, K_TILE])
            acc = pl.matmul_acc(acc, a, w, b_trans=True, init_cond=(kb == 0))
        value = pl.cast(acc, pl.BF16, mode="rint")
        output[t0:t0 + M_TILE, n0:n0 + N_TILE] = pl.set_validshape(value, rows, N_TILE)
    return output


if TP_SIZE not in (1, 2, 4):
    raise ValueError("Decode SWA currently supports TP1, TP2, and TP4; TP8 requires head-tile padding")

project_qa = make_projection(D, Q_LORA)
project_qb = make_projection(Q_LORA, LOCAL_H * HEAD_DIM)
project_kv = make_projection(D, HEAD_DIM)
project_ob = make_projection(LOCAL_O_WIDTH, D, pl.FP32)
normalize_q = make_norm(Q_LORA)
normalize_kv = make_norm(HEAD_DIM)
rotate_q = make_rope(LOCAL_H)
rotate_kv = make_rope(1)
rotate_output = make_rope(LOCAL_H, inverse=True)


def golden_decode_swa(
    x: torch.Tensor,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    attn_sink: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    window_slots: torch.Tensor,
    window_indices: torch.Tensor,
    window_cache: torch.Tensor,
    window_cache_scale: torch.Tensor,
) -> AttentionGoldenResult:
    return golden_swa_attention(
        x,
        wq_a,
        wq_a_scale,
        q_norm_weight,
        wq_b,
        wq_b_scale,
        wkv,
        wkv_scale,
        kv_norm_weight,
        attn_sink,
        wo_a,
        wo_b,
        wo_b_scale,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache,
        window_cache_scale,
    )


@pl.jit.inline(auto_scope=False)
def decode_swa_partial(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.FP8E8M0],
    output: pl.Tensor[[T_DYN, D], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    """Write the FP32 local output; return the cache-read completion task for caller WAR ordering."""
    tokens = pl.tensor.dim(x, 0)
    qa = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
    project_qa(x, wq_a, wq_a_scale, qa, num_tokens)
    qr = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
    normalize_q(qa, q_norm_weight, qr, num_tokens)
    qb = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    project_qb(qr, wq_b, wq_b_scale, qb, num_tokens)
    q = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    rotate_q(qb, rope_cos, rope_sin, q, num_tokens)
    kv_projection = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    project_kv(x, wkv, wkv_scale, kv_projection, num_tokens)
    kv_normalized = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    normalize_kv(kv_projection, kv_norm_weight, kv_normalized, num_tokens)
    kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    rotate_kv(kv_normalized, rope_cos, rope_sin, kv, num_tokens)
    publish_window(kv, window_slots, window_cache, window_cache_scale, num_tokens, cache_ready)
    selected = pl.create_tensor([tokens, 128, HEAD_DIM], dtype=pl.BF16)
    cache_consumed = gather_window(window_cache, window_cache_scale, window_indices, selected, num_tokens)
    attended = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    attend_window(q, selected, window_indices, attn_sink, attended, num_tokens)
    unrotated = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    rotate_output(attended, rope_cos, rope_sin, unrotated, num_tokens)
    latent = pl.create_tensor([tokens, LOCAL_O_WIDTH], dtype=pl.BF16)
    grouped_output(unrotated, wo_a, latent, num_tokens)
    project_ob(latent, wo_b, wo_b_scale, output, num_tokens)
    return cache_consumed


@pl.jit.inline(auto_scope=False)
def decode_swa(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.FP8E8M0],
    output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Write BF16 TP output using zero-initialized windows and consecutive 1-based epochs."""
    # A later epoch must not overwrite cache or transport storage still being read.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_previous_epoch", allow_early_resolve=False) as cache_ready:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=(attention_epoch - 1) * 2,
                            cmp=pld.WaitCmp.Ge)
    tokens = pl.tensor.dim(x, 0)
    partial = pl.create_tensor([tokens, D], dtype=pl.FP32)
    decode_swa_partial(x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
                       wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_b, wo_b_scale,
                       rope_cos, rope_sin, window_slots, window_indices, window_cache,
                       window_cache_scale, partial, num_tokens, cache_ready)
    decode_tp_output_all_reduce(partial, output_window, output_arrived, output,
                                group_base, tp_rank, num_tokens, attention_epoch)
    return output


__all__ = ["decode_swa", "golden_decode_swa"]


if __name__ == "__main__":
    from models.deepseek_v4_1_flash._golden_smoke import run_attention_golden

    run_attention_golden(golden_decode_swa, ratio=0, mode="swa")
