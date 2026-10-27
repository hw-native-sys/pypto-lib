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

from config import FLASH as M, T_DYN, TP_SIZE
from gemm import (
    MX_M_TILE, MX_K_TILE,
    gemm_bf16_via_mxfp8_to_bf16_staged, gemm_mxfp8_to_bf16,
    golden_gemm_bf16_via_mxfp8_to_bf16_sequential, golden_mxfp8_linear_grouped_sum,
)
from quant import QUANT_SCALE_TILE, quantize_bf16_to_mxfp8
from rmsnorm import golden_rms_norm, rms_norm


# model config
LOCAL_H = M.num_attention_heads // TP_SIZE
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
EPS = M.rms_norm_eps

# tiling
N_TILE = 128
K_TILE = 256
Q_A_N_TILE = 128
Q_A_K_TILE = 256


@pl.jit.inline
def rope(
    x: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    heads: pl.constexpr,
    name_hint: pl.constexpr,
    max_workers: pl.constexpr,
):
    blocks = num_tokens * heads
    workers = pl.min(blocks, max_workers)
    for worker in pl.spmd(workers, name_hint=name_hint):
        for block in pl.range(worker, blocks, workers):
            t = block // heads
            h = block % heads
            base = h * HEAD_DIM
            output[t : t + 1, base : base + (HEAD_DIM - ROPE_DIM)] = x[t : t + 1, base : base + (HEAD_DIM - ROPE_DIM)]
            tail = pl.cast(x[t : t + 1, base + (HEAD_DIM - ROPE_DIM) : base + HEAD_DIM], pl.FP32)
            even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            c = cos[t : t + 1, :]
            s = pl.mul(sin[t : t + 1, :], 1.0)
            re = pl.sub(pl.mul(even, c), pl.mul(odd, s))
            im = pl.add(pl.mul(even, s), pl.mul(odd, c))
            rotated = pl.full([1, ROPE_DIM], dtype=pl.FP32, value=0.0)
            rotated = pl.tensor.scatter(re, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
            rotated = pl.tensor.scatter(im, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
            output[t : t + 1, base + (HEAD_DIM - ROPE_DIM) : base + HEAD_DIM] = pl.cast(rotated, pl.BF16, mode="rint")
    return output


@pl.jit.inline(auto_scope=False)
def _quantize_projection_input(
    x: pl.Tensor,
    payload: pl.Tensor,
    scale_backing: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    width: pl.constexpr,
    name_hint: pl.constexpr,
):
    """Quantize one M16 projection input into shared MX payload and packed scales."""
    for block in pl.spmd(width // MX_K_TILE, name_hint=name_hint):
        k0 = block * MX_K_TILE
        rows = pl.min(MX_M_TILE, num_tokens)
        quantized_tile, packed_scale = quantize_bf16_to_mxfp8(x, 0, k0, rows, MX_M_TILE)
        payload = pl.store(quantized_tile, [0, k0], payload)
        scale_offset = block * MX_M_TILE * (MX_K_TILE // 32)
        packed_flat = pl.reshape(packed_scale, [1, MX_M_TILE * (MX_K_TILE // 32)])
        scale_backing = pl.store(packed_flat, [0, scale_offset], scale_backing)
    return payload, scale_backing


@pl.jit.inline(auto_scope=False)
def _q_normalize(
    projected: pl.Tensor,
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    query_latent: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    norm_blocks = (num_tokens + 7) // 8
    for norm_worker in pl.spmd(norm_blocks, name_hint="norm_q"):
        rms_norm(projected, q_norm_weight, query_latent, norm_worker, num_tokens, width=Q_LORA)
    return query_latent


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
    gemm_bf16_via_mxfp8_to_bf16_staged(
        x, wq_a, wq_a_scale, projected, num_tokens,
        D, Q_LORA, "proj_q_a", "quant_q_a", Q_A_N_TILE, Q_A_K_TILE, 1, 2147483647,
    )
    _q_normalize(projected, q_norm_weight, query_latent, num_tokens)
    return query_latent


@pl.jit.inline(auto_scope=False)
def _q_proj_rope_from_mx_m80(
    payload: pl.Tensor,
    scale_backing: pl.Tensor,
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    q_b_n_tile: pl.constexpr,
    max_workers: pl.constexpr = 2147483647,
):
    """Project 80 Q-B rows and apply M16 RoPE epilogues."""
    gemm_mxfp8_to_bf16(
        payload, scale_backing, wq_b, wq_b_scale, query,
        num_tokens, 0, 0, 0, Q_LORA, LOCAL_H * HEAD_DIM,
        "proj_q_b_ragged", q_b_n_tile, K_TILE, 1, 0, 0, 80, max_workers, 4,
    )
    rope_workers = 2 * pl.system.available_cluster_count()
    for worker in pl.spmd(rope_workers, name_hint="attention_q_rope"):
        for block in pl.range(worker, (80 // MX_M_TILE) * LOCAL_H, rope_workers):
            t0 = block // LOCAL_H * MX_M_TILE
            n0 = block % LOCAL_H * HEAD_DIM
            rows = pl.max(pl.min(MX_M_TILE, num_tokens - t0), 0)
            tail_bf16 = pl.load(query, [t0, n0 + HEAD_DIM - ROPE_DIM], [MX_M_TILE, ROPE_DIM], valid_shape=[rows, ROPE_DIM])
            tail = pl.cast(tail_bf16, pl.FP32)
            even = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            cosine = pl.load(rope_cos, [t0, 0], [MX_M_TILE, ROPE_DIM // 2], valid_shape=[rows, ROPE_DIM // 2])
            sine = pl.load(rope_sin, [t0, 0], [MX_M_TILE, ROPE_DIM // 2], valid_shape=[rows, ROPE_DIM // 2])
            cosine = pl.fillpad(cosine, pad_value=pl.PadValue.zero)
            cosine = pl.set_validshape(cosine, MX_M_TILE, ROPE_DIM // 2)
            sine = pl.fillpad(sine, pad_value=pl.PadValue.zero)
            sine = pl.set_validshape(sine, MX_M_TILE, ROPE_DIM // 2)
            sine = pl.mul(sine, 1.0)
            rope_even_cos = pl.mul(even, cosine)
            rope_odd_sin = pl.mul(odd, sine)
            re = pl.sub(rope_even_cos, rope_odd_sin)
            rope_even_sin = pl.mul(even, sine)
            rope_odd_cos = pl.mul(odd, cosine)
            im = pl.add(rope_even_sin, rope_odd_cos)
            real_lanes = pl.tile.full([MX_M_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
            real_lanes = pl.tile.scatter_mask(real_lanes, re, mask_pattern=pl.tile.MaskPattern.P0101)
            imag_lanes = pl.tile.full([MX_M_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
            imag_lanes = pl.tile.scatter_mask(imag_lanes, im, mask_pattern=pl.tile.MaskPattern.P1010)
            real_bits = pl.reinterpret_view(real_lanes, pl.UINT32)
            imag_bits = pl.reinterpret_view(imag_lanes, pl.UINT32)
            rotated_bits = pl.or_(real_bits, imag_bits)
            rotated = pl.reinterpret_view(rotated_bits, pl.FP32)
            rounded = pl.cast(rotated, pl.BF16, mode="rint")
            valid = pl.set_validshape(rounded, rows, ROPE_DIM)
            query = pl.store(valid, [t0, n0 + HEAD_DIM - ROPE_DIM], query)
    return query


@pl.jit.inline(auto_scope=False)
def q_proj_rope(
    query_latent: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    max_workers: pl.constexpr = 2147483647,
):
    tokens = pl.tensor.dim(query_latent, 0)
    expanded = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    gemm_bf16_via_mxfp8_to_bf16_staged(
        query_latent, wq_b, wq_b_scale, expanded, num_tokens,
        Q_LORA, LOCAL_H * HEAD_DIM, "proj_q_b", "quant_q_b", N_TILE, K_TILE, 1, 2147483647,
    )
    rope(expanded, rope_cos, rope_sin, query, num_tokens, LOCAL_H, "rope_q", max_workers)

    return query


@pl.jit.inline(auto_scope=False)
def _kv_norm_rope_fused(
    projected: pl.Tensor,
    weight: pl.Tensor,
    cosine: pl.Tensor,
    sine: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    weight_flat = pl.reshape(weight, [1, HEAD_DIM])
    for block in pl.spmd((num_tokens + 7) // 8, name_hint="kv_norm_rope_fused"):
        t0 = block * 8
        rows = pl.min(8, num_tokens - t0)
        source = pl.load(projected, [t0, 0], [8, HEAD_DIM], valid_shape=[rows, HEAD_DIM])
        source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 8, HEAD_DIM)
        value = pl.cast(source, pl.FP32)
        reduce_tmp = pl.create_tile([8, HEAD_DIM], dtype=pl.FP32)
        squares = pl.mul(value, value)
        sq_sum = pl.reshape(pl.row_sum(squares, tmp_tile=reduce_tmp), [1, 8])
        mean_square = pl.mul(sq_sum, 1.0 / HEAD_DIM)
        argument = pl.add(mean_square, EPS)
        rsqrt_tmp = pl.create_tile([1, 8], dtype=pl.FP32)
        inverse = pl.reshape(pl.tile.rsqrt(argument, tmp=rsqrt_tmp), [8, 1])
        gamma = pl.cast(pl.load(weight_flat, [0, 0], [1, HEAD_DIM]), pl.FP32)
        normalized = pl.col_expand_mul(pl.row_expand_mul(value, inverse), gamma)
        rounded = pl.cast(normalized, pl.BF16, mode="rint")
        prefix = pl.tile.slice(rounded, [8, HEAD_DIM - ROPE_DIM], [0, 0], valid_shape=[rows, HEAD_DIM - ROPE_DIM])
        pl.store(prefix, [t0, 0], output)
        tail = pl.cast(pl.tile.slice(rounded, [8, ROPE_DIM], [0, HEAD_DIM - ROPE_DIM]), pl.FP32)
        even = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P0101)
        odd = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P1010)
        c = pl.load(cosine, [t0, 0], [8, ROPE_DIM // 2], valid_shape=[rows, ROPE_DIM // 2])
        c = pl.set_validshape(pl.fillpad(c, pad_value=pl.PadValue.zero), 8, ROPE_DIM // 2)
        s = pl.load(sine, [t0, 0], [8, ROPE_DIM // 2], valid_shape=[rows, ROPE_DIM // 2])
        s = pl.set_validshape(pl.fillpad(s, pad_value=pl.PadValue.zero), 8, ROPE_DIM // 2)
        s = pl.mul(s, 1.0)
        real = pl.sub(pl.mul(even, c), pl.mul(odd, s))
        imag = pl.add(pl.mul(even, s), pl.mul(odd, c))
        real_lanes = pl.tile.full([8, ROPE_DIM], dtype=pl.FP32, value=0.0)
        real_lanes = pl.tile.scatter_mask(real_lanes, real, mask_pattern=pl.tile.MaskPattern.P0101)
        imag_lanes = pl.tile.full([8, ROPE_DIM], dtype=pl.FP32, value=0.0)
        imag_lanes = pl.tile.scatter_mask(imag_lanes, imag, mask_pattern=pl.tile.MaskPattern.P1010)
        real_bits = pl.reinterpret_view(real_lanes, pl.UINT32)
        imag_bits = pl.reinterpret_view(imag_lanes, pl.UINT32)
        rotated = pl.reinterpret_view(pl.or_(real_bits, imag_bits), pl.FP32)
        result = pl.set_validshape(pl.cast(rotated, pl.BF16, mode="rint"), rows, ROPE_DIM)
        pl.store(result, [t0, HEAD_DIM - ROPE_DIM], output)
    return output


@pl.jit.inline(auto_scope=False)
def _kv_normalize_rope(
    projected: pl.Tensor,
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    window_kv: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    tokens = pl.tensor.dim(projected, 0)
    normalized = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    norm_blocks = (num_tokens + 7) // 8
    for norm_worker in pl.spmd(norm_blocks, name_hint="norm_kv"):
        rms_norm(projected, kv_norm_weight, normalized, norm_worker, num_tokens, width=HEAD_DIM)
    rope(normalized, rope_cos, rope_sin, window_kv, num_tokens, 1, "rope_kv", 2147483647)
    return window_kv


@pl.jit.inline(auto_scope=False)
def kv_proj_rope(
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
    gemm_bf16_via_mxfp8_to_bf16_staged(
        x, wkv, wkv_scale, projected,
        num_tokens,
        D, HEAD_DIM, "proj_kv", "quant_kv", N_TILE, K_TILE, 1, 2147483647,
    )
    _kv_normalize_rope(projected, kv_norm_weight, rope_cos, rope_sin, window_kv, num_tokens)
    return window_kv


@pl.jit.inline(auto_scope=False)
def _q_normalize_to_mx(
    projected: pl.Tensor,
    q_norm_weight: pl.Tensor,
    query_latent: pl.Tensor,
    payload: pl.Tensor,
    scale_backing: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    workers: pl.constexpr,
):
    weight_row = pl.tensor.view(q_norm_weight, [1, Q_LORA])
    for worker in pl.spmd(workers, name_hint="attention_q_norm_quant"):
        normalized_rows = pl.create_tile([MX_M_TILE, Q_LORA], dtype=pl.BF16)
        for half in pl.unroll(2):
            t = half * 8
            rows = pl.max(pl.min(8, num_tokens - t), 0)
            zero_source = pl.tile.full([8, Q_LORA], dtype=pl.BF16, value=0.0)
            source = pl.fillpad(zero_source, pad_value=pl.PadValue.zero)
            if num_tokens > t:
                loaded = pl.load(projected, [t, 0], [8, Q_LORA], valid_shape=[rows, Q_LORA])
                source = pl.set_validshape(pl.fillpad(loaded, pad_value=pl.PadValue.zero), 8, Q_LORA)
            value = pl.cast(source, pl.FP32)
            squared = pl.mul(value, value)
            norm_tmp = pl.create_tile([8, Q_LORA], dtype=pl.FP32)
            row_sum = pl.row_sum(squared, tmp_tile=norm_tmp)
            mean = pl.mul(row_sum, 1.0 / Q_LORA)
            rms_argument = pl.add(mean, EPS)
            rsqrt_tmp = pl.create_tile([8, 1], dtype=pl.FP32)
            row_inverse = pl.tile.rsqrt(rms_argument, tmp=rsqrt_tmp)
            gamma = pl.cast(pl.load(weight_row, [0, 0], [1, Q_LORA]), pl.FP32)
            normalized = pl.col_expand_mul(pl.row_expand_mul(value, row_inverse), gamma)
            rounded = pl.cast(normalized, pl.BF16, mode="rint")
            rounded_valid = pl.set_validshape(rounded, rows, Q_LORA)
            if worker == 0 and num_tokens > t:
                query_latent = pl.store(rounded_valid, [t, 0], query_latent)
            padded = pl.set_validshape(pl.fillpad(rounded_valid, pad_value=pl.PadValue.zero), 8, Q_LORA)
            padded_bits = pl.reinterpret_view(padded, pl.INT16)
            zero_bits = pl.tile.full([8, Q_LORA], dtype=pl.INT16, value=0)
            dense_bits = pl.add(zero_bits, padded_bits)
            dense_rounded = pl.reinterpret_view(dense_bits, pl.BF16)
            normalized_rows = pl.tile.assemble(normalized_rows, dense_rounded, [t, 0])
        for k0 in pl.range(worker * MX_K_TILE, Q_LORA, workers * MX_K_TILE):
            quant_source = pl.tile.slice(normalized_rows, [MX_M_TILE, MX_K_TILE], [0, k0])
            source_fp32 = pl.cast(quant_source, pl.FP32)
            groups = pl.reshape(source_fp32, [MX_M_TILE * (MX_K_TILE // 32), 32])
            reduce_tmp = pl.create_tile([MX_M_TILE * (MX_K_TILE // 32), 32], dtype=pl.FP32)
            magnitude = pl.abs(groups)
            maximum = pl.row_max(magnitude, tmp_tile=reduce_tmp)
            bits = pl.reinterpret_view(maximum, pl.INT32)
            biased_exponent = pl.shrs(bits, 23)
            scale_exponent = pl.sub(biased_exponent, 8)
            exponent = pl.maximum(scale_exponent, 0)
            # Scale code zero has a zero reciprocal.
            negative_exponent = pl.mul(exponent, -1)
            reciprocal_exponent = pl.add(negative_exponent, 254)
            nonzero_exponent = pl.minimum(exponent, 1)
            reciprocal_code = pl.mul(reciprocal_exponent, nonzero_exponent)
            reciprocal_bits = pl.shls(reciprocal_code, 23)
            reciprocal = pl.reinterpret_view(reciprocal_bits, pl.FP32)
            quant_normalized = pl.row_expand_mul(groups, reciprocal)
            quant_normalized = pl.maximum(quant_normalized, -448.0)
            quant_normalized = pl.minimum(quant_normalized, 448.0)
            payload_fp8 = pl.cast(quant_normalized, pl.FP8E4M3FN, mode="rint")
            quantized_tile = pl.reshape(payload_fp8, [MX_M_TILE, MX_K_TILE])
            sign_bit = pl.shrs(exponent, 7)
            sign_offset = pl.mul(sign_bit, 256)
            signed = pl.sub(exponent, sign_offset)
            signed_codes = pl.cast(signed, pl.INT8)
            codes = pl.reinterpret_view(signed_codes, pl.UINT8)
            flat = pl.reshape(codes, [1, MX_M_TILE * (MX_K_TILE // 32)])
            pack_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
            packed = pl.tmov_x2zz(flat, pack_tmp, group_axis=1, dst_rows=MX_M_TILE, dst_cols=QUANT_SCALE_TILE)
            packed_scale = pl.reinterpret_view(packed, pl.FP8E8M0)
            payload = pl.store(quantized_tile, [0, k0], payload)
            scale_offset = k0 // MX_K_TILE * MX_M_TILE * (MX_K_TILE // 32)
            packed_flat = pl.reshape(packed_scale, [1, MX_M_TILE * (MX_K_TILE // 32)])
            scale_backing = pl.store(packed_flat, [0, scale_offset], scale_backing)
    return query_latent, payload, scale_backing


@pl.jit.inline(auto_scope=False)
def _qa_kv_ragged_group(
    payload_all: pl.Tensor,
    scales_all: pl.Tensor,
    wq_a: pl.Tensor,
    wq_a_scale: pl.Tensor,
    q_norm_weight: pl.Tensor,
    wkv: pl.Tensor,
    wkv_scale: pl.Tensor,
    kv_norm_weight: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    query_latent: pl.Tensor,
    kv_projection: pl.Tensor,
    window_kv: pl.Tensor,
    qb_payload_all: pl.Tensor,
    qb_scales_all: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    group_start: pl.Scalar[pl.INDEX],
    m_tile: pl.constexpr,
    fuse_qnorm: pl.constexpr,
    fuse_kv_norm_rope: pl.constexpr,
):
    """Project one Q-A/KV row group and apply normalization and RoPE epilogues."""
    group_rows = pl.max(pl.min(m_tile, num_tokens - group_start), 0)
    group_payload = pl.tensor.slice(payload_all, [m_tile, D], [group_start, 0])
    group_scales = pl.tensor.slice(
        scales_all, [1, m_tile * (D // 32)], [0, group_start * (D // 32)],
    )
    projected = pl.create_tensor([m_tile, Q_LORA], dtype=pl.BF16)
    gemm_mxfp8_to_bf16(
        group_payload, group_scales, wq_a, wq_a_scale, projected,
        group_rows, 0, 0, 0,
        D, Q_LORA, "proj_q_a_ragged", Q_A_N_TILE, Q_A_K_TILE, 1, 0, 0, m_tile, 2147483647, 2,
    )
    for q_mt in pl.parallel(m_tile // MX_M_TILE):
        q_local = q_mt * MX_M_TILE
        q_start = group_start + q_local
        q_rows = pl.max(pl.min(MX_M_TILE, group_rows - q_local), 0)
        projected_tile = pl.slice(projected, [q_rows, Q_LORA], [q_local, 0])
        latent_tile = pl.slice(query_latent, [q_rows, Q_LORA], [q_start, 0])
        qb_payload_tile = pl.tensor.slice(qb_payload_all, [MX_M_TILE, Q_LORA], [q_start, 0])
        qb_scales_tile = pl.tensor.slice(
            qb_scales_all, [1, MX_M_TILE * (Q_LORA // 32)], [0, q_start * (Q_LORA // 32)],
        )
        if fuse_qnorm:
            _q_normalize_to_mx(
                projected_tile, q_norm_weight, latent_tile,
                qb_payload_tile, qb_scales_tile, q_rows, 5,
            )
        else:
            _q_normalize(projected_tile, q_norm_weight, latent_tile, q_rows)
            qb_payload_tile, qb_scales_tile = _quantize_projection_input(
                latent_tile, qb_payload_tile, qb_scales_tile, q_rows, Q_LORA, "quant_q_b",
            )
    projection_group = pl.slice(kv_projection, [group_rows, HEAD_DIM], [group_start, 0])
    gemm_mxfp8_to_bf16(
        group_payload, group_scales, wkv, wkv_scale, projection_group,
        group_rows, 0, 0, 0,
        D, HEAD_DIM, "proj_kv_ragged", N_TILE, K_TILE, 1, 0, 0, m_tile, 2147483647, 2,
    )
    if fuse_kv_norm_rope:
        window_group = pl.slice(window_kv, [group_rows, HEAD_DIM], [group_start, 0])
        cosine_group = pl.slice(rope_cos, [group_rows, ROPE_DIM // 2], [group_start, 0])
        sine_group = pl.slice(rope_sin, [group_rows, ROPE_DIM // 2], [group_start, 0])
        _kv_norm_rope_fused(
            projection_group, kv_norm_weight, cosine_group, sine_group, window_group, group_rows,
        )
    else:
        for kv_mt in pl.parallel(m_tile // MX_M_TILE):
            kv_local = kv_mt * MX_M_TILE
            kv_start = group_start + kv_local
            kv_rows = pl.max(pl.min(MX_M_TILE, group_rows - kv_local), 0)
            projection_tile = pl.slice(kv_projection, [kv_rows, HEAD_DIM], [kv_start, 0])
            window_tile = pl.slice(window_kv, [kv_rows, HEAD_DIM], [kv_start, 0])
            cosine_tile = pl.slice(rope_cos, [kv_rows, ROPE_DIM // 2], [kv_start, 0])
            sine_tile = pl.slice(rope_sin, [kv_rows, ROPE_DIM // 2], [kv_start, 0])
            _kv_normalize_rope(
                projection_tile, kv_norm_weight, cosine_tile, sine_tile, window_tile, kv_rows,
            )


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
    kv_projection: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    window_kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    max_workers: pl.constexpr = 2147483647,
    fuse_qnorm: pl.constexpr = True,
    q_b_n_tile: pl.constexpr = N_TILE,
    q_b_max_workers: pl.constexpr = 2147483647,
    fuse_kv_norm_rope: pl.constexpr = False,
):
    """Project 72 active decode rows with shared M80 input quantization."""
    input_payload_all = pl.create_tensor([80, D], dtype=pl.FP8E4M3FN)
    input_scales_all = pl.create_tensor([1, 80 * (D // 32)], dtype=pl.FP8E8M0)
    qb_payload_all = pl.create_tensor([80, Q_LORA], dtype=pl.FP8E4M3FN)
    qb_scales_all = pl.create_tensor([1, 80 * (Q_LORA // 32)], dtype=pl.FP8E8M0)
    quant_blocks = (80 // MX_M_TILE) * (D // MX_K_TILE)
    quant_workers = pl.min(pl.min(quant_blocks, 64), 2 * pl.system.available_cluster_count())
    for worker in pl.spmd(quant_workers, name_hint="attention_qkv_input_quant_pooled"):
        for work in pl.range(worker, quant_blocks, quant_workers):
            t0 = (work // (D // MX_K_TILE)) * MX_M_TILE
            kt = work % (D // MX_K_TILE)
            k0 = kt * MX_K_TILE
            rows = pl.max(pl.min(MX_M_TILE, num_tokens - t0), 0)
            quantized_tile, packed_scale = quantize_bf16_to_mxfp8(x, t0, k0, rows, MX_M_TILE)
            input_payload_all = pl.store(quantized_tile, [t0, k0], input_payload_all)
            scale_offset = t0 * (D // 32) + kt * MX_M_TILE * (MX_K_TILE // 32)
            packed_flat = pl.reshape(packed_scale, [1, MX_M_TILE * (MX_K_TILE // 32)])
            input_scales_all = pl.store(packed_flat, [0, scale_offset], input_scales_all)
    _qa_kv_ragged_group(
        input_payload_all, input_scales_all, wq_a, wq_a_scale, q_norm_weight,
        wkv, wkv_scale, kv_norm_weight, rope_cos, rope_sin,
        query_latent, kv_projection, window_kv, qb_payload_all, qb_scales_all,
        num_tokens, 0, 80, fuse_qnorm, fuse_kv_norm_rope,
    )
    _q_proj_rope_from_mx_m80(
        qb_payload_all, qb_scales_all, wq_b, wq_b_scale, rope_cos, rope_sin,
        query, num_tokens, q_b_n_tile, q_b_max_workers,
    )
    return query_latent, query, kv_projection, window_kv


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


def golden_rope_tail(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Rotate the trailing 2*cos.shape[-1] lanes as adjacent (real, imag) pairs."""
    rope_dim = cos.shape[-1] * 2
    rotated = golden_rope_interleave(x[..., -rope_dim:], cos, sin, inverse)
    return torch.cat((x[..., :-rope_dim], rotated), dim=-1)


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
    *,
    sequential_query: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project normalized query latent and shared KV, then rotate their RoPE tails."""
    def project(value, weight, scale):
        if sequential_query and scale is not None:
            return golden_gemm_bf16_via_mxfp8_to_bf16_sequential(value, weight, scale)
        return golden_mxfp8_linear_grouped_sum(value, weight, scale)

    if wq_a_scale is not None:
        projected = golden_gemm_bf16_via_mxfp8_to_bf16_sequential(x, wq_a, wq_a_scale)
    else:
        projected = golden_mxfp8_linear_grouped_sum(x, wq_a, wq_a_scale)
    qr = golden_rms_norm(projected, q_norm_weight)
    head_dim = wkv.shape[-1]
    num_heads = wq_b.shape[-1] // head_dim
    q = project(qr, wq_b, wq_b_scale).unflatten(-1, (num_heads, head_dim))
    kv = golden_rms_norm(project(x, wkv, wkv_scale), kv_norm_weight)
    q = golden_rope_tail(q, cos, sin)
    kv = golden_rope_tail(kv, cos, sin)
    return q, kv, qr
