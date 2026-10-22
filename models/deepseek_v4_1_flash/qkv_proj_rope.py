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
        D, Q_LORA, "proj_q_a", "quant_q_a", Q_A_N_TILE, Q_A_K_TILE, 1,
    )
    _q_normalize(projected, q_norm_weight, query_latent, num_tokens)
    return query_latent


@pl.jit.inline(auto_scope=False)
def _q_proj_rope_from_mx(
    payload: pl.Tensor,
    scale_backing: pl.Tensor,
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    q_b_n_tile: pl.constexpr,
):
    """Project a prequantized query latent and apply trailing RoPE."""
    activation_scale = pl.tensor.view(scale_backing, [MX_M_TILE, Q_LORA // 32], layout=pl.MX_A_ZZ)
    weight_scale_bytes = pl.tensor.view(wq_b_scale, [1, Q_LORA // 32 * LOCAL_H * HEAD_DIM], layout=pl.ND)
    weight_scale = pl.tensor.view(weight_scale_bytes, [Q_LORA // 32, LOCAL_H * HEAD_DIM], layout=pl.MX_B_NN)
    workers = pl.min(LOCAL_H * HEAD_DIM // q_b_n_tile, pl.system.available_cluster_count())
    for worker in pl.spmd(workers, name_hint='attention_q_b_rope', sync_start=True):
        block_num = pl.tile.get_block_num()
        for wave in pl.range((LOCAL_H * HEAD_DIM // q_b_n_tile + block_num - 1) // block_num):
            block = wave * block_num + worker
            if block < LOCAL_H * HEAD_DIM // q_b_n_tile:
                n0 = block * q_b_n_tile
                rows = pl.min(MX_M_TILE, num_tokens)
                a0 = pl.load(payload, [0, 0], [MX_M_TILE, K_TILE])
                sa0 = pl.load(activation_scale, [0, 0], [MX_M_TILE, K_TILE // 32])
                b0 = pl.load(wq_b, [0, n0], [K_TILE, q_b_n_tile])
                sb0 = pl.load(weight_scale, [0, n0], [K_TILE // 32, q_b_n_tile])
                acc = pl.matmul_mx(a0, sa0, b0, sb0)
                for k0 in pl.pipeline(K_TILE, Q_LORA, K_TILE, stage=2):
                    a = pl.load(payload, [0, k0], [MX_M_TILE, K_TILE])
                    sa = pl.load(activation_scale, [0, k0 // 32], [MX_M_TILE, K_TILE // 32])
                    b = pl.load(wq_b, [k0, n0], [K_TILE, q_b_n_tile])
                    sb = pl.load(weight_scale, [k0 // 32, n0], [K_TILE // 32, q_b_n_tile])
                    acc = pl.matmul_mx_acc(acc, a, sa, b, sb)
                value = pl.cast(acc, pl.BF16, mode='rint')
                if block % (HEAD_DIM // q_b_n_tile) == HEAD_DIM // q_b_n_tile - 1:
                    valid_prefix = pl.tile.slice(value, [MX_M_TILE, q_b_n_tile - ROPE_DIM], [0, 0], valid_shape=[rows, q_b_n_tile - ROPE_DIM])
                    query = pl.store(valid_prefix, [0, n0], query)
                    tail = pl.cast(value[:, q_b_n_tile - ROPE_DIM:q_b_n_tile], pl.FP32)
                    even = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P0101)
                    odd = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P1010)
                    cosine = pl.load(rope_cos, [0, 0], [MX_M_TILE, ROPE_DIM // 2], valid_shape=[rows, ROPE_DIM // 2])
                    sine = pl.load(rope_sin, [0, 0], [MX_M_TILE, ROPE_DIM // 2], valid_shape=[rows, ROPE_DIM // 2])
                    cosine = pl.fillpad(cosine, pad_value=pl.PadValue.zero)
                    cosine = pl.set_validshape(cosine, MX_M_TILE, ROPE_DIM // 2)
                    sine = pl.fillpad(sine, pad_value=pl.PadValue.zero)
                    sine = pl.set_validshape(sine, MX_M_TILE, ROPE_DIM // 2)
                    sine = pl.mul(sine, 1.0)
                    re = pl.sub(pl.mul(even, cosine), pl.mul(odd, sine))
                    im = pl.add(pl.mul(even, sine), pl.mul(odd, cosine))
                    real_lanes = pl.tile.full([MX_M_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
                    real_lanes = pl.tile.scatter_mask(real_lanes, re, mask_pattern=pl.tile.MaskPattern.P0101)
                    imag_lanes = pl.tile.full([MX_M_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
                    imag_lanes = pl.tile.scatter_mask(imag_lanes, im, mask_pattern=pl.tile.MaskPattern.P1010)
                    real_bits = pl.reinterpret_view(real_lanes, pl.UINT32)
                    imag_bits = pl.reinterpret_view(imag_lanes, pl.UINT32)
                    rotated = pl.reinterpret_view(pl.or_(real_bits, imag_bits), pl.FP32)
                    rounded = pl.cast(rotated, pl.BF16, mode='rint')
                    valid_rotated = pl.set_validshape(rounded, rows, ROPE_DIM)
                    query = pl.store(valid_rotated, [0, n0 + q_b_n_tile - ROPE_DIM], query)
                else:
                    query = pl.store(pl.set_validshape(value, rows, q_b_n_tile), [0, n0], query)
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
    if TP_SIZE == 1 and num_tokens > 0 and num_tokens <= MX_M_TILE:
        payload = pl.create_tensor([MX_M_TILE, Q_LORA], dtype=pl.FP8E4M3FN)
        scale_backing = pl.create_tensor([1, MX_M_TILE * (Q_LORA // 32)], dtype=pl.FP8E8M0)
        payload, scale_backing = _quantize_projection_input(
            query_latent, payload, scale_backing, num_tokens, Q_LORA, "quant_q_b",
        )
        _q_proj_rope_from_mx(
            payload, scale_backing, wq_b, wq_b_scale, rope_cos, rope_sin,
            query, num_tokens, N_TILE,
        )
    else:
        tokens = pl.tensor.dim(query_latent, 0)
        expanded = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
        gemm_bf16_via_mxfp8_to_bf16_staged(
            query_latent, wq_b, wq_b_scale, expanded, num_tokens,
            Q_LORA, LOCAL_H * HEAD_DIM, "proj_q_b", "quant_q_b", N_TILE, K_TILE, 1,
        )
        rope(expanded, rope_cos, rope_sin, query, num_tokens, LOCAL_H, "rope_q", max_workers)

    return query


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
        D, HEAD_DIM, "proj_kv", "quant_kv", N_TILE, K_TILE, 1,
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
):
    """Project Q and KV with one shared small-row input quantization."""
    if TP_SIZE == 1 and num_tokens > 0 and num_tokens <= MX_M_TILE:
        payload = pl.create_tensor([MX_M_TILE, D], dtype=pl.FP8E4M3FN)
        scale_backing = pl.create_tensor([1, MX_M_TILE * (D // 32)], dtype=pl.FP8E8M0)
        payload, scale_backing = _quantize_projection_input(
            x, payload, scale_backing, num_tokens, D, "attention_qkv_input_quant",
        )
        tokens = pl.tensor.dim(x, 0)
        projected = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
        gemm_mxfp8_to_bf16(
            payload, scale_backing, wq_a, wq_a_scale, projected,
            num_tokens, 0, 0, 0,
            D, Q_LORA, "proj_q_a", Q_A_N_TILE, Q_A_K_TILE, 1, 0, 0, MX_M_TILE,
        )
        if fuse_qnorm:
            qb_payload = pl.create_tensor([MX_M_TILE, Q_LORA], dtype=pl.FP8E4M3FN)
            qb_scales = pl.create_tensor([1, MX_M_TILE * (Q_LORA // 32)], dtype=pl.FP8E8M0)
            _q_normalize_to_mx(projected, q_norm_weight, query_latent, qb_payload, qb_scales, num_tokens, 5)
            _q_proj_rope_from_mx(
                qb_payload, qb_scales, wq_b, wq_b_scale, rope_cos, rope_sin,
                query, num_tokens, q_b_n_tile,
            )
        else:
            _q_normalize(projected, q_norm_weight, query_latent, num_tokens)
            qb_payload = pl.create_tensor([MX_M_TILE, Q_LORA], dtype=pl.FP8E4M3FN)
            qb_scales = pl.create_tensor([1, MX_M_TILE * (Q_LORA // 32)], dtype=pl.FP8E8M0)
            qb_payload, qb_scales = _quantize_projection_input(
                query_latent, qb_payload, qb_scales, num_tokens, Q_LORA, "quant_q_b",
            )
            _q_proj_rope_from_mx(
                qb_payload, qb_scales, wq_b, wq_b_scale, rope_cos, rope_sin,
                query, num_tokens, q_b_n_tile,
            )
        gemm_mxfp8_to_bf16(
            payload, scale_backing, wkv, wkv_scale, kv_projection,
            num_tokens, 0, 0, 0,
            D, HEAD_DIM, "proj_kv", N_TILE, K_TILE, 1, 0, 0, MX_M_TILE,
        )
        _kv_normalize_rope(kv_projection, kv_norm_weight, rope_cos, rope_sin, window_kv, num_tokens)
    else:
        q_proj_qr(x, wq_a, wq_a_scale, q_norm_weight, query_latent, num_tokens)
        q_proj_rope(query_latent, wq_b, wq_b_scale, rope_cos, rope_sin, query, num_tokens, max_workers)
        kv_proj_rope(x, wkv, wkv_scale, kv_norm_weight, rope_cos, rope_sin, kv_projection, window_kv, num_tokens)
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
