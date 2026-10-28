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

from config import FLASH as M, TP_SIZE
from gemm import (
    MX_M_TILE, MX_K_TILE,
    gemm_mxfp8_to_bf16,
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
PROJ_M_TILE = 80


@pl.jit.inline(auto_scope=False)
def quantize_projection_input(
    x: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    """Quantize shared Q/KV input into M80-padded payload and M16-packed scales."""
    capacity = pl.tensor.dim(x, 0)
    padded_rows = ((capacity + PROJ_M_TILE - 1) // PROJ_M_TILE) * PROJ_M_TILE
    payload = pl.create_tensor([padded_rows, D], dtype=pl.FP8E4M3FN)
    scales = pl.create_tensor([1, padded_rows * (D // 32)], dtype=pl.FP8E8M0)
    quant_blocks = (padded_rows // MX_M_TILE) * (D // MX_K_TILE)
    workers = pl.min(pl.min(quant_blocks, 32), 2 * pl.system.available_cluster_count())
    for worker in pl.spmd(workers, name_hint="attention_qkv_input_quant_pooled"):
        for work in pl.range(worker, quant_blocks, workers):
            t0 = (work // (D // MX_K_TILE)) * MX_M_TILE
            k0 = (work % (D // MX_K_TILE)) * MX_K_TILE
            rows = pl.max(pl.min(MX_M_TILE, num_tokens - t0), 0)
            quantized_tile, packed_scale = quantize_bf16_to_mxfp8(x, t0, k0, rows, MX_M_TILE)
            payload = pl.store(quantized_tile, [t0, k0], payload)
            scale_offset = t0 * (D // 32) + k0 // MX_K_TILE * MX_M_TILE * (MX_K_TILE // 32)
            packed_flat = pl.reshape(packed_scale, [1, MX_M_TILE * (MX_K_TILE // 32)])
            scales = pl.store(packed_flat, [0, scale_offset], scales)
    return payload, scales


@pl.jit.inline(auto_scope=False)
def _rope_inplace(
    query: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    row_start: pl.Scalar[pl.INDEX],
    num_tokens: pl.Scalar[pl.INT32],
    heads: pl.constexpr,
    name_hint: pl.constexpr,
):
    """Rotate M16 tails in one M80 row group in place."""
    rope_workers = pl.min((PROJ_M_TILE // MX_M_TILE) * heads, 2 * pl.system.available_cluster_count())
    for worker in pl.spmd(rope_workers, name_hint=name_hint):
        for block in pl.range(worker, (PROJ_M_TILE // MX_M_TILE) * heads, rope_workers):
            t0 = block // heads * MX_M_TILE
            output_t0 = row_start + t0
            n0 = block % heads * HEAD_DIM
            rows = pl.max(pl.min(MX_M_TILE, num_tokens - t0), 0)
            tail_bf16 = pl.load(query, [output_t0, n0 + HEAD_DIM - ROPE_DIM], [MX_M_TILE, ROPE_DIM], valid_shape=[rows, ROPE_DIM])
            tail = pl.cast(tail_bf16, pl.FP32)
            even = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            cosine = pl.load(rope_cos, [output_t0, 0], [MX_M_TILE, ROPE_DIM // 2], valid_shape=[rows, ROPE_DIM // 2])
            sine = pl.load(rope_sin, [output_t0, 0], [MX_M_TILE, ROPE_DIM // 2], valid_shape=[rows, ROPE_DIM // 2])
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
            query = pl.store(valid, [output_t0, n0 + HEAD_DIM - ROPE_DIM], query)
    return query


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
def q_proj_rope(
    payload: pl.Tensor,
    scales: pl.Tensor,
    wq_a: pl.Tensor,
    wq_a_scale: pl.Tensor,
    q_norm_weight: pl.Tensor,
    wq_b: pl.Tensor,
    wq_b_scale: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    query_latent: pl.Tensor,
    query: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    q_b_n_tile: pl.constexpr = 256,
    q_b_max_workers: pl.constexpr = 2147483647,
):
    """Project QA, normalize and quantize the latent, then project QB and rotate Q."""
    for group_start in pl.parallel(0, num_tokens, PROJ_M_TILE):
        group_rows = pl.min(PROJ_M_TILE, num_tokens - group_start)
        group_payload = pl.tensor.slice(payload, [PROJ_M_TILE, D], [group_start, 0])
        scale_start = group_start * (D // 32)
        group_scales = pl.tensor.slice(scales, [1, PROJ_M_TILE * (D // 32)], [0, scale_start])
        projected = pl.create_tensor([PROJ_M_TILE, Q_LORA], dtype=pl.BF16)
        gemm_mxfp8_to_bf16(
            group_payload, group_scales, wq_a, wq_a_scale, projected,
            group_rows, 0, 0, 0,
            D, Q_LORA, "proj_q_a", Q_A_N_TILE, Q_A_K_TILE, 1, 0, 0, PROJ_M_TILE, 2147483647, 2,
        )
        qb_payload = pl.create_tensor([PROJ_M_TILE, Q_LORA], dtype=pl.FP8E4M3FN)
        qb_scales = pl.create_tensor([1, PROJ_M_TILE * (Q_LORA // 32)], dtype=pl.FP8E8M0)
        for q_mt in pl.parallel(PROJ_M_TILE // MX_M_TILE):
            q_local = q_mt * MX_M_TILE
            q_start = group_start + pl.min(q_local, group_rows)
            q_rows = pl.max(pl.min(MX_M_TILE, group_rows - q_local), 0)
            projected_tile = pl.slice(projected, [q_rows, Q_LORA], [q_local, 0])
            latent_tile = pl.slice(query_latent, [q_rows, Q_LORA], [q_start, 0])
            qb_payload_tile = pl.tensor.slice(qb_payload, [MX_M_TILE, Q_LORA], [q_local, 0])
            qb_scale_start = q_local * (Q_LORA // 32)
            qb_scales_tile = pl.tensor.slice(qb_scales, [1, MX_M_TILE * (Q_LORA // 32)], [0, qb_scale_start])
            _q_normalize_to_mx(
                projected_tile, q_norm_weight, latent_tile,
                qb_payload_tile, qb_scales_tile, q_rows, 5,
            )
        group_tokens = pl.cast(group_rows, pl.INT32)
        qb_payload_group = qb_payload
        qb_scales_group = qb_scales
        query_group = pl.slice(query, [group_rows, LOCAL_H * HEAD_DIM], [group_start, 0])
        cosine_group = pl.slice(rope_cos, [group_rows, ROPE_DIM // 2], [group_start, 0])
        sine_group = pl.slice(rope_sin, [group_rows, ROPE_DIM // 2], [group_start, 0])
        _q_proj_rope_from_mx(
            qb_payload_group, qb_scales_group, wq_b, wq_b_scale,
            cosine_group, sine_group, query_group, group_tokens,
            q_b_n_tile, PROJ_M_TILE, q_b_max_workers,
        )
    return query_latent, query


@pl.jit.inline(auto_scope=False)
def _q_proj_rope_from_mx(
    payload: pl.Tensor,
    scale_backing: pl.Tensor,
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    query: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    q_b_n_tile: pl.constexpr,
    m_tile: pl.constexpr,
    max_workers: pl.constexpr = 2147483647,
):
    """Fuse the Q-B MXFP8 projection and RoPE epilogue for one M80 group."""
    activation_scale = pl.tensor.view(scale_backing, [m_tile, Q_LORA // 32], layout=pl.MX_A_ZZ)
    weight_scale_bytes = pl.tensor.view(wq_b_scale, [1, Q_LORA // 32 * LOCAL_H * HEAD_DIM], layout=pl.ND)
    weight_scale = pl.tensor.view(weight_scale_bytes, [Q_LORA // 32, LOCAL_H * HEAD_DIM], layout=pl.MX_B_NN)
    output_blocks = LOCAL_H * HEAD_DIM // q_b_n_tile
    head_panels = HEAD_DIM // q_b_n_tile
    workers = pl.min(output_blocks, max_workers)
    workers = pl.min(workers, pl.system.available_cluster_count())
    for worker in pl.spmd(workers, name_hint="attention_q_b_rope", sync_start=True):
        full_a = pl.load(payload, [0, 0], [m_tile, Q_LORA], target_memory=pl.MemorySpace.Mat)
        block_num = pl.tile.get_block_num()
        for wave in pl.range((output_blocks + block_num - 1) // block_num):
            worker_block = wave * block_num + worker
            head_start = worker_block // head_panels * head_panels
            selected_panel = worker_block % head_panels
            if block_num % head_panels == 0:
                selected_panel = (worker + wave) % head_panels
            for head_panel in pl.range(head_panels):
                block = head_start + head_panel
                if head_panel == selected_panel and block < output_blocks:
                    n0 = block * q_b_n_tile
                    a0 = pl.tile.slice(full_a, [m_tile, K_TILE], [0, 0])
                    sa0 = pl.load(activation_scale, [0, 0], [m_tile, K_TILE // 32])
                    b0 = pl.load(wq_b, [0, n0], [K_TILE, q_b_n_tile])
                    sb0 = pl.load(weight_scale, [0, n0], [K_TILE // 32, q_b_n_tile])
                    acc = pl.matmul_mx(a0, sa0, b0, sb0)
                    for k0 in pl.pipeline(K_TILE, Q_LORA, K_TILE, stage=4):
                        a = pl.tile.slice(full_a, [m_tile, K_TILE], [0, k0])
                        sa = pl.load(activation_scale, [0, k0 // 32], [m_tile, K_TILE // 32])
                        b = pl.load(wq_b, [k0, n0], [K_TILE, q_b_n_tile])
                        sb = pl.load(weight_scale, [k0 // 32, n0], [K_TILE // 32, q_b_n_tile])
                        acc = pl.matmul_mx_acc(acc, a, sa, b, sb)
                    full_vector = pl.tile.move(
                        acc, target_memory=pl.MemorySpace.Vec,
                        blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box,
                    )
                    for t0 in pl.range(0, num_tokens, MX_M_TILE):
                        rows = pl.min(MX_M_TILE, num_tokens - t0)
                        acc_rows = pl.tile.slice(full_vector, [MX_M_TILE, q_b_n_tile], [t0, 0])
                        value = pl.cast(acc_rows, pl.BF16, mode="rint")
                        last_panel = block % head_panels == head_panels - 1
                        if last_panel:
                            prefix = pl.tile.slice(
                                value, [MX_M_TILE, q_b_n_tile - ROPE_DIM], [0, 0],
                                valid_shape=[rows, q_b_n_tile - ROPE_DIM],
                            )
                            query = pl.store(prefix, [t0, n0], query)
                            tail = pl.cast(value[:, q_b_n_tile - ROPE_DIM:q_b_n_tile], pl.FP32)
                            even = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P0101)
                            odd = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P1010)
                            cosine = pl.load(rope_cos, [t0, 0], [MX_M_TILE, ROPE_DIM // 2], valid_shape=[rows, ROPE_DIM // 2])
                            sine = pl.load(rope_sin, [t0, 0], [MX_M_TILE, ROPE_DIM // 2], valid_shape=[rows, ROPE_DIM // 2])
                            cosine = pl.set_validshape(pl.fillpad(cosine, pad_value=pl.PadValue.zero), MX_M_TILE, ROPE_DIM // 2)
                            sine = pl.set_validshape(pl.fillpad(sine, pad_value=pl.PadValue.zero), MX_M_TILE, ROPE_DIM // 2)
                            sine = pl.mul(sine, 1.0)
                            real = pl.sub(pl.mul(even, cosine), pl.mul(odd, sine))
                            imag = pl.add(pl.mul(even, sine), pl.mul(odd, cosine))
                            real_lanes = pl.tile.full([MX_M_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
                            real_lanes = pl.tile.scatter_mask(real_lanes, real, mask_pattern=pl.tile.MaskPattern.P0101)
                            imag_lanes = pl.tile.full([MX_M_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
                            imag_lanes = pl.tile.scatter_mask(imag_lanes, imag, mask_pattern=pl.tile.MaskPattern.P1010)
                            real_bits = pl.reinterpret_view(real_lanes, pl.UINT32)
                            imag_bits = pl.reinterpret_view(imag_lanes, pl.UINT32)
                            rotated = pl.reinterpret_view(pl.or_(real_bits, imag_bits), pl.FP32)
                            rotated = pl.cast(rotated, pl.BF16, mode="rint")
                            rotated = pl.set_validshape(rotated, rows, ROPE_DIM)
                            query = pl.store(rotated, [t0, n0 + q_b_n_tile - ROPE_DIM], query)
                        else:
                            query = pl.store(pl.set_validshape(value, rows, q_b_n_tile), [t0, n0], query)
    return query


@pl.jit.inline(auto_scope=False)
def q_proj_rope_retain(
    payload: pl.Tensor,
    scales: pl.Tensor,
    wq_a: pl.Tensor,
    wq_a_scale: pl.Tensor,
    q_norm_weight: pl.Tensor,
    wq_b: pl.Tensor,
    wq_b_scale: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    query_latent: pl.Tensor,
    query: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    q_b_n_tile: pl.constexpr = 256,
    q_b_max_workers: pl.constexpr = 2147483647,
):
    """Project Q and retain the M80 Q-B payload/scales for Index-Q reuse."""
    payload_rows = pl.tensor.dim(payload, 0)
    qb_payload = pl.create_tensor([payload_rows, Q_LORA], dtype=pl.FP8E4M3FN)
    qb_scales = pl.create_tensor([1, payload_rows * (Q_LORA // 32)], dtype=pl.FP8E8M0)
    for group_start in pl.parallel(0, num_tokens, PROJ_M_TILE):
        group_rows = pl.min(PROJ_M_TILE, num_tokens - group_start)
        group_payload = pl.tensor.slice(payload, [PROJ_M_TILE, D], [group_start, 0])
        scale_start = group_start * (D // 32)
        group_scales = pl.tensor.slice(scales, [1, PROJ_M_TILE * (D // 32)], [0, scale_start])
        projected = pl.create_tensor([PROJ_M_TILE, Q_LORA], dtype=pl.BF16)
        gemm_mxfp8_to_bf16(
            group_payload, group_scales, wq_a, wq_a_scale, projected,
            group_rows, 0, 0, 0,
            D, Q_LORA, "proj_q_a", Q_A_N_TILE, Q_A_K_TILE, 1, 0, 0, PROJ_M_TILE, 2147483647, 2,
        )
        qb_payload_group = pl.tensor.slice(qb_payload, [PROJ_M_TILE, Q_LORA], [group_start, 0])
        qb_scale_start = group_start * (Q_LORA // 32)
        qb_scales_group = pl.tensor.slice(
            qb_scales, [1, PROJ_M_TILE * (Q_LORA // 32)], [0, qb_scale_start],
        )
        for q_mt in pl.parallel(PROJ_M_TILE // MX_M_TILE):
            q_local = q_mt * MX_M_TILE
            q_start = group_start + pl.min(q_local, group_rows)
            q_rows = pl.max(pl.min(MX_M_TILE, group_rows - q_local), 0)
            projected_tile = pl.slice(projected, [q_rows, Q_LORA], [q_local, 0])
            latent_tile = pl.slice(query_latent, [q_rows, Q_LORA], [q_start, 0])
            qb_payload_tile = pl.tensor.slice(qb_payload_group, [MX_M_TILE, Q_LORA], [q_local, 0])
            qb_scale_offset = q_local * (Q_LORA // 32)
            qb_scales_tile = pl.tensor.slice(
                qb_scales_group, [1, MX_M_TILE * (Q_LORA // 32)], [0, qb_scale_offset],
            )
            _q_normalize_to_mx(
                projected_tile, q_norm_weight, latent_tile,
                qb_payload_tile, qb_scales_tile, q_rows, 5,
            )
        query_group = pl.slice(query, [group_rows, LOCAL_H * HEAD_DIM], [group_start, 0])
        group_tokens = pl.cast(group_rows, pl.INT32)
        cosine_group = pl.slice(rope_cos, [group_rows, ROPE_DIM // 2], [group_start, 0])
        sine_group = pl.slice(rope_sin, [group_rows, ROPE_DIM // 2], [group_start, 0])
        _q_proj_rope_from_mx(
            qb_payload_group, qb_scales_group, wq_b, wq_b_scale,
            cosine_group, sine_group, query_group, group_tokens,
            q_b_n_tile, PROJ_M_TILE, q_b_max_workers,
        )
    return query_latent, query, qb_payload, qb_scales


@pl.jit.inline(auto_scope=False)
def kv_proj_rope(
    payload: pl.Tensor,
    scales: pl.Tensor,
    wkv: pl.Tensor,
    wkv_scale: pl.Tensor,
    kv_norm_weight: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    kv_projection: pl.Tensor,
    window_kv: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    """Project KV, normalize to BF16, and rotate its RoPE tail in M80 groups."""
    for group_start in pl.parallel(0, num_tokens, PROJ_M_TILE):
        group_rows = pl.min(PROJ_M_TILE, num_tokens - group_start)
        group_payload = pl.tensor.slice(payload, [PROJ_M_TILE, D], [group_start, 0])
        scale_start = group_start * (D // 32)
        group_scales = pl.tensor.slice(scales, [1, PROJ_M_TILE * (D // 32)], [0, scale_start])
        projection_group = pl.slice(kv_projection, [group_rows, HEAD_DIM], [group_start, 0])
        gemm_mxfp8_to_bf16(
            group_payload, group_scales, wkv, wkv_scale, projection_group,
            group_rows, 0, 0, 0,
            D, HEAD_DIM, "proj_kv", N_TILE, K_TILE, 1, 0, 0, PROJ_M_TILE, 2147483647, 2,
        )
        normalized_group = pl.slice(window_kv, [group_rows, HEAD_DIM], [group_start, 0])
        group_tokens = pl.cast(group_rows, pl.INT32)
        for norm_worker in pl.spmd((group_tokens + 7) // 8, name_hint="norm_kv"):
            rms_norm(projection_group, kv_norm_weight, normalized_group, norm_worker, group_tokens, width=HEAD_DIM)
        _rope_inplace(window_kv, rope_cos, rope_sin, group_start, group_tokens, 1, "rope_kv")
    return kv_projection, window_kv


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
