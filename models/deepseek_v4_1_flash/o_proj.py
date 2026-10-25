# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Re-enable CPU validation when flat MX scale packing is corrected:
# https://github.com/hw-native-sys/pto-isa/issues/357
"""V4.1 inverse RoPE, grouped W8A8 MXFP8 output projection, and TP reductions."""

from gemm import (
    MX_M_TILE, gemm_mxfp8_to_bf16, gemm_mxfp8_to_mxfp8, gemm_mxfp8_to_mxfp8_m80_dual,
    gemm_bf16_via_mxfp8_to_bf16_staged, gemm_bf16_via_mxfp8_to_bf16_fused,
)

import torch

from quant import MX_GROUP, _quantize_mxfp8_activation, decode_e8m0, unpack_mx_b_scale

from qkv_proj_rope import golden_rope_interleave as rope_interleave
from gemm import golden_gemm_bf16_via_mxfp8_to_bf16_sequential, golden_mxfp8_linear_grouped_sum


import pypto.language as pl
import pypto.language.distributed as pld

from config import (
    PREFILL_MAX_TOKENS,
    FLASH as M, SP_T_DYN, T_DYN, TP_SIZE,
)


# model config
LOCAL_H = M.num_attention_heads // TP_SIZE
LOCAL_O_GROUPS = M.o_groups // TP_SIZE
LOCAL_O_WIDTH = M.o_groups // TP_SIZE * M.o_lora_rank
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
O_LORA = M.o_lora_rank
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim


# tiling
M_TILE = 32
N_TILE = 128
O_A_N_TILE = 256
K_TILE = 256
D_TILE = 512
GROUPED_WORKER_TILE = 4096


@pl.jit.inline(auto_scope=False)
def _rotate_output(
    x: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    max_workers: pl.constexpr,
    name_hint: pl.constexpr,
):
    blocks = num_tokens * LOCAL_H
    workers = pl.max(pl.min(blocks, max_workers), 1)
    with pl.spmd(workers, name_hint=name_hint):
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, blocks, workers):
            t = block // LOCAL_H
            h = block % LOCAL_H
            base = h * HEAD_DIM
            output[t : t + 1, base : base + (HEAD_DIM - ROPE_DIM)] = x[t : t + 1, base : base + (HEAD_DIM - ROPE_DIM)]
            tail = pl.cast(x[t : t + 1, base + (HEAD_DIM - ROPE_DIM) : base + HEAD_DIM], pl.FP32)
            even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            c = cos[t : t + 1, :]
            s = pl.mul(sin[t : t + 1, :], -1.0)
            re = pl.sub(pl.mul(even, c), pl.mul(odd, s))
            im = pl.add(pl.mul(even, s), pl.mul(odd, c))
            rotated = pl.full([1, ROPE_DIM], dtype=pl.FP32, value=0.0)
            rotated = pl.tensor.scatter(re, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
            rotated = pl.tensor.scatter(im, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
            rotated = pl.cast(rotated, pl.BF16, mode="rint")
            output[t : t + 1, base + (HEAD_DIM - ROPE_DIM) : base + HEAD_DIM] = rotated
    return output


@pl.jit.inline(auto_scope=False)
def _grouped_output_inline_quant(
    x: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    weight: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    output: pl.Tensor[[T_DYN, LOCAL_O_WIDTH], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    max_workers: pl.constexpr = GROUPED_WORKER_TILE,
):
    tokens = pl.tensor.dim(x, 0)
    accumulated = pl.create_tensor([tokens, LOCAL_O_WIDTH], dtype=pl.FP32)
    blocks = (num_tokens + M_TILE - 1) // M_TILE * (LOCAL_O_WIDTH // N_TILE)
    workers = pl.max(pl.min(blocks, max_workers), 1)
    cube_tids = pl.array.create(LOCAL_O_GROUPS * (O_LORA // N_TILE), pl.TASK_ID)
    group_blocks = (num_tokens + M_TILE - 1) // M_TILE
    group_workers = pl.max(pl.min(group_blocks, max_workers), 1)
    for group in pl.parallel(LOCAL_O_GROUPS):
        weight_group = pl.reshape(weight[group:group + 1, :, :], [O_GROUP_IN, O_LORA])
        scale_mx: pl.Tensor[[O_GROUP_IN // 32, O_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.reshape(
            scale[group:group + 1, :, :], [O_GROUP_IN // 32, O_LORA]
        )
        for nb in pl.parallel(O_LORA // N_TILE):
            local_n = nb * N_TILE
            with pl.spmd(group_workers, name_hint="proj_o_a_cube") as cube_tid:
                worker = pl.tile.get_block_idx()
                for block in pl.range(worker, group_blocks, max_workers):
                    t0 = block * M_TILE
                    output_n = group * O_LORA + local_n
                    rows = pl.min(M_TILE, num_tokens - t0)
                    first = pl.load(x, [t0, group * O_GROUP_IN], [M_TILE, K_TILE], valid_shape=[rows, K_TILE])
                    first = pl.set_validshape(pl.fillpad(first, pad_value=pl.PadValue.zero), M_TILE, K_TILE)
                    fq = pl.reshape(pl.cast(first, pl.FP32), [M_TILE * (K_TILE // 32), 32])
                    fq_tmp = pl.create_tile([M_TILE * (K_TILE // 32), 32], dtype=pl.FP32)
                    fq_max = pl.row_max(pl.abs(fq), tmp_tile=fq_tmp)
                    fq_bits = pl.reinterpret_view(fq_max, pl.INT32)
                    fq_exp = pl.maximum(pl.sub(pl.shrs(fq_bits, 23), 8), 0)
                    # Use a normal divisor for scale code zero, whose payload is zero.
                    fq_scale = pl.reinterpret_view(pl.shls(pl.maximum(fq_exp, 1), 23), pl.FP32)
                    fq_normalized = pl.row_expand_div(fq, fq_scale)
                    fq_normalized = pl.row_expand_mul(fq_normalized, pl.cast(pl.minimum(fq_exp, 1), pl.FP32))
                    fq_normalized = pl.minimum(pl.maximum(fq_normalized, -448.0), 448.0)
                    a = pl.reshape(pl.cast(fq_normalized, pl.FP8E4M3FN, mode="rint"), [M_TILE, K_TILE])
                    fq_se = pl.sub(fq_exp, pl.mul(pl.shrs(fq_exp, 7), 256))
                    fq_codes = pl.reinterpret_view(pl.cast(fq_se, pl.INT8), pl.UINT8)
                    fq_flat = pl.reshape(fq_codes, [1, M_TILE * (K_TILE // 32)])
                    fq_packed = pl.tmov_x2zz(
                        fq_flat, pl.create_tile([1, 96], dtype=pl.UINT8), group_axis=1, dst_rows=M_TILE, dst_cols=8
                    )
                    sa = pl.reinterpret_view(fq_packed, pl.FP8E8M0)
                    b = pl.load(weight_group, [0, local_n], [K_TILE, N_TILE])
                    sb = pl.load(scale_mx, [0, local_n], [K_TILE // 32, N_TILE])
                    acc = pl.matmul_mx(a, sa, b, sb)

                    for kb in pl.range(1, O_GROUP_IN // K_TILE):
                        k0 = kb * K_TILE
                        values = pl.load(x, [t0, group * O_GROUP_IN + k0], [M_TILE, K_TILE], valid_shape=[rows, K_TILE])
                        values = pl.set_validshape(pl.fillpad(values, pad_value=pl.PadValue.zero), M_TILE, K_TILE)
                        q = pl.reshape(pl.cast(values, pl.FP32), [M_TILE * (K_TILE // 32), 32])
                        q_tmp = pl.create_tile([M_TILE * (K_TILE // 32), 32], dtype=pl.FP32)
                        q_max = pl.row_max(pl.abs(q), tmp_tile=q_tmp)
                        q_bits = pl.reinterpret_view(q_max, pl.INT32)
                        q_exp = pl.maximum(pl.sub(pl.shrs(q_bits, 23), 8), 0)
                        q_scale = pl.reinterpret_view(pl.shls(pl.maximum(q_exp, 1), 23), pl.FP32)
                        q_normalized = pl.row_expand_div(q, q_scale)
                        q_normalized = pl.row_expand_mul(q_normalized, pl.cast(pl.minimum(q_exp, 1), pl.FP32))
                        q_normalized = pl.minimum(pl.maximum(q_normalized, -448.0), 448.0)
                        a = pl.reshape(pl.cast(q_normalized, pl.FP8E4M3FN, mode="rint"), [M_TILE, K_TILE])
                        q_se = pl.sub(q_exp, pl.mul(pl.shrs(q_exp, 7), 256))
                        q_codes = pl.reinterpret_view(pl.cast(q_se, pl.INT8), pl.UINT8)
                        q_flat = pl.reshape(q_codes, [1, M_TILE * (K_TILE // 32)])
                        q_packed = pl.tmov_x2zz(
                            q_flat, pl.create_tile([1, 96], dtype=pl.UINT8), group_axis=1, dst_rows=M_TILE, dst_cols=8
                        )
                        sa = pl.reinterpret_view(q_packed, pl.FP8E8M0)
                        b = pl.load(weight_group, [k0, local_n], [K_TILE, N_TILE])
                        sb = pl.load(scale_mx, [k0 // 32, local_n], [K_TILE // 32, N_TILE])
                        acc = pl.matmul_mx_acc(acc, a, sa, b, sb)

                    accumulated = pl.store(pl.set_validshape(pl.mul(acc, 1.0), rows, N_TILE), [t0, output_n], accumulated)
            cube_tids[group * (O_LORA // N_TILE) + nb] = cube_tid
    with pl.spmd(workers, name_hint="cast_o_a", deps=[cube_tids]):
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, blocks, max_workers):
            t0 = block // (LOCAL_O_WIDTH // N_TILE) * M_TILE
            n0 = block % (LOCAL_O_WIDTH // N_TILE) * N_TILE
            rows = pl.min(M_TILE, num_tokens - t0)
            value = pl.slice(accumulated, [M_TILE, N_TILE], [t0, n0], valid_shape=[rows, N_TILE])
            output[t0:t0 + M_TILE, n0:n0 + N_TILE] = pl.cast(value, pl.BF16, mode="rint")
    return output


@pl.jit.inline(auto_scope=False)
def grouped_output(
    x: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    weight: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    output: pl.Tensor[[T_DYN, LOCAL_O_WIDTH], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    max_workers: pl.constexpr = GROUPED_WORKER_TILE,
):
    if TP_SIZE == 1 and num_tokens <= 16 and max_workers >= O_GROUP_IN // K_TILE:
        gemm_bf16_via_mxfp8_to_bf16_staged(
            x, weight, scale, output, num_tokens,
            O_GROUP_IN, O_LORA, "proj_o_a", "quant_o_a",
            O_A_N_TILE, K_TILE,  LOCAL_O_GROUPS, 2147483647,
        )
    else:
        _grouped_output_inline_quant(x, weight, scale, output, num_tokens, max_workers)
    return output


@pl.jit.inline(auto_scope=False)
def _grouped_output_rope_to_mx(
    attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    weight: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    output: pl.Tensor,
    output_scale: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    """Apply inverse RoPE and grouped O-A with OCP input and output MX quantization."""
    weight_flat = pl.reshape(weight, [LOCAL_O_GROUPS * O_GROUP_IN, O_LORA])
    for group in pl.parallel(LOCAL_O_GROUPS):
        payload = pl.create_tensor([MX_M_TILE, O_GROUP_IN], dtype=pl.FP8E4M3FN)
        scale_backing = pl.create_tensor([1, MX_M_TILE * (O_GROUP_IN // 32)], dtype=pl.FP8E8M0)
        for block in pl.spmd(O_GROUP_IN // (2 * K_TILE), name_hint="rope_quant_o"):
            rows = pl.min(MX_M_TILE, num_tokens)
            cosine = pl.load(cos, [0, 0], [MX_M_TILE, ROPE_DIM // 2], valid_shape=[rows, ROPE_DIM // 2])
            cosine = pl.fillpad(cosine, pad_value=pl.PadValue.zero)
            cosine = pl.set_validshape(cosine, MX_M_TILE, ROPE_DIM // 2)
            sine = pl.load(sin, [0, 0], [MX_M_TILE, ROPE_DIM // 2], valid_shape=[rows, ROPE_DIM // 2])
            sine = pl.fillpad(sine, pad_value=pl.PadValue.zero)
            sine = pl.mul(pl.set_validshape(sine, MX_M_TILE, ROPE_DIM // 2), -1.0)
            for chunk in pl.unroll(2):
                k0 = (chunk * (O_GROUP_IN // (2 * K_TILE)) + block) * K_TILE
                input_n = group * O_GROUP_IN + k0
                raw_source = pl.load(attended, [0, input_n], [MX_M_TILE, K_TILE], valid_shape=[rows, K_TILE])
                if block % (HEAD_DIM // K_TILE) == HEAD_DIM // K_TILE - 1:
                    tail = pl.cast(raw_source[:, K_TILE - ROPE_DIM:K_TILE], pl.FP32)
                    even = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P0101)
                    odd = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P1010)
                    re = pl.sub(pl.mul(even, cosine), pl.mul(odd, sine))
                    im = pl.add(pl.mul(even, sine), pl.mul(odd, cosine))
                    real_lanes = pl.tile.full([MX_M_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
                    real_lanes = pl.tile.scatter_mask(real_lanes, re, mask_pattern=pl.tile.MaskPattern.P0101)
                    imag_lanes = pl.tile.full([MX_M_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
                    imag_lanes = pl.tile.scatter_mask(imag_lanes, im, mask_pattern=pl.tile.MaskPattern.P1010)
                    real_bits = pl.reinterpret_view(real_lanes, pl.UINT32)
                    imag_bits = pl.reinterpret_view(imag_lanes, pl.UINT32)
                    rotated = pl.reinterpret_view(pl.or_(real_bits, imag_bits), pl.FP32)
                    rotated = pl.cast(rotated, pl.BF16, mode="rint")
                    valid_rotated = pl.set_validshape(rotated, rows, ROPE_DIM)
                    assembled_source = pl.tile.assemble(raw_source, valid_rotated, [0, K_TILE - ROPE_DIM])
                    valid_source = pl.set_validshape(assembled_source, rows, K_TILE)
                else:
                    valid_source = raw_source
                padded_source = pl.fillpad(valid_source, pad_value=pl.PadValue.zero)
                source = pl.set_validshape(padded_source, MX_M_TILE, K_TILE)
                groups = pl.reshape(pl.cast(source, pl.FP32), [MX_M_TILE * (K_TILE // 32), 32])
                reduce_tmp = pl.create_tile([MX_M_TILE * (K_TILE // 32), 32], dtype=pl.FP32)
                maximum = pl.row_max(pl.abs(groups), tmp_tile=reduce_tmp)
                bits = pl.reinterpret_view(maximum, pl.INT32)
                exponent = pl.maximum(pl.sub(pl.shrs(bits, 23), 8), 0)
                divisor = pl.reinterpret_view(pl.shls(pl.maximum(exponent, 1), 23), pl.FP32)
                normalized = pl.row_expand_div(groups, divisor)
                floor_mask = pl.cast(pl.minimum(exponent, 1), pl.FP32)
                normalized = pl.row_expand_mul(normalized, floor_mask)
                normalized = pl.minimum(pl.maximum(normalized, -448.0), 448.0)
                quantized = pl.cast(normalized, pl.FP8E4M3FN, mode="rint")
                quantized_tile = pl.reshape(quantized, [MX_M_TILE, K_TILE])
                payload = pl.store(quantized_tile, [0, k0], payload)
                signed = pl.sub(exponent, pl.mul(pl.shrs(exponent, 7), 256))
                codes = pl.reinterpret_view(pl.cast(signed, pl.INT8), pl.UINT8)
                flat = pl.reshape(codes, [1, MX_M_TILE * (K_TILE // 32)])
                pack_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                packed = pl.tmov_x2zz(flat, pack_tmp, group_axis=1, dst_rows=MX_M_TILE, dst_cols=8)
                packed_scale = pl.reinterpret_view(packed, pl.FP8E8M0)
                scale_offset = k0 // K_TILE * MX_M_TILE * (K_TILE // 32)
                packed_flat = pl.reshape(packed_scale, [1, MX_M_TILE * (K_TILE // 32)])
                scale_backing = pl.store(packed_flat, [0, scale_offset], scale_backing)
        output, output_scale = gemm_mxfp8_to_mxfp8(
            payload, scale_backing, weight_flat, scale, output, output_scale,
            num_tokens, group, O_GROUP_IN, O_LORA, "proj_o_a",
            O_A_N_TILE, K_TILE, LOCAL_O_GROUPS, 8,
        )
    return output, output_scale


@pl.jit.inline(auto_scope=False)
def o_proj(
    attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    output: pl.Tensor[[T_DYN, D], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    grouped_max_workers: pl.constexpr = GROUPED_WORKER_TILE,
):
    """Project attention output with native MXFP8 GEMM."""
    tokens = pl.tensor.dim(attended, 0)
    projected = pl.create_tensor([tokens, D], dtype=pl.BF16)
    if TP_SIZE == 1 and num_tokens > 0 and num_tokens <= 16 and grouped_max_workers >= O_GROUP_IN // K_TILE:
        latent_payload = pl.create_tensor([16, LOCAL_O_WIDTH], dtype=pl.FP8E4M3FN)
        latent_scale_backing = pl.create_tensor([1, 16 * (LOCAL_O_WIDTH // 32)], dtype=pl.FP8E8M0)
        latent_payload, latent_scale_backing = _grouped_output_rope_to_mx(
            attended, rope_cos, rope_sin, wo_a, wo_a_scale,
            latent_payload, latent_scale_backing, num_tokens,
        )
        gemm_mxfp8_to_bf16(
            latent_payload, latent_scale_backing, wo_b, wo_b_scale, projected, num_tokens, 0, 0, 0,
            LOCAL_O_WIDTH, D, "proj_o_b", 256, K_TILE,  1, 0, 0, MX_M_TILE, 2147483647,
        )
    else:
        unrotated = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
        _rotate_output(attended, rope_cos, rope_sin, unrotated, num_tokens, 64, "rope_o")
        latent = pl.create_tensor([tokens, LOCAL_O_WIDTH], dtype=pl.BF16)
        grouped_output(
            unrotated, wo_a, wo_a_scale, latent, num_tokens, max_workers=grouped_max_workers,
        )
        gemm_bf16_via_mxfp8_to_bf16_fused(
            latent, wo_b, wo_b_scale, projected, num_tokens,
            width=LOCAL_O_WIDTH, output_width=D,
            n_tile=128, name_hint="proj_o_b",
        )
    export_blocks = (num_tokens + MX_M_TILE - 1) // MX_M_TILE
    export_workers = pl.max(pl.min(export_blocks, 64), 1)
    for worker in pl.spmd(export_workers, name_hint="export_o_b"):
        for block in pl.range(worker, export_blocks, export_workers):
            t0 = block * MX_M_TILE
            rows = pl.min(MX_M_TILE, num_tokens - t0)
            for d0 in pl.range(0, D, D_TILE):
                value = pl.slice(projected, [MX_M_TILE, D_TILE], [t0, d0], valid_shape=[rows, D_TILE])
                output[t0:t0 + MX_M_TILE, d0:d0 + D_TILE] = pl.cast(value, pl.FP32)
    return output


@pl.jit.inline(auto_scope=False)
def _ob_hc_post_from_mx(
    payload: pl.Tensor,
    activation_scale_backing: pl.Tensor,
    weight: pl.Tensor,
    weight_scale: pl.Tensor,
    sublayer: pl.Tensor,
    residual: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    """Finish each TP1 O-B panel with the BF16 output and HC residual expansion."""
    capacity = pl.tensor.dim(sublayer, 0)
    residual_flat = pl.reshape(residual, [capacity, HC_DIM])
    residual_mix_flat = pl.reshape(residual_mix, [capacity, HC_MULT * HC_MULT])
    output_flat = pl.reshape(output, [capacity, HC_DIM])
    activation_scale = pl.tensor.view(activation_scale_backing, [MX_M_TILE, LOCAL_O_WIDTH // 32], layout=pl.MX_A_ZZ)
    scale_group_size = (LOCAL_O_WIDTH // 32) * D
    scale_backing_flat = pl.tensor.view(weight_scale, [1, scale_group_size], layout=pl.ND)
    scale_group_backing = pl.tensor.slice(scale_backing_flat, [1, scale_group_size], [0, 0])
    scale_group = pl.tensor.view(scale_group_backing, [LOCAL_O_WIDTH // 32, D], layout=pl.MX_B_NN)
    workers = pl.min(D // N_TILE, pl.system.available_cluster_count())
    for worker in pl.spmd(workers, name_hint="attention_o_b_hc_post", sync_start=True):
        post_rows = pl.load(post_mix, [0, 0], [MX_M_TILE, 8], valid_shape=[capacity, HC_MULT])
        post_rows = pl.fillpad(post_rows, pad_value=pl.PadValue.zero)
        post_rows = pl.set_validshape(post_rows, MX_M_TILE, 8)
        post_bits = pl.reinterpret_view(post_rows, pl.INT32)
        post_zero = pl.tile.full([MX_M_TILE, 8], dtype=pl.INT32, value=0)
        post_dense_bits = pl.add(post_zero, post_bits)
        post_dense = pl.reinterpret_view(post_dense_bits, pl.FP32)
        post_transposed = pl.tile.transpose(post_dense, 0, 1)
        mix_rows = pl.load(residual_mix_flat, [0, 0], [MX_M_TILE, HC_MULT * HC_MULT], valid_shape=[capacity, HC_MULT * HC_MULT])
        mix_rows = pl.fillpad(mix_rows, pad_value=pl.PadValue.zero)
        mix_rows = pl.set_validshape(mix_rows, MX_M_TILE, HC_MULT * HC_MULT)
        mix_bits = pl.reinterpret_view(mix_rows, pl.INT32)
        mix_zero = pl.tile.full([MX_M_TILE, HC_MULT * HC_MULT], dtype=pl.INT32, value=0)
        mix_dense_bits = pl.add(mix_zero, mix_bits)
        mix_dense = pl.reinterpret_view(mix_dense_bits, pl.FP32)
        mix_transposed = pl.tile.transpose(mix_dense, 0, 1)
        block_num = pl.tile.get_block_num()
        for wave in pl.range((D // N_TILE + block_num - 1) // block_num):
            block = wave * block_num + worker
            if block < D // N_TILE:
                n0 = block * N_TILE
                rows = pl.min(MX_M_TILE, num_tokens)
                a0 = pl.load(payload, [0, 0], [MX_M_TILE, K_TILE])
                sa0 = pl.load(activation_scale, [0, 0], [MX_M_TILE, K_TILE // 32])
                b0 = pl.load(weight, [0, n0], [K_TILE, N_TILE])
                sb0 = pl.load(scale_group, [0, n0], [K_TILE // 32, N_TILE])
                acc = pl.matmul_mx(a0, sa0, b0, sb0)
                for k0 in pl.pipeline(K_TILE, LOCAL_O_WIDTH, K_TILE, stage=2):
                    a = pl.load(payload, [0, k0], [MX_M_TILE, K_TILE])
                    sa = pl.load(activation_scale, [0, k0 // 32], [MX_M_TILE, K_TILE // 32])
                    b = pl.load(weight, [k0, n0], [K_TILE, N_TILE])
                    sb = pl.load(scale_group, [k0 // 32, n0], [K_TILE // 32, N_TILE])
                    acc = pl.matmul_mx_acc(acc, a, sa, b, sb)
                rounded_sublayer = pl.cast(acc, pl.BF16, mode="rint")
                active_sublayer = pl.set_validshape(rounded_sublayer, rows, N_TILE)
                padded_sublayer = pl.fillpad(active_sublayer, pad_value=pl.PadValue.zero)
                sublayer_tile = pl.set_validshape(padded_sublayer, MX_M_TILE, N_TILE)
                valid_sublayer = pl.set_validshape(sublayer_tile, capacity, N_TILE)
                pl.store(valid_sublayer, [0, n0], sublayer)
                x_tile = pl.cast(sublayer_tile, pl.FP32)
                residual0 = pl.load(residual_flat, [0, n0], [MX_M_TILE, N_TILE], valid_shape=[capacity, N_TILE])
                residual1 = pl.load(residual_flat, [0, D + n0], [MX_M_TILE, N_TILE], valid_shape=[capacity, N_TILE])
                residual2 = pl.load(residual_flat, [0, 2 * D + n0], [MX_M_TILE, N_TILE], valid_shape=[capacity, N_TILE])
                residual3 = pl.load(residual_flat, [0, 3 * D + n0], [MX_M_TILE, N_TILE], valid_shape=[capacity, N_TILE])
                residual0 = pl.set_validshape(pl.fillpad(residual0, pad_value=pl.PadValue.zero), MX_M_TILE, N_TILE)
                residual1 = pl.set_validshape(pl.fillpad(residual1, pad_value=pl.PadValue.zero), MX_M_TILE, N_TILE)
                residual2 = pl.set_validshape(pl.fillpad(residual2, pad_value=pl.PadValue.zero), MX_M_TILE, N_TILE)
                residual3 = pl.set_validshape(pl.fillpad(residual3, pad_value=pl.PadValue.zero), MX_M_TILE, N_TILE)
                for out_h in pl.unroll(HC_MULT):
                    post_row = pl.tile.slice(post_transposed, [1, MX_M_TILE], [out_h, 0])
                    post_col = pl.reshape(post_row, [MX_M_TILE, 1])
                    mix0_row = pl.tile.slice(mix_transposed, [1, MX_M_TILE], [out_h, 0])
                    mix1_row = pl.tile.slice(mix_transposed, [1, MX_M_TILE], [HC_MULT + out_h, 0])
                    mix2_row = pl.tile.slice(mix_transposed, [1, MX_M_TILE], [2 * HC_MULT + out_h, 0])
                    mix3_row = pl.tile.slice(mix_transposed, [1, MX_M_TILE], [3 * HC_MULT + out_h, 0])
                    mix0_col = pl.reshape(mix0_row, [MX_M_TILE, 1])
                    mix1_col = pl.reshape(mix1_row, [MX_M_TILE, 1])
                    mix2_col = pl.reshape(mix2_row, [MX_M_TILE, 1])
                    mix3_col = pl.reshape(mix3_row, [MX_M_TILE, 1])
                    value = pl.row_expand_mul(x_tile, post_col)
                    value = pl.add(value, pl.row_expand_mul(residual0, mix0_col))
                    value = pl.add(value, pl.row_expand_mul(residual1, mix1_col))
                    value = pl.add(value, pl.row_expand_mul(residual2, mix2_col))
                    value = pl.add(value, pl.row_expand_mul(residual3, mix3_col))
                    output_bf16 = pl.cast(value, pl.BF16, mode="rint")
                    output_tile = pl.cast(output_bf16, pl.FP32)
                    valid_output = pl.set_validshape(output_tile, capacity, N_TILE)
                    pl.store(valid_output, [0, out_h * D + n0], output_flat)
    return sublayer, output


@pl.jit.inline(auto_scope=False)
def _grouped_output_rope_to_mx_m80(
    attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    weight: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    output: pl.Tensor,
    output_scale: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    """Pool M16 RoPE/quantization work and project eight O-A groups."""
    row_tiles = (pl.tensor.dim(attended, 0) + MX_M_TILE - 1) // MX_M_TILE
    padded_rows = 80
    weight_flat = pl.reshape(weight, [LOCAL_O_GROUPS * O_GROUP_IN, O_LORA])
    payload_all = pl.create_tensor([LOCAL_O_GROUPS * padded_rows, O_GROUP_IN], dtype=pl.FP8E4M3FN)
    scales_all = pl.create_tensor([LOCAL_O_GROUPS, padded_rows * (O_GROUP_IN // 32)], dtype=pl.FP8E8M0)
    group_quant_blocks = row_tiles * (O_GROUP_IN // (2 * K_TILE))
    quant_blocks = LOCAL_O_GROUPS * group_quant_blocks
    quant_workers = pl.min(pl.min(quant_blocks, 64), 2 * pl.system.available_cluster_count())
    for worker in pl.spmd(quant_workers, name_hint="rope_quant_o_pooled"):
        for work in pl.range(worker, quant_blocks, quant_workers):
            quant_group = work // group_quant_blocks
            group_work = work % group_quant_blocks
            t0 = group_work // (O_GROUP_IN // (2 * K_TILE)) * MX_M_TILE
            block = group_work % (O_GROUP_IN // (2 * K_TILE))
            rows = pl.min(MX_M_TILE, num_tokens - t0)
            cosine = pl.load(cos, [t0, 0], [MX_M_TILE, ROPE_DIM // 2], valid_shape=[rows, ROPE_DIM // 2])
            cosine = pl.fillpad(cosine, pad_value=pl.PadValue.zero)
            cosine = pl.set_validshape(cosine, MX_M_TILE, ROPE_DIM // 2)
            sine = pl.load(sin, [t0, 0], [MX_M_TILE, ROPE_DIM // 2], valid_shape=[rows, ROPE_DIM // 2])
            sine = pl.fillpad(sine, pad_value=pl.PadValue.zero)
            valid_sine = pl.set_validshape(sine, MX_M_TILE, ROPE_DIM // 2)
            sine = pl.mul(valid_sine, -1.0)
            for chunk in pl.unroll(2):
                k0 = (chunk * (O_GROUP_IN // (2 * K_TILE)) + block) * K_TILE
                input_n = quant_group * O_GROUP_IN + k0
                raw_source = pl.load(attended, [t0, input_n], [MX_M_TILE, K_TILE], valid_shape=[rows, K_TILE])
                if block % (HEAD_DIM // K_TILE) == HEAD_DIM // K_TILE - 1:
                    tail = pl.cast(raw_source[:, K_TILE - ROPE_DIM:K_TILE], pl.FP32)
                    even = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P0101)
                    odd = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P1010)
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
                    rotated = pl.cast(rotated, pl.BF16, mode="rint")
                    valid_rotated = pl.set_validshape(rotated, rows, ROPE_DIM)
                    assembled_source = pl.tile.assemble(raw_source, valid_rotated, [0, K_TILE - ROPE_DIM])
                    valid_source = pl.set_validshape(assembled_source, rows, K_TILE)
                else:
                    valid_source = raw_source
                padded_source = pl.fillpad(valid_source, pad_value=pl.PadValue.zero)
                source = pl.set_validshape(padded_source, MX_M_TILE, K_TILE)
                source_fp32 = pl.cast(source, pl.FP32)
                groups = pl.reshape(source_fp32, [MX_M_TILE * (K_TILE // 32), 32])
                reduce_tmp = pl.create_tile([MX_M_TILE * (K_TILE // 32), 32], dtype=pl.FP32)
                magnitude = pl.abs(groups)
                maximum = pl.row_max(magnitude, tmp_tile=reduce_tmp)
                bits = pl.reinterpret_view(maximum, pl.INT32)
                biased_exponent = pl.shrs(bits, 23)
                scale_exponent = pl.sub(biased_exponent, 8)
                exponent = pl.maximum(scale_exponent, 0)
                nonzero_exponent = pl.maximum(exponent, 1)
                divisor_bits = pl.shls(nonzero_exponent, 23)
                divisor = pl.reinterpret_view(divisor_bits, pl.FP32)
                normalized = pl.row_expand_div(groups, divisor)
                floor_code = pl.minimum(exponent, 1)
                floor_mask = pl.cast(floor_code, pl.FP32)
                normalized = pl.row_expand_mul(normalized, floor_mask)
                lower_clamped = pl.maximum(normalized, -448.0)
                normalized = pl.minimum(lower_clamped, 448.0)
                quantized = pl.cast(normalized, pl.FP8E4M3FN, mode="rint")
                quantized_tile = pl.reshape(quantized, [MX_M_TILE, K_TILE])
                payload_all = pl.store(quantized_tile, [quant_group * padded_rows + t0, k0], payload_all)
                sign_bit = pl.shrs(exponent, 7)
                sign_offset = pl.mul(sign_bit, 256)
                signed = pl.sub(exponent, sign_offset)
                signed_codes = pl.cast(signed, pl.INT8)
                codes = pl.reinterpret_view(signed_codes, pl.UINT8)
                flat = pl.reshape(codes, [1, MX_M_TILE * (K_TILE // 32)])
                pack_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                packed = pl.tmov_x2zz(flat, pack_tmp, group_axis=1, dst_rows=MX_M_TILE, dst_cols=8)
                packed_scale = pl.reinterpret_view(packed, pl.FP8E8M0)
                scale_offset = t0 * (O_GROUP_IN // 32) + k0 // K_TILE * MX_M_TILE * (K_TILE // 32)
                packed_flat = pl.reshape(packed_scale, [1, MX_M_TILE * (K_TILE // 32)])
                scales_all = pl.store(packed_flat, [quant_group, scale_offset], scales_all)
    for group in pl.parallel(LOCAL_O_GROUPS):
        payload = pl.tensor.slice(payload_all, [padded_rows, O_GROUP_IN], [group * padded_rows, 0])
        scale_backing = pl.tensor.slice(scales_all, [1, padded_rows * (O_GROUP_IN // 32)], [group, 0])
        output, output_scale = gemm_mxfp8_to_mxfp8_m80_dual(
            payload, scale_backing, weight_flat, scale, output, output_scale,
            num_tokens, group, O_GROUP_IN, O_LORA, "proj_o_a",
            256, K_TILE, LOCAL_O_GROUPS, 4,
        )
    return output, output_scale


@pl.jit.inline(auto_scope=False)
def _ob_hc_post_from_mx_m80(
    payload: pl.Tensor,
    activation_scale_backing: pl.Tensor,
    weight: pl.Tensor,
    weight_scale: pl.Tensor,
    sublayer: pl.Tensor,
    residual: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    """Project 80 O-B rows and apply M16 HC epilogues."""
    capacity = pl.tensor.dim(sublayer, 0)
    residual_flat = pl.reshape(residual, [capacity, HC_DIM])
    residual_mix_flat = pl.reshape(residual_mix, [capacity, HC_MULT * HC_MULT])
    output_flat = pl.reshape(output, [capacity, HC_DIM])
    activation_scale = pl.tensor.view(activation_scale_backing, [80, LOCAL_O_WIDTH // 32], layout=pl.MX_A_ZZ)
    scale_group_size = (LOCAL_O_WIDTH // 32) * D
    scale_backing_flat = pl.tensor.view(weight_scale, [1, scale_group_size], layout=pl.ND)
    scale_group_backing = pl.tensor.slice(scale_backing_flat, [1, scale_group_size], [0, 0])
    scale_group = pl.tensor.view(scale_group_backing, [LOCAL_O_WIDTH // 32, D], layout=pl.MX_B_NN)
    workers = pl.min(D // N_TILE, pl.system.available_cluster_count())
    for worker in pl.spmd(workers, name_hint="attention_o_b_hc_post", sync_start=True):
        block_num = pl.tile.get_block_num()
        for ni in pl.range(0, D // N_TILE):
            if ni % block_num == worker:
                n0 = ni * N_TILE
                a0 = pl.load(payload, [0, 0], [80, K_TILE])
                sa0 = pl.load(activation_scale, [0, 0], [80, K_TILE // 32])
                b0 = pl.load(weight, [0, n0], [K_TILE, N_TILE])
                sb0 = pl.load(scale_group, [0, n0], [K_TILE // 32, N_TILE])
                acc = pl.matmul_mx(a0, sa0, b0, sb0)
                for k0 in pl.pipeline(K_TILE, LOCAL_O_WIDTH, K_TILE, stage=2):
                    a = pl.load(payload, [0, k0], [80, K_TILE])
                    sa = pl.load(activation_scale, [0, k0 // 32], [80, K_TILE // 32])
                    b = pl.load(weight, [k0, n0], [K_TILE, N_TILE])
                    sb = pl.load(scale_group, [k0 // 32, n0], [K_TILE // 32, N_TILE])
                    acc = pl.matmul_mx_acc(acc, a, sa, b, sb)
                full_vector = pl.tile.move(
                    acc, target_memory=pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box,
                )
                for t0 in pl.range(0, 80, MX_M_TILE):
                    capacity_rows = pl.min(MX_M_TILE, capacity - t0)
                    rows = pl.min(MX_M_TILE, num_tokens - t0)
                    post_rows = pl.load(post_mix, [t0, 0], [MX_M_TILE, 8], valid_shape=[capacity_rows, HC_MULT])
                    post_rows = pl.fillpad(post_rows, pad_value=pl.PadValue.zero)
                    post_rows = pl.set_validshape(post_rows, MX_M_TILE, 8)
                    post_bits = pl.reinterpret_view(post_rows, pl.INT32)
                    post_zero = pl.tile.full([MX_M_TILE, 8], dtype=pl.INT32, value=0)
                    post_dense_bits = pl.add(post_zero, post_bits)
                    post_dense = pl.reinterpret_view(post_dense_bits, pl.FP32)
                    post_transposed = pl.tile.transpose(post_dense, 0, 1)
                    mix_rows = pl.load(residual_mix_flat, [t0, 0], [MX_M_TILE, HC_MULT * HC_MULT], valid_shape=[capacity_rows, HC_MULT * HC_MULT])
                    mix_rows = pl.fillpad(mix_rows, pad_value=pl.PadValue.zero)
                    mix_rows = pl.set_validshape(mix_rows, MX_M_TILE, HC_MULT * HC_MULT)
                    mix_bits = pl.reinterpret_view(mix_rows, pl.INT32)
                    mix_zero = pl.tile.full([MX_M_TILE, HC_MULT * HC_MULT], dtype=pl.INT32, value=0)
                    mix_dense_bits = pl.add(mix_zero, mix_bits)
                    mix_dense = pl.reinterpret_view(mix_dense_bits, pl.FP32)
                    mix_transposed = pl.tile.transpose(mix_dense, 0, 1)
                    acc_rows = pl.tile.slice(full_vector, [MX_M_TILE, N_TILE], [t0, 0])
                    rounded_sublayer = pl.cast(acc_rows, pl.BF16, mode="rint")
                    active_sublayer = pl.set_validshape(rounded_sublayer, rows, N_TILE)
                    padded_sublayer = pl.fillpad(active_sublayer, pad_value=pl.PadValue.zero)
                    sublayer_tile = pl.set_validshape(padded_sublayer, MX_M_TILE, N_TILE)
                    valid_sublayer = pl.set_validshape(sublayer_tile, capacity_rows, N_TILE)
                    pl.store(valid_sublayer, [t0, n0], sublayer)
                    x_tile = pl.cast(sublayer_tile, pl.FP32)
                    residual0 = pl.load(residual_flat, [t0, n0], [MX_M_TILE, N_TILE], valid_shape=[capacity_rows, N_TILE])
                    residual1 = pl.load(residual_flat, [t0, D + n0], [MX_M_TILE, N_TILE], valid_shape=[capacity_rows, N_TILE])
                    residual2 = pl.load(residual_flat, [t0, 2 * D + n0], [MX_M_TILE, N_TILE], valid_shape=[capacity_rows, N_TILE])
                    residual3 = pl.load(residual_flat, [t0, 3 * D + n0], [MX_M_TILE, N_TILE], valid_shape=[capacity_rows, N_TILE])
                    padded_residual0 = pl.fillpad(residual0, pad_value=pl.PadValue.zero)
                    residual0 = pl.set_validshape(padded_residual0, MX_M_TILE, N_TILE)
                    padded_residual1 = pl.fillpad(residual1, pad_value=pl.PadValue.zero)
                    residual1 = pl.set_validshape(padded_residual1, MX_M_TILE, N_TILE)
                    padded_residual2 = pl.fillpad(residual2, pad_value=pl.PadValue.zero)
                    residual2 = pl.set_validshape(padded_residual2, MX_M_TILE, N_TILE)
                    padded_residual3 = pl.fillpad(residual3, pad_value=pl.PadValue.zero)
                    residual3 = pl.set_validshape(padded_residual3, MX_M_TILE, N_TILE)
                    for out_h in pl.unroll(HC_MULT):
                        post_row = pl.tile.slice(post_transposed, [1, MX_M_TILE], [out_h, 0])
                        post_col = pl.reshape(post_row, [MX_M_TILE, 1])
                        mix0_row = pl.tile.slice(mix_transposed, [1, MX_M_TILE], [out_h, 0])
                        mix1_row = pl.tile.slice(mix_transposed, [1, MX_M_TILE], [HC_MULT + out_h, 0])
                        mix2_row = pl.tile.slice(mix_transposed, [1, MX_M_TILE], [2 * HC_MULT + out_h, 0])
                        mix3_row = pl.tile.slice(mix_transposed, [1, MX_M_TILE], [3 * HC_MULT + out_h, 0])
                        mix0_col = pl.reshape(mix0_row, [MX_M_TILE, 1])
                        mix1_col = pl.reshape(mix1_row, [MX_M_TILE, 1])
                        mix2_col = pl.reshape(mix2_row, [MX_M_TILE, 1])
                        mix3_col = pl.reshape(mix3_row, [MX_M_TILE, 1])
                        value = pl.row_expand_mul(x_tile, post_col)
                        mixed_residual0 = pl.row_expand_mul(residual0, mix0_col)
                        value = pl.add(value, mixed_residual0)
                        mixed_residual1 = pl.row_expand_mul(residual1, mix1_col)
                        value = pl.add(value, mixed_residual1)
                        mixed_residual2 = pl.row_expand_mul(residual2, mix2_col)
                        value = pl.add(value, mixed_residual2)
                        mixed_residual3 = pl.row_expand_mul(residual3, mix3_col)
                        value = pl.add(value, mixed_residual3)
                        output_bf16 = pl.cast(value, pl.BF16, mode="rint")
                        output_tile = pl.cast(output_bf16, pl.FP32)
                        valid_output = pl.set_validshape(output_tile, capacity_rows, N_TILE)
                        pl.store(valid_output, [t0, out_h * D + n0], output_flat)
    return sublayer, output


@pl.jit.inline(auto_scope=False)
def o_proj_hc_post(
    attended: pl.Tensor,
    wo_a: pl.Tensor,
    wo_a_scale: pl.Tensor,
    wo_b: pl.Tensor,
    wo_b_scale: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    sublayer: pl.Tensor,
    residual: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    """Compose O-A and the TP1 O-B/HC epilogue in private M16 row tiles."""
    capacity = pl.tensor.dim(attended, 0)
    if TP_SIZE == 1 and capacity > 64 and capacity <= 80 and num_tokens == capacity:
        pooled_latent_payload = pl.create_tensor([80, LOCAL_O_WIDTH], dtype=pl.FP8E4M3FN)
        pooled_latent_scale_backing = pl.create_tensor([1, 80 * (LOCAL_O_WIDTH // 32)], dtype=pl.FP8E8M0)
        pooled_latent_payload, pooled_latent_scale_backing = _grouped_output_rope_to_mx_m80(
            attended, rope_cos, rope_sin, wo_a, wo_a_scale,
            pooled_latent_payload, pooled_latent_scale_backing, num_tokens,
        )
        _ob_hc_post_from_mx_m80(
            pooled_latent_payload, pooled_latent_scale_backing, wo_b, wo_b_scale,
            sublayer, residual, post_mix, residual_mix, output, num_tokens,
        )
    else:
        for mt in pl.parallel((capacity + MX_M_TILE - 1) // MX_M_TILE):
            t0 = mt * MX_M_TILE
            rows = pl.min(MX_M_TILE, capacity - t0)
            active = pl.min(rows, num_tokens - t0)
            attended_rows = pl.slice(attended, [rows, LOCAL_H * HEAD_DIM], [t0, 0])
            cos_rows = pl.slice(rope_cos, [rows, ROPE_DIM // 2], [t0, 0])
            sin_rows = pl.slice(rope_sin, [rows, ROPE_DIM // 2], [t0, 0])
            sublayer_rows = pl.slice(sublayer, [rows, D], [t0, 0])
            residual_rows = pl.slice(residual, [rows, HC_MULT, D], [t0, 0, 0])
            post_rows = pl.slice(post_mix, [rows, HC_MULT], [t0, 0])
            mix_rows = pl.slice(residual_mix, [rows, HC_MULT, HC_MULT], [t0, 0, 0])
            output_rows = pl.slice(output, [rows, HC_MULT, D], [t0, 0, 0])
            latent_payload = pl.create_tensor([MX_M_TILE, LOCAL_O_WIDTH], dtype=pl.FP8E4M3FN)
            latent_scale_backing = pl.create_tensor([1, MX_M_TILE * (LOCAL_O_WIDTH // 32)], dtype=pl.FP8E8M0)
            latent_payload, latent_scale_backing = _grouped_output_rope_to_mx(
                attended_rows, cos_rows, sin_rows, wo_a, wo_a_scale, latent_payload, latent_scale_backing, active,
            )
            _ob_hc_post_from_mx(
                latent_payload, latent_scale_backing, wo_b, wo_b_scale,
                sublayer_rows, residual_rows, post_rows, mix_rows, output_rows, active,
            )
    return sublayer, output


def grouped_mxfp8_linear(
    grouped: torch.Tensor,
    wo_a: torch.Tensor,
    wo_a_scale: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Evaluate grouped W8A8 MXFP8 with [G, K, N] weights and MX_B_NN scales."""
    activation, activation_scale = _quantize_mxfp8_activation(grouped, scale_alg=0)
    weight_scale = decode_e8m0(unpack_mx_b_scale(wo_a_scale))
    weight_groups = wo_a.float().unflatten(-2, (-1, MX_GROUP))
    partials = torch.einsum("tgmk,gmkr->tgmr", activation.float(), weight_groups)
    partials = partials * activation_scale.unsqueeze(-1) * weight_scale.unsqueeze(0)
    return partials.sum(dim=2).to(output_dtype or grouped.dtype)

def golden_grouped_output(
    grouped: torch.Tensor, wo_a: torch.Tensor, wo_a_scale: torch.Tensor,
) -> torch.Tensor:
    """Grouped W8A8 MXFP8 output GEMM with one BF16 output cast."""
    return grouped_mxfp8_linear(grouped, wo_a, wo_a_scale, output_dtype=torch.bfloat16)


def golden_o_proj(
    attended: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor | None,
    *,
    wo_a_scale: torch.Tensor,
    output_dtype: torch.dtype | None = None,
    sequential_projection: bool = False,
) -> torch.Tensor:
    rope_dim = cos.shape[-1] * 2
    tail = rope_interleave(attended[..., -rope_dim:], cos, sin, inverse=True)
    attended = torch.cat((attended[..., :-rope_dim], tail), dim=-1)
    groups = wo_a.shape[0]
    grouped = attended.flatten(-2).unflatten(-1, (groups, -1))
    latent = golden_grouped_output(grouped, wo_a, wo_a_scale)
    if sequential_projection and wo_b_scale is not None:
        projected = golden_gemm_bf16_via_mxfp8_to_bf16_sequential(latent.flatten(-2), wo_b, wo_b_scale)
    else:
        projected = golden_mxfp8_linear_grouped_sum(latent.flatten(-2), wo_b, wo_b_scale, output_dtype=torch.bfloat16)
    return projected.to(output_dtype or torch.bfloat16)


@pl.jit.inline(auto_scope=False)
def retire_prefill_signals(
    completion_anchor: pl.Tensor,
    arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Clear ready and consumed signals at a serialized entry boundary."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_signal_retire", allow_early_resolve=False):
        anchor = pl.read(completion_anchor, [0, 0, 0])
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                arrived, peer=group_base + tp_rank, offsets=[peer, 0],
                value=0, op=pld.NotifyOp.Set,
            )
        pl.write(completion_anchor, [0, 0, 0], anchor)
    return completion_anchor


@pl.jit.inline(auto_scope=False)
def prefill_sp_output_reduce_scatter(
    partial: pl.Tensor[[T_DYN, D], pl.FP32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[SP_T_DYN, D], pl.BF16],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Sum FP32 head-TP contributions for this rank's token rows, then cast once."""
    local_tokens = pl.tensor.dim(output, 0)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_reduce_reuse", allow_early_resolve=False) as reuse_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(
                output_arrived,
                offsets=[peer, 0], expected=(attention_epoch - 1) * 2, cmp=pld.WaitCmp.Ge,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_reduce_publish", deps=[reuse_tid]) as publish_tid:
        if num_tokens > 0:
            pld.tensor.put(
                dst=output_window, peer=group_base + tp_rank, src=partial,
                dst_offsets=[0, 0], src_offsets=[0, 0], shape=[num_tokens, D],
                chunk_rows=1, chunk_cols=D_TILE, pipeline=True,
            )
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                output_arrived,
                peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_reduce_wait", deps=[publish_tid],
               allow_early_resolve=False) as wait_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=attention_epoch * 2 - 1, cmp=pld.WaitCmp.Ge)
    with pl.spmd(64, name_hint="sp_reduce_scatter", deps=[wait_tid]) as reduce_tid:
        worker = pl.tile.get_block_idx()
        for tile in pl.range(worker, local_tokens * (D // D_TILE), 64):
            local_row = tile // (D // D_TILE)
            col = tile % (D // D_TILE) * D_TILE
            row = tp_rank * local_tokens + local_row
            acc = pl.tile.full([1, D_TILE], dtype=pl.FP32, value=0.0)
            if row < num_tokens:
                for peer in pl.range(TP_SIZE):
                    value = pld.tile.remote_load(
                        output_window,
                        peer=group_base + peer, offsets=[row, col], shape=[1, D_TILE],
                    )
                    acc = pl.add(acc, value)
            result = pl.cast(acc, pl.BF16, mode="rint")
            output = pl.store(result, [local_row, col], output)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_reduce_release", deps=[reduce_tid]) as release_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                output_arrived,
                peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_reduce_consumed", deps=[release_tid],
               allow_early_resolve=False):
        for peer in pl.range(TP_SIZE):
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=attention_epoch * 2, cmp=pld.WaitCmp.Ge)
        anchor = pl.read(output, [0, 0])
        pl.write(output, [0, 0], anchor)
    return output


@pl.jit.inline(auto_scope=False)
def prefill_tp_output_all_reduce(
    output_partial: pl.Tensor[[T_DYN, D], pl.FP32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_reuse", allow_early_resolve=False) as reuse_tid:
        for peer in pl.range(TP_SIZE):
            previous_epoch = (attention_epoch - 1) * 2
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=previous_epoch, cmp=pld.WaitCmp.Ge)
    with pl.spmd(64, name_hint="prefill_tp_publish", deps=[reuse_tid]) as publish_tid:
        worker = pl.tile.get_block_idx()
        for tile in pl.range(worker, num_tokens * (D // 512), 64):
            row = tile // (D // 512)
            col = tile % (D // 512) * 512
            value = pl.load(output_partial, [row, col], [1, 512])
            pld.tile.remote_store(value, output_window, peer=group_base + tp_rank, offsets=[row, col])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_ready", deps=[publish_tid]) as ready_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                output_arrived,
                peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_wait", deps=[ready_tid],
               allow_early_resolve=False) as wait_tid:
        for peer in pl.range(TP_SIZE):
            ready_epoch = attention_epoch * 2 - 1
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=ready_epoch, cmp=pld.WaitCmp.Ge)
    with pl.spmd(64, name_hint="prefill_tp_reduce", deps=[wait_tid]) as reduce_tid:
        worker = pl.tile.get_block_idx()
        for tile in pl.range(worker, num_tokens * (D // 512), 64):
            row = tile // (D // 512)
            col = tile % (D // 512) * 512
            acc = pl.tile.full([1, 512], dtype=pl.FP32, value=0.0)
            for peer in pl.range(TP_SIZE):
                value = pld.tile.remote_load(
                    output_window,
                    peer=group_base + peer, offsets=[row, col], shape=[1, 512],
                )
                acc = pl.add(acc, value)
            output = pl.store(pl.cast(acc, pl.BF16, mode="rint"), [row, col], output)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_release", deps=[reduce_tid]) as release_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                output_arrived,
                peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_consumed", deps=[release_tid],
               allow_early_resolve=False):
        for peer in pl.range(TP_SIZE):
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=attention_epoch * 2, cmp=pld.WaitCmp.Ge)
        anchor = pl.read(output, [0, 0])
        pl.write(output, [0, 0], anchor)
    return output


def build_o_proj_validation(grouped_max_workers: int):
    """Build the production output projection with a fixed grouped worker limit."""
    @pl.jit
    def o_proj_validation(
        attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
        wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
        wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
        wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
        rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        output: pl.InOut[pl.Tensor[[T_DYN, D], pl.FP32]],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        attended.bind_dynamic(0, T_DYN)
        rope_cos.bind_dynamic(0, T_DYN)
        rope_sin.bind_dynamic(0, T_DYN)
        output.bind_dynamic(0, T_DYN)
        o_proj(
            attended, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
            output, num_tokens, grouped_max_workers=grouped_max_workers,
        )
        return output

    return o_proj_validation


def build_o_proj_specs(active_tokens: int):
    """Build deterministic output-projection fixtures with padded inactive rows."""
    from golden import ScalarSpec, TensorSpec
    from quant import quantize_mxfp8_upward_reference, pack_mx_b_scale, quantize_grouped_mxfp8_weight

    tokens = max(16, (active_tokens + 15) // 16 * 16)
    values = {}

    def inputs():
        if not values:
            generator = torch.Generator().manual_seed(17)
            attended = torch.randn(tokens, LOCAL_H * HEAD_DIM, generator=generator).bfloat16()
            attended[active_tokens:] = float("nan")
            wo_a = torch.randn(LOCAL_O_GROUPS, O_LORA, O_GROUP_IN, generator=generator) / O_GROUP_IN**0.5
            wo_a, wo_a_scale = quantize_grouped_mxfp8_weight(wo_a.bfloat16())
            wo_b = torch.randn(D, LOCAL_O_WIDTH, generator=generator) / LOCAL_O_WIDTH**0.5
            wo_b, wo_b_scale = quantize_mxfp8_upward_reference(wo_b)
            frequencies = 10000.0 ** (-torch.arange(0, ROPE_DIM, 2).float() / ROPE_DIM)
            angles = torch.arange(tokens).float()[:, None] * frequencies
            values.update(
                attended=attended, wo_a=wo_a, wo_a_scale=wo_a_scale,
                wo_b=wo_b.T.contiguous(),
                wo_b_scale=pack_mx_b_scale(wo_b_scale.T.contiguous()).view(torch.float8_e8m0fnu),
                rope_cos=angles.cos(), rope_sin=angles.sin(),
            )
        return values

    shapes = (
        ("attended", [tokens, LOCAL_H * HEAD_DIM], torch.bfloat16),
        ("wo_a", [LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], torch.float8_e4m3fn),
        ("wo_a_scale", [LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], torch.float8_e8m0fnu),
        ("wo_b", [LOCAL_O_WIDTH, D], torch.float8_e4m3fn),
        ("wo_b_scale", [LOCAL_O_WIDTH // 32, D], torch.float8_e8m0fnu),
        ("rope_cos", [tokens, ROPE_DIM // 2], torch.float32),
        ("rope_sin", [tokens, ROPE_DIM // 2], torch.float32),
    )
    specs = [TensorSpec(name, shape, dtype, init_value=lambda name=name: inputs()[name])
             for name, shape, dtype in shapes]
    return specs + [
        TensorSpec("output", [tokens, D], torch.float32, init_value=13.0),
        ScalarSpec("num_tokens", torch.int32, active_tokens),
    ]


def golden_o_proj_validation(tensors):
    """Apply the existing projection reference only to active rows."""
    active = int(tensors["num_tokens"])
    if active == 0:
        return
    attended = tensors["attended"][:active].unflatten(-1, (LOCAL_H, HEAD_DIM))
    result = golden_o_proj(
        attended, tensors["rope_cos"][:active], tensors["rope_sin"][:active],
        tensors["wo_a"], tensors["wo_b"], tensors["wo_b_scale"],
        wo_a_scale=tensors["wo_a_scale"], output_dtype=torch.float32, sequential_projection=True,
    )
    tensors["output"][:active].copy_(result)


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="V4.1 output projection branch and padding validation")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--case", choices=["all", "empty", "fast", "low-workers", "tail"], default="all")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    args = parser.parse_args()
    cases = (("empty", 0, GROUPED_WORKER_TILE), ("fast", 12, GROUPED_WORKER_TILE), ("low-workers", 12, 8), ("tail", 17, GROUPED_WORKER_TILE))
    for name, active, workers in cases:
        if args.case not in ("all", name):
            continue
        print(f"[O_PROJ] {name}: active={active}, grouped_max_workers={workers}", flush=True)
        active_compare = ratio_allclose(atol=1e-2, rtol=1.0 / 128)

        def compare(actual, expected, **kwargs):
            if active == 0:
                unchanged = torch.equal(actual, expected)
                return unchanged, f"empty input leaves output unchanged={unchanged}"
            passed, detail = active_compare(actual[:active], expected[:active], **kwargs)
            inactive_ok = torch.equal(actual[active:], expected[active:])
            return passed and inactive_ok, f"{detail}; inactive rows unchanged={inactive_ok}"

        result = run(
            fn=build_o_proj_validation(workers),
            specs=build_o_proj_specs(active),
            golden_fn=golden_o_proj_validation,
            compare_fn={"output": compare},
            config={
                "backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device,
            },
            compile_only=args.compile_only,
            save_data=args.save_data,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
