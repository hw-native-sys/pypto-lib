# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared BF16 and MXFP8 GEMM kernels and Torch references."""

import torch

from quant import (
    MX_GROUP, QUANT_K_TILE, _quantize_mxfp8_activation, decode_e8m0,
    quantize_bf16_to_mxfp8, unpack_mx_b_scale,
)

import pypto.language as pl


# tiling
MX_M_TILE = 16
MX_K_TILE = QUANT_K_TILE


@pl.jit.inline(auto_scope=False)
def gemm_mxfp8_to_bf16(
    payload: pl.Tensor,
    activation_scale_backing: pl.Tensor,
    weight: pl.Tensor,
    weight_scale: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    t0: pl.Scalar[pl.INDEX],
    weight_row: pl.Scalar[pl.INDEX],
    output_col: pl.Scalar[pl.INDEX],
    width: pl.constexpr,
    output_width: pl.constexpr,
    name_hint: pl.constexpr,
    n_tile: pl.constexpr,
    k_tile: pl.constexpr,
    group_count: pl.constexpr,
    scale_row: pl.Scalar[pl.INDEX],
    payload_row: pl.Scalar[pl.INDEX],
    m_tile: pl.constexpr,
    max_workers: pl.constexpr = 2147483647,
    k_stages: pl.constexpr = 2,
):
    """Project one prequantized row tile into BF16 output."""
    scale_slice = activation_scale_backing[scale_row:scale_row + 1, :]
    activation_scale = pl.tensor.view(scale_slice, [m_tile, width // 32], layout=pl.MX_A_ZZ)
    scale_group_size = (width // 32) * output_width
    scale_backing_flat = pl.tensor.view(weight_scale, [1, group_count * scale_group_size], layout=pl.ND)
    scale_group_offset = weight_row // 32 * output_width
    scale_group_end = scale_group_offset + scale_group_size
    scale_group_backing = scale_backing_flat[0:1, scale_group_offset:scale_group_end]
    scale_group = pl.tensor.view(scale_group_backing, [width // 32, output_width], layout=pl.MX_B_NN)
    available_clusters = pl.system.available_cluster_count()
    workers = pl.min(pl.min(output_width // n_tile, max_workers), available_clusters)
    for worker in pl.spmd(workers, name_hint=name_hint, sync_start=True):
        full_a = pl.load(payload, [payload_row, 0], [m_tile, width], target_memory=pl.MemorySpace.Mat)
        block_num = pl.tile.get_block_num()
        for wave in pl.range((output_width // n_tile + block_num - 1) // block_num):
            block = wave * block_num + worker
            if block < output_width // n_tile:
                n0 = block * n_tile
                rows = pl.min(m_tile, num_tokens - t0)
                if m_tile == 80:
                    a0 = full_a[:, :k_tile]
                else:
                    a0 = pl.load(payload, [payload_row, 0], [m_tile, k_tile])
                sa0 = pl.load(activation_scale, [0, 0], [m_tile, k_tile // 32])
                b0 = pl.load(weight, [weight_row, n0], [k_tile, n_tile])
                sb0 = pl.load(scale_group, [0, n0], [k_tile // 32, n_tile])
                acc = pl.matmul_mx(a0, sa0, b0, sb0)
                for k0 in pl.pipeline(k_tile, width, k_tile, stage=k_stages):
                    if m_tile == 80:
                        a = pl.tile.slice(full_a, [m_tile, k_tile], [0, k0])
                    else:
                        a = pl.load(payload, [payload_row, k0], [m_tile, k_tile])
                    sa = pl.load(activation_scale, [0, k0 // 32], [m_tile, k_tile // 32])
                    b = pl.load(weight, [weight_row + k0, n0], [k_tile, n_tile])
                    sb = pl.load(scale_group, [k0 // 32, n0], [k_tile // 32, n_tile])
                    acc = pl.matmul_mx_acc(acc, a, sa, b, sb)
                output_n = output_col + n0
                if m_tile == 80:
                    valid_acc = pl.set_validshape(acc, rows, n_tile)
                    output = pl.store(valid_acc, [t0, output_n], output)
                else:
                    value = pl.cast(acc, pl.BF16, mode="rint")
                    valid_bf16 = pl.set_validshape(value, rows, n_tile)
                    output = pl.store(valid_bf16, [t0, output_n], output)
    return output


@pl.jit.inline(auto_scope=False)
def gemm_mxfp8_to_mxfp8_m80_dual(
    payload: pl.Tensor,
    scale_backing: pl.Tensor,
    weight_flat: pl.Tensor,
    scale: pl.Tensor,
    output: pl.Tensor,
    output_scale: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    group: pl.Scalar[pl.INDEX],
    width: pl.constexpr,
    output_width: pl.constexpr,
    name_hint: pl.constexpr,
    n_tile: pl.constexpr,
    k_tile: pl.constexpr,
    group_count: pl.constexpr,
    pack_cols: pl.constexpr,
):
    """Project 80 rows and quantize M16/N128 shards on two AIVs."""
    activation_scale = pl.tensor.view(scale_backing, [80, width // 32], layout=pl.MX_A_ZZ)
    scale_group_size = (width // 32) * output_width
    scale_backing_flat = pl.tensor.view(scale, [1, group_count * scale_group_size], layout=pl.ND)
    scale_group_offset = (group * width) // 32 * output_width
    scale_group_backing = pl.tensor.slice(scale_backing_flat, [1, scale_group_size], [0, scale_group_offset])
    scale_group = pl.tensor.view(scale_group_backing, [width // 32, output_width], layout=pl.MX_B_NN)
    blocks = output_width // n_tile
    workers = pl.min(blocks, pl.system.available_cluster_count())
    for worker in pl.spmd(workers, name_hint=name_hint, sync_start=True):
        full_a = pl.load(payload, [0, 0], [80, width], target_memory=pl.MemorySpace.Mat)
        block_num = pl.tile.get_block_num()
        for ni in pl.range(0, output_width // n_tile):
            if ni % block_num == worker:
                n0 = ni * n_tile
                a0 = full_a[:, :k_tile]
                sa0 = pl.load(activation_scale, [0, 0], [80, k_tile // 32])
                b0 = pl.load(weight_flat, [(group * width), n0], [k_tile, n_tile])
                sb0 = pl.load(scale_group, [0, n0], [k_tile // 32, n_tile])
                acc = pl.matmul_mx(a0, sa0, b0, sb0)
                for k0 in pl.pipeline(k_tile, width, k_tile, stage=2):
                    a = pl.tile.slice(full_a, [80, k_tile], [0, k0])
                    sa = pl.load(activation_scale, [0, k0 // 32], [80, k_tile // 32])
                    b = pl.load(weight_flat, [(group * width) + k0, n0], [k_tile, n_tile])
                    sb = pl.load(scale_group, [k0 // 32, n0], [k_tile // 32, n_tile])
                    acc = pl.matmul_mx_acc(acc, a, sa, b, sb)
                for shard_lane in pl.split_aiv(2, mode=pl.SplitMode.LEFT_RIGHT):
                    full_vector = pl.aiv_shard(acc)
                for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    for t0 in pl.range(0, 80, MX_M_TILE):
                        rows = pl.max(pl.min(MX_M_TILE, num_tokens - t0), 0)
                        acc_rows = pl.tile.slice(full_vector, [MX_M_TILE, 128], [t0, 0])
                        rounded_value = pl.cast(acc_rows, pl.BF16, mode="rint")
                        rounded = pl.set_validshape(rounded_value, rows, 128)
                        source = pl.fillpad(rounded, pad_value=pl.PadValue.zero)
                        source = pl.set_validshape(source, MX_M_TILE, 128)
                        source_fp32 = pl.cast(source, pl.FP32)
                        groups = pl.reshape(source_fp32, [MX_M_TILE * (128 // 32), 32])
                        reduce_tmp = pl.create_tile([MX_M_TILE * (128 // 32), 32], dtype=pl.FP32)
                        magnitude = pl.abs(groups)
                        maximum = pl.row_max(magnitude, tmp_tile=reduce_tmp)
                        bits = pl.reinterpret_view(maximum, pl.INT32)
                        biased_exponent = pl.shrs(bits, 23)
                        scale_exponent = pl.sub(biased_exponent, 8)
                        exponent = pl.maximum(scale_exponent, 0)
                        negative_exponent = pl.mul(exponent, -1)
                        reciprocal_exponent = pl.add(negative_exponent, 254)
                        nonzero_exponent = pl.minimum(exponent, 1)
                        reciprocal_code = pl.mul(reciprocal_exponent, nonzero_exponent)
                        reciprocal_bits = pl.shls(reciprocal_code, 23)
                        reciprocal = pl.reinterpret_view(reciprocal_bits, pl.FP32)
                        normalized = pl.row_expand_mul(groups, reciprocal)
                        normalized = pl.maximum(normalized, -448.0)
                        normalized = pl.minimum(normalized, 448.0)
                        payload_fp8 = pl.cast(normalized, pl.FP8E4M3FN, mode="rint")
                        quantized_tile = pl.reshape(payload_fp8, [MX_M_TILE, 128])
                        output_n = group * output_width + n0 + lane * 128
                        output = pl.store(quantized_tile, [t0, output_n], output)
                        sign_bit = pl.shrs(exponent, 7)
                        sign_offset = pl.mul(sign_bit, 256)
                        signed = pl.sub(exponent, sign_offset)
                        signed_codes = pl.cast(signed, pl.INT8)
                        codes = pl.reinterpret_view(signed_codes, pl.UINT8)
                        flat = pl.reshape(codes, [1, MX_M_TILE * (128 // 32)])
                        pack_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                        packed = pl.tmov_x2zz(flat, pack_tmp, group_axis=1, dst_rows=MX_M_TILE, dst_cols=pack_cols)
                        packed_scale = pl.reinterpret_view(packed, pl.FP8E8M0)
                        packed_flat = pl.reshape(packed_scale, [1, MX_M_TILE * (128 // 32)])
                        scale_offset = t0 * (group_count * output_width // 32) + output_n // 32 * MX_M_TILE
                        output_scale = pl.store(packed_flat, [0, scale_offset], output_scale)
    return output, output_scale



@pl.jit.inline(auto_scope=False)
def gemm_bf16_via_mxfp8_to_bf16_staged(
    x: pl.Tensor,
    weight: pl.Tensor,
    scale: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    width: pl.constexpr,
    output_width: pl.constexpr,
    name_hint: pl.constexpr,
    quant_name: pl.constexpr,
    n_tile: pl.constexpr,
    k_tile: pl.constexpr,
    group_count: pl.constexpr = 1,
    max_workers: pl.constexpr = 2147483647,
):
    """Quantize activation tiles once, then run separate MX projection tasks."""
    weight_flat = pl.reshape(weight, [group_count * width, output_width])
    for group in pl.parallel(group_count):
        for mt in pl.parallel((num_tokens + MX_M_TILE - 1) // MX_M_TILE):
            t0 = mt * MX_M_TILE
            payload = pl.create_tensor([MX_M_TILE, width], dtype=pl.FP8E4M3FN)
            scale_size = MX_M_TILE * (width // 32)
            scale_backing = pl.create_tensor([1, scale_size], dtype=pl.FP8E8M0)
            quant_chunks = 1
            if group_count > 1 and width % (2 * MX_K_TILE) == 0:
                quant_chunks = 2
            for block in pl.spmd(width // (MX_K_TILE * quant_chunks), name_hint=quant_name):
                rows = pl.min(MX_M_TILE, num_tokens - t0)
                for chunk in pl.unroll(2):
                    if chunk == 0 or (group_count > 1 and width % (2 * MX_K_TILE) == 0):
                        k0 = (chunk * (width // (MX_K_TILE * quant_chunks)) + block) * MX_K_TILE
                        input_k = group * width + k0
                        quantized_tile, packed_scale = quantize_bf16_to_mxfp8(x, t0, input_k, rows, MX_M_TILE)
                        payload = pl.store(quantized_tile, [0, k0], payload)
                        scale_offset = k0 // MX_K_TILE * MX_M_TILE * (MX_K_TILE // 32)
                        packed_flat = pl.reshape(packed_scale, [1, MX_M_TILE * (MX_K_TILE // 32)])
                        scale_backing = pl.store(packed_flat, [0, scale_offset], scale_backing)
            output = gemm_mxfp8_to_bf16(
                payload, scale_backing, weight_flat, scale, output, num_tokens,
                t0, group * width, group * output_width,
                width, output_width, name_hint, n_tile, k_tile,  group_count, 0, 0, MX_M_TILE, max_workers, 2,
            )
    return output


@pl.jit.inline(auto_scope=False)
def gemm_bf16_via_mxfp8_to_bf16_fused(
    x: pl.Tensor,
    weight: pl.Tensor,
    scale: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    width: pl.constexpr,
    output_width: pl.constexpr,
    n_tile: pl.constexpr,
    name_hint: pl.constexpr,
):
    """Compute native M32 MXFP8 products with in-task activation quantization."""
    for mt in pl.parallel((num_tokens + 31) // 32):
        for block in pl.spmd(output_width // n_tile, name_hint=name_hint):
            t0 = mt * 32
            n0 = block * n_tile
            rows = pl.min(32, num_tokens - t0)
            a0, sa0 = quantize_bf16_to_mxfp8(x, t0, 0, rows, 32)
            b0 = pl.load(weight, [0, n0], [MX_K_TILE, n_tile])
            sb0 = pl.load(scale, [0, n0], [MX_K_TILE // 32, n_tile])
            acc = pl.matmul_mx(a0, sa0, b0, sb0)
            for kb in pl.range(1, width // MX_K_TILE):
                k0 = kb * MX_K_TILE
                a, sa = quantize_bf16_to_mxfp8(x, t0, k0, rows, 32)
                b = pl.load(weight, [k0, n0], [MX_K_TILE, n_tile])
                sb = pl.load(scale, [k0 // 32, n0], [MX_K_TILE // 32, n_tile])
                acc = pl.matmul_mx_acc(acc, a, sa, b, sb)
            value = pl.cast(acc, target_type=pl.BF16, mode="rint")
            output = pl.store(pl.set_validshape(value, rows, n_tile), [t0, n0], output)
    return output


@pl.jit.inline(auto_scope=False)
def gemm_bf16_to_bf16_staged(
    source: pl.Tensor,
    weight: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
    width: pl.constexpr,
    output_width: pl.constexpr,
    n_tile: pl.constexpr,
    k_tile: pl.constexpr,
    workers: pl.constexpr,
    name_hint: pl.constexpr,
    cast_name: pl.constexpr,
):
    """Stage FP32 GEMM results before the separate BF16 rounding task."""
    tokens = pl.tensor.dim(source, 0)
    accumulated = pl.create_tensor([tokens, output_width], dtype=pl.FP32)
    blocks = (num_tokens + 16 - 1) // 16 * (output_width // n_tile)
    task_workers = pl.max(pl.min(blocks, workers), 1)
    entry_deps = pl.array.create(1, pl.TASK_ID)
    entry_deps[0] = ready
    with pl.spmd(task_workers, name_hint=name_hint, deps=[entry_deps[0]]) as cube_tid:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, blocks, task_workers):
            row = block // (output_width // n_tile) * 16
            col = block % (output_width // n_tile) * n_tile
            rows = pl.min(16, num_tokens - row)
            acc = pl.create_tensor([16, n_tile], dtype=pl.FP32)
            for k in pl.range(0, width, k_tile):
                a = pl.slice(source, [16, k_tile], [row, k], valid_shape=[rows, k_tile])
                # Establish Vector startup before the local Cube result transfer (#2829).
                a = pl.cast(pl.cast(a, pl.FP32), pl.BF16, mode="rint")
                w = pl.slice(weight, [k_tile, n_tile], [k, col])
                acc = pl.matmul_acc(acc, a, w, init_cond=(k == 0))
            value = pl.set_validshape(pl.mul(acc, 1.0), rows, n_tile)
            accumulated[row:row + 16, col:col + n_tile] = value
    with pl.spmd(task_workers, name_hint=cast_name, deps=[cube_tid]) as cast_tid:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, blocks, task_workers):
            row = block // (output_width // n_tile) * 16
            col = block % (output_width // n_tile) * n_tile
            rows = pl.min(16, num_tokens - row)
            value = pl.slice(accumulated, [16, n_tile], [row, col], valid_shape=[rows, n_tile])
            output[row:row + 16, col:col + n_tile] = pl.cast(value, pl.BF16, mode="rint")
    return output, cast_tid


def golden_mxfp8_linear_grouped_sum(
    x: torch.Tensor,
    weight: torch.Tensor,
    packed_weight_scale: torch.Tensor | None,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Evaluate a linear projection with an optional explicit output dtype."""
    if packed_weight_scale is None:
        if output_dtype is not None:
            return torch.matmul(x.float(), weight.float()).to(output_dtype)
        return torch.matmul(x, weight)
    activation, activation_scale = _quantize_mxfp8_activation(x, scale_alg=0)
    logical_scale = unpack_mx_b_scale(packed_weight_scale)
    weight_scale = decode_e8m0(logical_scale)
    weight_groups = weight.float().unflatten(0, (-1, MX_GROUP))
    partials = torch.einsum("...gk,gkn->...gn", activation.float(), weight_groups)
    partials = partials * activation_scale.unsqueeze(-1) * weight_scale
    return partials.sum(dim=-2).to(output_dtype or x.dtype)


def golden_gemm_bf16_via_mxfp8_to_bf16_sequential(x, weight, packed_scale):
    """Apply OCP activation quantization and sequential group-32 FP32 accumulation."""
    payload, scale_a = _quantize_mxfp8_activation(x, scale_alg=0)
    activation, weights = payload.flatten(-2).float(), weight.float()
    scale_b = decode_e8m0(unpack_mx_b_scale(packed_scale))
    result = torch.zeros(x.shape[0], weight.shape[1], dtype=torch.float32)
    for group in range(weight.shape[0] // MX_GROUP):
        start = group * MX_GROUP
        product = activation[:, start : start + MX_GROUP] @ weights[start : start + MX_GROUP]
        scaled = product * scale_a[:, group : group + 1]
        scaled = scaled * scale_b[group : group + 1]
        result = result + scaled
    return result.bfloat16()


def golden_gemm_mxfp8_to_fp32_dequantized(a, a_scale, b, b_scale):
    """Evaluate the FP32 MX matmul golden using logical E8M0 scales."""
    a_codes = a_scale.contiguous().view(torch.uint8)
    b_codes = b_scale.contiguous().view(torch.uint8)
    a_f = a.float() * decode_e8m0(a_codes).repeat_interleave(MX_GROUP, -1)
    b_f = b.float() * decode_e8m0(b_codes).repeat_interleave(MX_GROUP, -2)
    return torch.matmul(a_f.to(torch.float32), b_f.to(torch.float32))
