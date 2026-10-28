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

from pathlib import Path

from gemm import MX_M_TILE

import torch

from quant import quantize_bf16_to_mxfp8, MX_GROUP, _quantize_mxfp8_activation, decode_e8m0, unpack_mx_b_scale

from qkv_proj_rope import golden_rope_interleave as rope_interleave
from gemm import golden_gemm_bf16_via_mxfp8_to_bf16_sequential, golden_mxfp8_linear_grouped_sum


import pypto.language as pl
from pypto.runtime import pto_isa_include_dir
from hc import hc_post
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
O_A_N_TILE = 256
O_B_N_TILE = 256
K_TILE = 256
D_TILE = 512


@pl.jit.extern(core_type="aic", source=Path(__file__).parent / "kernels" / "o_proj_a.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _o_proj_a(
    payload: pl.Tensor[[LOCAL_O_GROUPS * 80, O_GROUP_IN], pl.FP8E4M3FN],
    weight: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    activation_scale: pl.Tensor[[LOCAL_O_GROUPS, 80 * (O_GROUP_IN // 32)], pl.FP8E8M0],
    weight_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    output: pl.Out[pl.Tensor[[80, LOCAL_O_WIDTH], pl.BF16]],
    output_width: pl.Scalar[pl.INDEX],
) -> pl.Tensor:
    """Project grouped NZ weights and MX scales into concatenated BF16 columns."""
    ...


@pl.jit.extern(core_type="aic", source=Path(__file__).parent / "kernels" / "o_proj_b.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _o_proj_b(
    payload: pl.Tensor[[80, LOCAL_O_WIDTH], pl.FP8E4M3FN],
    weight: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    activation_scale: pl.Tensor[[1, 80 * (LOCAL_O_WIDTH // 32)], pl.FP8E8M0],
    weight_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    rows: pl.Scalar[pl.INDEX], reduction: pl.Scalar[pl.INDEX],
) -> pl.Tensor:
    """Project an 80-row group; scales contain MX_A_ZZ and MX_B_NN bytes."""
    ...


@pl.jit.inline(auto_scope=False)
def o_proj(
    attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Project row groups to BF16; both weights hold NZ bytes in logical [K, N] shapes."""
    total_rows = pl.tensor.dim(output, 0)
    for group_start in pl.parallel(0, total_rows, 80):
        group_rows = pl.min(80, total_rows - group_start)
        active_rows = pl.max(pl.min(group_rows, num_tokens - group_start), 0)
        attended_group = pl.tensor.slice(attended, [group_rows, LOCAL_H * HEAD_DIM], [group_start, 0])
        cosine_group = pl.tensor.slice(rope_cos, [group_rows, ROPE_DIM // 2], [group_start, 0])
        sine_group = pl.tensor.slice(rope_sin, [group_rows, ROPE_DIM // 2], [group_start, 0])
        output_group = pl.tensor.slice(output, [group_rows, D], [group_start, 0])
        # Inverse RoPE and input quantization.
        padded_rows = 80
        row_tiles = padded_rows // MX_M_TILE
        last_input_row = pl.max(pl.tensor.dim(attended_group, 0) - 1, 0)
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
                rows = pl.max(pl.min(MX_M_TILE, active_rows - t0), 0)
                input_t0 = pl.min(t0, last_input_row)
                cosine = pl.load(cosine_group, [input_t0, 0], [MX_M_TILE, ROPE_DIM // 2], valid_shape=[rows, ROPE_DIM // 2])
                cosine = pl.fillpad(cosine, pad_value=pl.PadValue.zero)
                cosine = pl.set_validshape(cosine, MX_M_TILE, ROPE_DIM // 2)
                sine = pl.load(sine_group, [input_t0, 0], [MX_M_TILE, ROPE_DIM // 2], valid_shape=[rows, ROPE_DIM // 2])
                sine = pl.fillpad(sine, pad_value=pl.PadValue.zero)
                valid_sine = pl.set_validshape(sine, MX_M_TILE, ROPE_DIM // 2)
                sine = pl.mul(valid_sine, -1.0)
                for chunk in pl.unroll(2):
                    k0 = (chunk * (O_GROUP_IN // (2 * K_TILE)) + block) * K_TILE
                    input_n = quant_group * O_GROUP_IN + k0
                    raw_source = pl.load(attended_group, [input_t0, input_n], [MX_M_TILE, K_TILE], valid_shape=[rows, K_TILE])
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
        # O-A projection.
        projected = pl.create_tensor([80, LOCAL_O_WIDTH], dtype=pl.BF16)
        oa_workers = pl.min(LOCAL_O_GROUPS * (O_LORA // O_A_N_TILE), pl.system.available_cluster_count())
        oa_workers = pl.max(1, oa_workers)
        with pl.spmd(oa_workers, name_hint="proj_o_a", sync_start=True):
            projected = _o_proj_a(payload_all, wo_a, scales_all, wo_a_scale, projected, LOCAL_O_WIDTH)
        # Quantize the O-A result for O-B.
        payload = pl.create_tensor([80, LOCAL_O_WIDTH], dtype=pl.FP8E4M3FN)
        activation_scale_backing = pl.create_tensor([1, 80 * (LOCAL_O_WIDTH // 32)], dtype=pl.FP8E8M0)
        output_blocks = row_tiles * (LOCAL_O_WIDTH // K_TILE)
        output_workers = pl.min(pl.min(output_blocks, 64), 2 * pl.system.available_cluster_count())
        for worker in pl.spmd(output_workers, name_hint="quant_o_a"):
            for work in pl.range(worker, output_blocks, output_workers):
                t0 = work // (LOCAL_O_WIDTH // K_TILE) * MX_M_TILE
                k0 = work % (LOCAL_O_WIDTH // K_TILE) * K_TILE
                rows = pl.max(pl.min(MX_M_TILE, active_rows - t0), 0)
                output_quantized, output_packed_scale = quantize_bf16_to_mxfp8(projected, t0, k0, rows, MX_M_TILE)
                payload = pl.store(output_quantized, [t0, k0], payload)
                scale_offset = t0 * (LOCAL_O_WIDTH // 32) + k0 // K_TILE * MX_M_TILE * (K_TILE // 32)
                output_packed_flat = pl.reshape(output_packed_scale, [1, MX_M_TILE * (K_TILE // 32)])
                activation_scale_backing = pl.store(output_packed_flat, [0, scale_offset], activation_scale_backing)
        # O-B projection.
        capacity = pl.tensor.dim(output_group, 0)
        with pl.spmd(D // O_B_N_TILE, name_hint="proj_o_b", sync_start=True):
            output_group = _o_proj_b(
                payload, wo_b, activation_scale_backing, wo_b_scale, output_group, capacity, LOCAL_O_WIDTH,
            )
    return output


def grouped_mxfp8_linear(
    grouped: torch.Tensor,
    wo_a: torch.Tensor,
    wo_a_scale: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Evaluate grouped W8A8 MXFP8 with NZ weights and MX_B_NN scales."""
    from quant import unpack_fp8_nz

    wo_a = unpack_fp8_nz(wo_a)
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
    from quant import unpack_fp8_nz

    wo_b = unpack_fp8_nz(wo_b)
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
    partial: pl.Tensor[[T_DYN, D], pl.BF16],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[SP_T_DYN, D], pl.BF16],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Sum BF16 head-TP contributions in FP32 for this rank's token rows."""
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
                    acc = pl.add(acc, pl.cast(value, pl.FP32))
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
    output_partial: pl.Tensor[[T_DYN, D], pl.BF16],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
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
                acc = pl.add(acc, pl.cast(value, pl.FP32))
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


@pl.jit
def o_proj_validation(
    attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    output: pl.InOut[pl.Tensor[[T_DYN, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    attended.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    o_proj(
        attended, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
        output, num_tokens,
    )
    return output

@pl.jit
def o_proj_with_hc_validation(
    attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    sublayer: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    attended.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    sublayer.bind_dynamic(0, T_DYN)
    residual.bind_dynamic(0, T_DYN)
    post_mix.bind_dynamic(0, T_DYN)
    residual_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    o_proj(
        attended, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
        sublayer, num_tokens,
    )
    hc_post_deps = pl.array.create(2, pl.TASK_ID)
    hc_post(sublayer, residual, post_mix, residual_mix, output, deps=hc_post_deps)
    return sublayer, output


def build_o_proj_specs(active_tokens: int, *, pad_to_tile: bool = True):
    """Build deterministic output-projection fixtures with padded inactive rows."""
    from golden import ScalarSpec, TensorSpec
    from quant import pack_fp8_nz, quantize_mxfp8_upward_reference, pack_mx_b_scale, quantize_grouped_mxfp8_weight

    tokens = max(16, (active_tokens + 15) // 16 * 16) if pad_to_tile else active_tokens
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
                wo_b=pack_fp8_nz(wo_b.T.contiguous()),
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
        TensorSpec("output", [tokens, D], torch.bfloat16, init_value=13.0),
        ScalarSpec("num_tokens", torch.int32, active_tokens),
    ]


def golden_o_proj_validation(tensors):
    """Project active rows to BF16 and zero inactive rows."""
    active = int(tensors["num_tokens"])
    tensors["output"].zero_()
    if active == 0:
        return
    attended = tensors["attended"][:active].unflatten(-1, (LOCAL_H, HEAD_DIM))
    result = golden_o_proj(
        attended, tensors["rope_cos"][:active], tensors["rope_sin"][:active],
        tensors["wo_a"], tensors["wo_b"], tensors["wo_b_scale"],
        wo_a_scale=tensors["wo_a_scale"], output_dtype=torch.bfloat16, sequential_projection=True,
    )
    tensors["output"][:active].copy_(result)


def build_o_proj_hc_post_specs(active_tokens: int = 72, *, pad_to_tile: bool = False):
    """Add genuine FP32 HC residuals and mixes to active or padded projection inputs."""
    from golden import ScalarSpec, TensorSpec

    specs = build_o_proj_specs(active_tokens, pad_to_tile=pad_to_tile)
    tokens = specs[0].shape[0]
    specs = [spec for spec in specs if spec.name not in ("output", "num_tokens")]
    generator = torch.Generator().manual_seed(29)
    return specs + [
        TensorSpec("sublayer", [tokens, D], torch.bfloat16),
        TensorSpec("residual", [tokens, HC_MULT, D], torch.float32,
                   init_value=lambda: torch.randn(tokens, HC_MULT, D, generator=generator)),
        TensorSpec("post_mix", [tokens, HC_MULT], torch.float32,
                   init_value=lambda: 2 * torch.sigmoid(torch.randn(tokens, HC_MULT, generator=generator))),
        TensorSpec("residual_mix", [tokens, HC_MULT, HC_MULT], torch.float32,
                   init_value=lambda: torch.randn(tokens, HC_MULT, HC_MULT, generator=generator)),
        TensorSpec("output", [tokens, HC_MULT, D], torch.float32),
        ScalarSpec("num_tokens", torch.int32, active_tokens),
    ]


def golden_o_proj_hc_post_validation(tensors):
    """Compute independent BF16 output projection followed by FP32 HC mixing."""
    from hc import golden_hc_post

    projected = dict(tensors, output=tensors["sublayer"])
    golden_o_proj_validation(projected)
    tensors["output"].copy_(golden_hc_post(
        tensors["sublayer"], tensors["residual"], tensors["post_mix"], tensors["residual_mix"],
    ))


def compare_o_proj_hc_post(actual, _expected, *, actual_outputs, expected_outputs, inputs, **kwargs):
    """Check FP32 HC mixing independently of the separately checked BF16 projection."""
    from golden import ratio_allclose
    from hc import golden_hc_post

    expected = golden_hc_post(
        actual_outputs["sublayer"], inputs["residual"], inputs["post_mix"], inputs["residual_mix"],
    )
    compare = ratio_allclose(atol=1e-6, rtol=1e-5, max_error_ratio=0.0)
    return compare(
        actual, expected, actual_outputs=actual_outputs, expected_outputs=expected_outputs, inputs=inputs, **kwargs,
    )


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="V4.1 output projection branch and padding validation")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tp", type=int, choices=[1, 2, 4, 8], default=TP_SIZE)
    parser.add_argument("--case", choices=["all", "empty", "fast", "tail", "hc-post"], default="all")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    args = parser.parse_args()
    cases = (("empty", 0), ("fast", 12), ("tail", 17))
    for name, active in cases:
        if args.case not in ("all", name):
            continue
        print(f"[O_PROJ] {name}: active={active}", flush=True)
        active_compare = ratio_allclose(atol=1e-2, rtol=1.0 / 128)

        def compare(actual, expected, **kwargs):
            if active == 0:
                zeroed = torch.equal(actual, expected)
                return zeroed, f"empty input zeros output={zeroed}"
            passed, detail = active_compare(actual[:active], expected[:active], **kwargs)
            inactive_ok = torch.equal(actual[active:], expected[active:])
            return passed and inactive_ok, f"{detail}; inactive rows zeroed={inactive_ok}"

        result = run(
            fn=o_proj_validation,
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
    if args.case in ("all", "hc-post"):
        for active, padded in ((72, False), (17, True), (0, True), (81, True)):
            print(f"[O_PROJ] hc-post: active={active}, padded={padded}, FP32 HC post", flush=True)
            result = run(
                fn=o_proj_with_hc_validation,
                specs=build_o_proj_hc_post_specs(active, pad_to_tile=padded),
                golden_fn=golden_o_proj_hc_post_validation,
                compare_fn={
                    "sublayer": ratio_allclose(atol=1e-2, rtol=1.0 / 128, valid_rows=active, zero_tail=True),
                    "output": compare_o_proj_hc_post,
                },
                config={
                    "backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device,
                },
                compile_only=args.compile_only,
                save_data=args.save_data,
            )
            if not result.passed:
                raise SystemExit(result.error or 1)
