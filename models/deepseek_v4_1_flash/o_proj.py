# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 inverse RoPE, grouped W8A8 MXFP8 output projection, and TP output reductions."""

import torch

from qkv_proj_rope import golden_rope_interleave as rope_interleave
from quant import grouped_mxfp8_linear, mxfp8_linear

import pypto.language as pl
import pypto.language.distributed as pld

from qkv_proj_rope import (
    K_TILE, MX_M_TILE, N_TILE,
    make_mx_projection, make_rope,
)
from config import (
    DECODE_MAX_TOKENS, PREFILL_MAX_TOKENS,
    FLASH as M, LOCAL_T_DYN, SP_T_DYN, T_DYN, TP_SIZE,
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


D_TILE = 512
_PREFILL_WORKERS = 64


def _make_grouped_output_tile(output_dtype):
    fp32_output = output_dtype == pl.FP32

    @pl.jit.inline
    def tile(
        x: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
        weight: pl.Tensor[[O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
        scale: pl.Tensor[[O_GROUP_IN // 32, O_LORA], pl.FP8E8M0, pl.MX_B_NN],
        output: pl.Tensor[[T_DYN, LOCAL_O_WIDTH], output_dtype],
        t0: pl.Scalar[pl.INDEX],
        n0: pl.Scalar[pl.INDEX],
        group: pl.Scalar[pl.INDEX],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        local_n = n0
        output_n = group * O_LORA + local_n
        rows = pl.min(MX_M_TILE, num_tokens - t0)
        first = pl.load(x, [t0, group * O_GROUP_IN], [MX_M_TILE, K_TILE], valid_shape=[rows, K_TILE])
        first = pl.set_validshape(pl.fillpad(first, pad_value=pl.PadValue.zero), MX_M_TILE, K_TILE)
        fq = pl.reshape(pl.cast(first, pl.FP32), [MX_M_TILE * (K_TILE // 32), 32])
        fq_tmp = pl.create_tile([MX_M_TILE * (K_TILE // 32), 32], dtype=pl.FP32)
        fq_max = pl.row_max(pl.abs(fq), tmp_tile=fq_tmp)
        fq_bits = pl.reinterpret_view(fq_max, pl.INT32)
        fq_exp = pl.maximum(pl.sub(pl.shrs(fq_bits, 23), 8), 0)
        # Use a normal divisor for scale code zero, whose payload is zero.
        fq_scale = pl.reinterpret_view(pl.shls(pl.maximum(fq_exp, 1), 23), pl.FP32)
        fq_normalized = pl.row_expand_div(fq, fq_scale)
        fq_normalized = pl.row_expand_mul(fq_normalized, pl.cast(pl.minimum(fq_exp, 1), pl.FP32))
        fq_normalized = pl.minimum(pl.maximum(fq_normalized, -448.0), 448.0)
        a = pl.reshape(pl.cast(fq_normalized, pl.FP8E4M3FN, mode="rint"), [MX_M_TILE, K_TILE])
        fq_se = pl.sub(fq_exp, pl.mul(pl.shrs(fq_exp, 7), 256))
        fq_codes = pl.reinterpret_view(pl.cast(fq_se, pl.INT8), pl.UINT8)
        fq_flat = pl.reshape(fq_codes, [1, MX_M_TILE * (K_TILE // 32)])
        fq_packed = pl.tmov_x2zz(
            fq_flat, pl.create_tile([1, 96], dtype=pl.UINT8), group_axis=1, dst_rows=MX_M_TILE, dst_cols=8
        )
        sa = pl.reinterpret_view(fq_packed, pl.FP8E8M0)
        b = pl.load(weight, [0, local_n], [K_TILE, N_TILE])
        sb = pl.load(scale, [0, local_n], [K_TILE // 32, N_TILE])
        acc = pl.matmul_mx(a, sa, b, sb)

        for kb in pl.range(1, O_GROUP_IN // K_TILE):
            k0 = kb * K_TILE
            values = pl.load(x, [t0, group * O_GROUP_IN + k0], [MX_M_TILE, K_TILE], valid_shape=[rows, K_TILE])
            values = pl.set_validshape(pl.fillpad(values, pad_value=pl.PadValue.zero), MX_M_TILE, K_TILE)
            q = pl.reshape(pl.cast(values, pl.FP32), [MX_M_TILE * (K_TILE // 32), 32])
            q_tmp = pl.create_tile([MX_M_TILE * (K_TILE // 32), 32], dtype=pl.FP32)
            q_max = pl.row_max(pl.abs(q), tmp_tile=q_tmp)
            q_bits = pl.reinterpret_view(q_max, pl.INT32)
            q_exp = pl.maximum(pl.sub(pl.shrs(q_bits, 23), 8), 0)
            q_scale = pl.reinterpret_view(pl.shls(pl.maximum(q_exp, 1), 23), pl.FP32)
            q_normalized = pl.row_expand_div(q, q_scale)
            q_normalized = pl.row_expand_mul(q_normalized, pl.cast(pl.minimum(q_exp, 1), pl.FP32))
            q_normalized = pl.minimum(pl.maximum(q_normalized, -448.0), 448.0)
            a = pl.reshape(pl.cast(q_normalized, pl.FP8E4M3FN, mode="rint"), [MX_M_TILE, K_TILE])
            q_se = pl.sub(q_exp, pl.mul(pl.shrs(q_exp, 7), 256))
            q_codes = pl.reinterpret_view(pl.cast(q_se, pl.INT8), pl.UINT8)
            q_flat = pl.reshape(q_codes, [1, MX_M_TILE * (K_TILE // 32)])
            q_packed = pl.tmov_x2zz(
                q_flat, pl.create_tile([1, 96], dtype=pl.UINT8), group_axis=1, dst_rows=MX_M_TILE, dst_cols=8
            )
            sa = pl.reinterpret_view(q_packed, pl.FP8E8M0)
            b = pl.load(weight, [k0, local_n], [K_TILE, N_TILE])
            sb = pl.load(scale, [k0 // 32, local_n], [K_TILE // 32, N_TILE])
            acc = pl.matmul_mx_acc(acc, a, sa, b, sb)

        if fp32_output:
            output = pl.store(pl.set_validshape(pl.mul(acc, 1.0), rows, N_TILE), [t0, output_n], output)
        else:
            value = pl.cast(acc, pl.BF16, mode="rint")
            output = pl.store(pl.set_validshape(value, rows, N_TILE), [t0, output_n], output)
        return output

    return tile


_grouped_output_tile = _make_grouped_output_tile(pl.BF16)
_grouped_output_fp32_tile = _make_grouped_output_tile(pl.FP32)


@pl.jit.inline(auto_scope=False)
def grouped_output(
    x: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    weight: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    output: pl.Tensor[[T_DYN, LOCAL_O_WIDTH], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Apply grouped W8A8 MXFP8 with FP32 accumulation and a BF16 boundary."""
    for group in pl.parallel(LOCAL_O_GROUPS):
        weight_slice = weight[group:group + 1, :, :]
        weight_group = pl.reshape(weight_slice, [O_GROUP_IN, O_LORA])
        scale_slice = scale[group:group + 1, :, :]
        scale_mx: pl.Tensor[[O_GROUP_IN // 32, O_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.reshape(
            scale_slice, [O_GROUP_IN // 32, O_LORA]
        )
        for mt in pl.parallel((num_tokens + MX_M_TILE - 1) // MX_M_TILE):
            for nb in pl.spmd(O_LORA // N_TILE, name_hint="attention_grouped_o_a"):
                _grouped_output_tile(
                    x, weight_group, scale_mx, output, mt * MX_M_TILE,
                    nb * N_TILE, group, num_tokens,
                )
    return output


@pl.jit.inline(auto_scope=False)
def _staged_grouped_output(
    x: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    weight: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    accumulated: pl.Tensor[[T_DYN, LOCAL_O_WIDTH], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    for group in pl.parallel(LOCAL_O_GROUPS):
        weight_slice = weight[group:group + 1, :, :]
        weight_group = pl.reshape(weight_slice, [O_GROUP_IN, O_LORA])
        scale_slice = scale[group:group + 1, :, :]
        scale_mx: pl.Tensor[[O_GROUP_IN // 32, O_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.reshape(
            scale_slice, [O_GROUP_IN // 32, O_LORA]
        )
        for mt in pl.parallel((num_tokens + MX_M_TILE - 1) // MX_M_TILE):
            for nb in pl.spmd(O_LORA // N_TILE, name_hint="prefill_grouped_o_a_cube"):
                _grouped_output_fp32_tile(
                    x, weight_group, scale_mx, accumulated, mt * MX_M_TILE,
                    nb * N_TILE, group, num_tokens,
                )
    return accumulated


@pl.jit.inline(auto_scope=False)
def _cast_grouped_output(
    accumulated: pl.Tensor[[T_DYN, LOCAL_O_WIDTH], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_O_WIDTH], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    with pl.spmd(
        (num_tokens + MX_M_TILE - 1) // MX_M_TILE * (LOCAL_O_WIDTH // N_TILE),
        name_hint="prefill_grouped_o_a_cast",
    ) as cast_tid:
        block = pl.tile.get_block_idx()
        t0 = block // (LOCAL_O_WIDTH // N_TILE) * MX_M_TILE
        n0 = block % (LOCAL_O_WIDTH // N_TILE) * N_TILE
        rows = pl.min(MX_M_TILE, num_tokens - t0)
        value = pl.load(accumulated, [t0, n0], [MX_M_TILE, N_TILE], valid_shape=[rows, N_TILE])
        output = pl.store(pl.cast(value, pl.BF16, mode="rint"), [t0, n0], output)
    return cast_tid


@pl.jit.inline(auto_scope=False)
def prefill_grouped_output(
    x: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    weight: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    output: pl.Tensor[[T_DYN, LOCAL_O_WIDTH], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Stage grouped MXFP8 products in FP32 before the BF16 boundary."""
    tokens = pl.tensor.dim(x, 0)
    accumulated = pl.create_tensor([tokens, LOCAL_O_WIDTH], dtype=pl.FP32)
    _staged_grouped_output(x, weight, scale, accumulated, num_tokens)
    _cast_grouped_output(accumulated, output, num_tokens)
    return output


@pl.jit.inline(auto_scope=False)
def grouped_output_with_deps(
    x: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    weight: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    output: pl.Tensor[[T_DYN, LOCAL_O_WIDTH], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    """Apply grouped Wo-A after an explicit producer dependency."""
    tokens = pl.tensor.dim(x, 0)
    accumulated = pl.create_tensor([tokens, LOCAL_O_WIDTH], dtype=pl.FP32)
    for group in pl.parallel(LOCAL_O_GROUPS):
        weight_slice = weight[group:group + 1, :, :]
        weight_group = pl.reshape(weight_slice, [O_GROUP_IN, O_LORA])
        scale_slice = scale[group:group + 1, :, :]
        scale_mx: pl.Tensor[[O_GROUP_IN // 32, O_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.reshape(
            scale_slice, [O_GROUP_IN // 32, O_LORA]
        )
        for mt in pl.parallel((num_tokens + MX_M_TILE - 1) // MX_M_TILE):
            with pl.spmd(O_LORA // N_TILE, name_hint="c1a_grouped_output", deps=[ready]):
                nb = pl.tile.get_block_idx()
                _grouped_output_fp32_tile(
                    x, weight_group, scale_mx, accumulated, mt * MX_M_TILE,
                    nb * N_TILE, group, num_tokens,
                )
    return _cast_grouped_output(accumulated, output, num_tokens)


def _make_o_proj(rotate, grouped, project):
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
    ):
        tokens = pl.tensor.dim(attended, 0)
        unrotated = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
        rotate(attended, rope_cos, rope_sin, unrotated, num_tokens)
        latent = pl.create_tensor([tokens, LOCAL_O_WIDTH], dtype=pl.BF16)
        grouped(unrotated, wo_a, wo_a_scale, latent, num_tokens)
        project(latent, wo_b, wo_b_scale, output, num_tokens)
        return output

    return o_proj


def make_o_proj_with_deps(rotate, grouped, project):
    """Compose dependency-aware C1A decode output primitives."""

    @pl.jit.inline(auto_scope=False)
    def o_proj_with_deps(
        attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
        wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
        wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
        wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
        rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        output: pl.Tensor[[T_DYN, D], pl.FP32],
        num_tokens: pl.Scalar[pl.INT32],
        ready: pl.Scalar[pl.TASK_ID],
    ):
        tokens = pl.tensor.dim(attended, 0)
        unrotated = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
        rotate_tid = rotate(attended, rope_cos, rope_sin, unrotated, num_tokens, ready)
        latent = pl.create_tensor([tokens, LOCAL_O_WIDTH], dtype=pl.BF16)
        grouped_tid = grouped(unrotated, wo_a, wo_a_scale, latent, num_tokens, rotate_tid)
        output_tid = project(latent, wo_b, wo_b_scale, output, num_tokens, grouped_tid)
        return output_tid

    return o_proj_with_deps


_project_ob = make_mx_projection(LOCAL_O_WIDTH, D, pl.FP32, name_hint="attention_o_b")
_rotate_output = make_rope(LOCAL_H, inverse=True, name_hint="attention_o_rope")
_prefill_rotate_output = make_rope(
    LOCAL_H,
    inverse=True,
    max_workers=_PREFILL_WORKERS,
    name_hint="prefill_attention_o_rope",
)

o_proj = _make_o_proj(_rotate_output, grouped_output, _project_ob)
prefill_o_proj = _make_o_proj(_prefill_rotate_output, prefill_grouped_output, _project_ob)


def golden_grouped_output(
    grouped: torch.Tensor, wo_a: torch.Tensor, wo_a_scale: torch.Tensor,
) -> torch.Tensor:
    """Grouped W8A8 MXFP8 output GEMM with one BF16 output cast."""
    return grouped_mxfp8_linear(grouped, wo_a, wo_a_scale, output_dtype=torch.bfloat16)


def _project_output(
    attended: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor | None,
    *,
    wo_a_scale: torch.Tensor,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    rope_dim = cos.shape[-1] * 2
    tail = rope_interleave(attended[..., -rope_dim:], cos, sin, inverse=True)
    attended = torch.cat((attended[..., :-rope_dim], tail), dim=-1)
    groups = wo_a.shape[0]
    grouped = attended.flatten(-2).unflatten(-1, (groups, -1))
    latent = golden_grouped_output(grouped, wo_a, wo_a_scale)
    return mxfp8_linear(latent.flatten(-2), wo_b, wo_b_scale, output_dtype=output_dtype)


def make_prefill_sp_output_reduce_scatter(capacity=PREFILL_MAX_TOKENS):
    """Build the prefill collective for the specified communication capacity."""
    @pl.jit.inline(auto_scope=False)
    def prefill_sp_output_reduce_scatter(
        partial: pl.Tensor[[T_DYN, D], pl.FP32],
        output_window: pld.DistributedTensor[[capacity, D], pl.FP32],
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
        return output

    return prefill_sp_output_reduce_scatter


prefill_sp_output_reduce_scatter = make_prefill_sp_output_reduce_scatter()


def make_prefill_tp_output_all_reduce(capacity=PREFILL_MAX_TOKENS):
    """Build the prefill collective for the specified communication capacity."""
    @pl.jit.inline(auto_scope=False)
    def prefill_tp_output_all_reduce(
        output_partial: pl.Tensor[[T_DYN, D], pl.FP32],
        output_window: pld.DistributedTensor[[capacity, D], pl.FP32],
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
        return output

    return prefill_tp_output_all_reduce


prefill_tp_output_all_reduce = make_prefill_tp_output_all_reduce()


def make_decode_tp_output_reduce(scatter=False):
    """Build the decode attention-output collective.

    ``scatter=False`` keeps the historical all-reduce result (every rank holds
    the full token range). ``scatter=True`` is the sequence-parallel boundary:
    every rank still contributes its own partial, but only the contiguous token
    rows this rank owns are written, which is the ReduceScatter(SUM) result and
    never a scaled copy of an all-reduce.

    Ownership follows the rank's *physical slab* (``ceil(tokens / tp)``) rather
    than the active row count, so ``T < TP`` and a partially filled batch keep
    the same row mapping the Attention-input AllGather publishes.  Rows outside
    the active range are written as zeros: the slab is always fully materialized.
    """

    @pl.jit.inline(auto_scope=False)
    def decode_tp_output_reduce(
        output_partial: pl.Tensor[[T_DYN, D], pl.FP32],
        output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
        output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        output: pl.Tensor[[LOCAL_T_DYN, D], pl.BF16],
        group_base: pl.Scalar[pl.INT32],
        tp_rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        first = pl.cast(0, pl.INT32)
        count = pl.cast(num_tokens, pl.INT32)
        if scatter:
            # The output tensor is the fixed physical slab.  Ownership must use
            # that capacity, not active ``num_tokens``; otherwise T<TP and
            # inactive suffixes shift rows between ranks.
            width = pl.tensor.dim(output, 0)
            first = pl.cast(pl.min(tp_rank * width, num_tokens), pl.INT32)
            count = pl.cast(pl.max(0, pl.min(width, num_tokens - first)), pl.INT32)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_tp_reuse", allow_early_resolve=False) as reuse_tid:
            for peer in pl.range(TP_SIZE):
                pld.system.wait(output_arrived, offsets=[peer, 0], expected=(attention_epoch - 1) * 2,
                                cmp=pld.WaitCmp.Ge)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_tp_publish", deps=[reuse_tid]) as publish_tid:
            pld.tensor.put(
                dst=output_window, peer=group_base + tp_rank, src=output_partial,
                dst_offsets=[0, 0], src_offsets=[0, 0], shape=[num_tokens, D],
                chunk_rows=1, chunk_cols=512, pipeline=True,
            )
            for peer in pl.range(TP_SIZE):
                pld.system.notify(output_arrived, peer=group_base + peer, offsets=[tp_rank, 0],
                                  value=1, op=pld.NotifyOp.AtomicAdd)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_tp_reduce", deps=[publish_tid],
                   allow_early_resolve=False) as reduce_tid:
            for peer in pl.range(TP_SIZE):
                pld.system.wait(output_arrived, offsets=[peer, 0], expected=attention_epoch * 2 - 1,
                                cmp=pld.WaitCmp.Ge)
            for t in pl.range(first, first + count):
                for col in pl.range(0, D, 512):
                    acc = pl.tile.full([1, 512], dtype=pl.FP32, value=0.0)
                    for peer in pl.range(TP_SIZE):
                        value = pld.tile.remote_load(output_window, peer=group_base + peer,
                                                     offsets=[t, col], shape=[1, 512])
                        acc = pl.add(acc, value)
                    output = pl.store(pl.cast(acc, pl.BF16, mode="rint"), [t - first, col], output)
            if scatter:
                # The physical slab is fully materialized.  Rows outside this
                # rank's active range (an empty owner, or the padded tail of the
                # last owner) are written as zeros, so no stale window or sentinel
                # value is ever handed to the next stage.
                for t in pl.range(count, pl.tensor.dim(output, 0)):
                    for col in pl.range(0, D, 512):
                        output = pl.store(
                            pl.tile.full([1, 512], dtype=pl.BF16, value=0.0), [t, col], output
                        )
            for peer in pl.range(TP_SIZE):
                pld.system.notify(output_arrived, peer=group_base + peer, offsets=[tp_rank, 0],
                                  value=1, op=pld.NotifyOp.AtomicAdd)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_tp_consumed", deps=[reduce_tid],
                   allow_early_resolve=False):
            for peer in pl.range(TP_SIZE):
                pld.system.wait(output_arrived, offsets=[peer, 0], expected=attention_epoch * 2,
                                cmp=pld.WaitCmp.Ge)
        return output

    return decode_tp_output_reduce


decode_tp_output_all_reduce = make_decode_tp_output_reduce()
decode_tp_output_reduce_scatter = make_decode_tp_output_reduce(scatter=True)
