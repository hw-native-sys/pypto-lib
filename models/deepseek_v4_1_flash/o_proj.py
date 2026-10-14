# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 inverse RoPE, grouped output projections, and TP output reductions."""

import torch

from qkv_proj_rope import golden_rope_interleave as rope_interleave
from quant import mxfp8_linear

import pypto.language as pl
import pypto.language.distributed as pld

from qkv_proj_rope import (
    K_TILE,
    M_TILE,
    N_TILE,
    make_mx_projection,
    make_rope,
)
from config import (
    DECODE_MAX_TOKENS, PREFILL_MAX_TOKENS, FLASH as M, LOCAL_T_DYN, SP_T_DYN, T_DYN, TP_SIZE,
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


@pl.jit.inline
def _grouped_output_block(
    x: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    weight: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    output: pl.Tensor[[T_DYN, LOCAL_O_WIDTH], pl.BF16],
    block: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    t0 = block // (LOCAL_O_WIDTH // N_TILE) * M_TILE
    n0 = block % (LOCAL_O_WIDTH // N_TILE) * N_TILE
    group = n0 // O_LORA
    local_n = n0 % O_LORA
    rows = pl.min(M_TILE, num_tokens - t0)
    acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.FP32)
    for kb in pl.range(O_GROUP_IN // K_TILE):
        k0 = kb * K_TILE
        a = pl.slice(
            x,
            [M_TILE, K_TILE],
            [t0, group * O_GROUP_IN + k0],
            valid_shape=[rows, K_TILE],
        )
        w = pl.reshape(
            weight[group : group + 1, local_n : local_n + N_TILE, k0 : k0 + K_TILE],
            [N_TILE, K_TILE],
        )
        acc = pl.matmul_acc(acc, a, w, b_trans=True, init_cond=(kb == 0))
    value = pl.cast(acc, pl.BF16, mode="rint")
    output[t0 : t0 + M_TILE, n0 : n0 + N_TILE] = pl.set_validshape(value, rows, N_TILE)
    return output


@pl.jit.inline
def grouped_output(
    x: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    weight: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    output: pl.Tensor[[T_DYN, LOCAL_O_WIDTH], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Apply grouped Wo-A with FP32 accumulation and a BF16 boundary."""
    for block in pl.spmd(
        (num_tokens + M_TILE - 1) // M_TILE * (LOCAL_O_WIDTH // N_TILE),
        name_hint="attention_grouped_o_a",
    ):
        _grouped_output_block(x, weight, output, block, num_tokens)
    return output


@pl.jit.inline
def prefill_grouped_output(
    x: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    weight: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    output: pl.Tensor[[T_DYN, LOCAL_O_WIDTH], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Stage Wo-A values with Vector startup and FP32 Vector stores before narrowing."""
    tokens = pl.tensor.dim(x, 0)
    accumulated = pl.create_tensor([tokens, LOCAL_O_WIDTH], dtype=pl.FP32)
    blocks = (num_tokens + M_TILE - 1) // M_TILE * (LOCAL_O_WIDTH // N_TILE)
    with pl.spmd(blocks, name_hint="prefill_grouped_o_a_cube") as cube_tid:
        block = pl.tile.get_block_idx()
        t0 = block // (LOCAL_O_WIDTH // N_TILE) * M_TILE
        n0 = block % (LOCAL_O_WIDTH // N_TILE) * N_TILE
        group = n0 // O_LORA
        local_n = n0 % O_LORA
        rows = pl.min(M_TILE, num_tokens - t0)
        acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.FP32)
        for kb in pl.range(O_GROUP_IN // K_TILE):
            k0 = kb * K_TILE
            a = pl.slice(x, [M_TILE, K_TILE], [t0, group * O_GROUP_IN + k0], valid_shape=[rows, K_TILE])
            a = pl.cast(pl.cast(a, pl.FP32), pl.BF16, mode="rint")
            w = pl.reshape(weight[group : group + 1, local_n : local_n + N_TILE, k0 : k0 + K_TILE], [N_TILE, K_TILE])
            acc = pl.matmul_acc(acc, a, w, b_trans=True, init_cond=(kb == 0))
        accumulated[t0 : t0 + M_TILE, n0 : n0 + N_TILE] = pl.set_validshape(pl.mul(acc, 1.0), rows, N_TILE)
    with pl.spmd(blocks, name_hint="prefill_grouped_o_a_cast", deps=[cube_tid]):
        block = pl.tile.get_block_idx()
        t0 = block // (LOCAL_O_WIDTH // N_TILE) * M_TILE
        n0 = block % (LOCAL_O_WIDTH // N_TILE) * N_TILE
        rows = pl.min(M_TILE, num_tokens - t0)
        value = pl.slice(accumulated, [M_TILE, N_TILE], [t0, n0], valid_shape=[rows, N_TILE])
        output[t0 : t0 + M_TILE, n0 : n0 + N_TILE] = pl.cast(value, pl.BF16, mode="rint")
    return output


@pl.jit.inline
def grouped_output_with_deps(
    x: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    weight: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    output: pl.Tensor[[T_DYN, LOCAL_O_WIDTH], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    """Apply grouped Wo-A after an explicit producer dependency."""
    with pl.spmd(
        (num_tokens + M_TILE - 1) // M_TILE * (LOCAL_O_WIDTH // N_TILE),
        name_hint="c1a_grouped_output",
        deps=[ready],
    ) as grouped_tid:
        block = pl.tile.get_block_idx()
        _grouped_output_block(x, weight, output, block, num_tokens)
    return grouped_tid


def _make_o_proj(rotate, grouped, project):
    @pl.jit.inline(auto_scope=False)
    def o_proj(
        attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
        wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
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
        grouped(unrotated, wo_a, latent, num_tokens)
        project(latent, wo_b, wo_b_scale, output, num_tokens)
        return output

    return o_proj


def make_o_proj_with_deps(rotate, grouped, project):
    """Compose dependency-aware C1A decode output primitives."""

    @pl.jit.inline(auto_scope=False)
    def o_proj_with_deps(
        attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
        wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
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
        grouped_tid = grouped(unrotated, wo_a, latent, num_tokens, rotate_tid)
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


def _project_output(
    attended: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor | None,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    rope_dim = cos.shape[-1] * 2
    tail = rope_interleave(attended[..., -rope_dim:], cos, sin, inverse=True)
    attended = torch.cat((attended[..., :-rope_dim], tail), dim=-1)
    groups = wo_a.shape[0]
    grouped = attended.flatten(-2).unflatten(-1, (groups, -1))
    latent = torch.einsum("tgd,grd->tgr", grouped.float(), wo_a.float()).to(attended.dtype)
    return mxfp8_linear(latent.flatten(-2), wo_b, wo_b_scale, output_dtype=output_dtype)


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
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=(attention_epoch - 1) * 2,
                            cmp=pld.WaitCmp.Ge)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_reduce_publish", deps=[reuse_tid]) as publish_tid:
        if num_tokens > 0:
            pld.tensor.put(
                dst=output_window, peer=group_base + tp_rank, src=partial,
                dst_offsets=[0, 0], src_offsets=[0, 0], shape=[num_tokens, D],
                chunk_rows=1, chunk_cols=D_TILE, pipeline=True,
            )
        for peer in pl.range(TP_SIZE):
            pld.system.notify(output_arrived, peer=group_base + peer, offsets=[tp_rank, 0],
                              value=1, op=pld.NotifyOp.AtomicAdd)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_reduce_wait", deps=[publish_tid],
               allow_early_resolve=False) as wait_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=attention_epoch * 2 - 1,
                            cmp=pld.WaitCmp.Ge)
    with pl.spmd(64, name_hint="sp_reduce_scatter", deps=[wait_tid]) as reduce_tid:
        worker = pl.tile.get_block_idx()
        for tile in pl.range(worker, local_tokens * (D // D_TILE), 64):
            local_row = tile // (D // D_TILE)
            col = tile % (D // D_TILE) * D_TILE
            row = tp_rank * local_tokens + local_row
            acc = pl.tile.full([1, D_TILE], dtype=pl.FP32, value=0.0)
            if row < num_tokens:
                for peer in pl.range(TP_SIZE):
                    value = pld.tile.remote_load(output_window, peer=group_base + peer,
                                                 offsets=[row, col], shape=[1, D_TILE])
                    acc = pl.add(acc, value)
            result = pl.cast(acc, pl.BF16, mode="rint")
            output = pl.store(result, [local_row, col], output)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_reduce_release", deps=[reduce_tid]) as release_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(output_arrived, peer=group_base + peer, offsets=[tp_rank, 0],
                              value=1, op=pld.NotifyOp.AtomicAdd)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_reduce_consumed", deps=[release_tid],
               allow_early_resolve=False):
        for peer in pl.range(TP_SIZE):
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=attention_epoch * 2,
                            cmp=pld.WaitCmp.Ge)
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
                output_arrived, peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
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
                value = pld.tile.remote_load(output_window, peer=group_base + peer, offsets=[row, col], shape=[1, 512])
                acc = pl.add(acc, value)
            output = pl.store(pl.cast(acc, pl.BF16, mode="rint"), [row, col], output)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_release", deps=[reduce_tid]) as release_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                output_arrived, peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_consumed", deps=[release_tid],
               allow_early_resolve=False):
        for peer in pl.range(TP_SIZE):
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=attention_epoch * 2, cmp=pld.WaitCmp.Ge)
    return output


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
            pld.tensor.put(dst=output_window, peer=group_base + tp_rank, src=output_partial,
                           dst_offsets=[0, 0], src_offsets=[0, 0], shape=[num_tokens, D],
                           chunk_rows=1, chunk_cols=512, pipeline=True)
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
