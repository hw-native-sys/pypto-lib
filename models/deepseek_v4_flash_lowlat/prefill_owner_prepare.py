# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exchange exact token-owner preparation through two bounded real transport lanes."""

import pypto.language as pl
import pypto.language.distributed as pld

from .prefill_moe_geometry import D, HC_MULT, N_RANKS, OWNER_WINDOW_ROWS, PREP_OWNER_ROWS, T, T_PAD


OWNER_COMB_D_TILE = HC_MULT * HC_MULT
OWNER_META_T_TILE = min(64, PREP_OWNER_ROWS)
OWNER_SCALE_D_TILE = min(256, PREP_OWNER_ROWS)
OWNER_POST_D_TILE = min(256, PREP_OWNER_ROWS * HC_MULT)


@pl.jit.inline(auto_scope=False)
def exchange_preparation(
    source_norm: pl.Tensor[[PREP_OWNER_ROWS, D], pl.INT8],
    source_scale: pl.Tensor[[PREP_OWNER_ROWS, 1], pl.FP32],
    source_post: pl.Tensor[[PREP_OWNER_ROWS, HC_MULT], pl.FP32],
    source_comb: pl.Tensor[[PREP_OWNER_ROWS, HC_MULT * HC_MULT], pl.FP32],
    source_indices: pl.Tensor[[PREP_OWNER_ROWS, 16], pl.INT32],
    source_weights: pl.Tensor[[PREP_OWNER_ROWS, 16], pl.FP32],
    norm: pl.Tensor[[T, D], pl.INT8],
    scale: pl.Tensor[[T, 1], pl.FP32],
    post: pl.Tensor[[T, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T, HC_MULT * HC_MULT], pl.FP32],
    indices: pl.Tensor[[T, 16], pl.INT32],
    weights: pl.Tensor[[T, 16], pl.FP32],
    norm_window: pld.DistributedTensor[[OWNER_WINDOW_ROWS, D], pl.INT8],
    scale_window: pld.DistributedTensor[[1, OWNER_WINDOW_ROWS], pl.FP32],
    post_window: pld.DistributedTensor[[1, OWNER_WINDOW_ROWS * HC_MULT], pl.FP32],
    comb_window: pld.DistributedTensor[[OWNER_WINDOW_ROWS, HC_MULT * HC_MULT], pl.FP32],
    indices_window: pld.DistributedTensor[[OWNER_WINDOW_ROWS, 16], pl.INT32],
    weights_window: pld.DistributedTensor[[OWNER_WINDOW_ROWS, 16], pl.FP32],
    signal: pld.DistributedTensor[[N_RANKS, 16], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    fields_ready: pl.Scalar[pl.TASK_ID],
    previous_copy: pl.Scalar[pl.TASK_ID],
):
    lane_base = pl.cast((epoch - 1) % 2, pl.INDEX) * T_PAD
    owner_base = pl.cast(rank, pl.INDEX) * PREP_OWNER_ROWS
    source_scale_flat = pl.reshape(source_scale, [1, PREP_OWNER_ROWS])
    source_post_flat = pl.reshape(source_post, [1, PREP_OWNER_ROWS * HC_MULT])
    scale_flat = pl.reshape(scale, [1, T])
    post_flat = pl.reshape(post, [1, T * HC_MULT])
    tasks = pl.array.create(3, pl.TASK_ID)
    with pl.manual_scope():
        with pl.spmd(N_RANKS, name_hint="prefill_prepare_owner_publish", deps=[fields_ready, previous_copy]) as published:
            peer = pl.tile.get_block_idx()
            row = lane_base + owner_base
            pld.tensor.put(dst=norm_window, peer=peer, src=source_norm, dst_offsets=[row, 0], src_offsets=[0, 0], shape=[PREP_OWNER_ROWS, D], chunk_rows=16, chunk_cols=D)
            pld.tensor.put(dst=scale_window, peer=peer, src=source_scale_flat, dst_offsets=[0, row], src_offsets=[0, 0], shape=[1, PREP_OWNER_ROWS], chunk_rows=1, chunk_cols=OWNER_SCALE_D_TILE)
            pld.tensor.put(dst=post_window, peer=peer, src=source_post_flat, dst_offsets=[0, row * HC_MULT], src_offsets=[0, 0], shape=[1, PREP_OWNER_ROWS * HC_MULT], chunk_rows=1, chunk_cols=OWNER_POST_D_TILE)
            pld.tensor.put(dst=comb_window, peer=peer, src=source_comb, dst_offsets=[row, 0], src_offsets=[0, 0], shape=[PREP_OWNER_ROWS, HC_MULT * HC_MULT], chunk_rows=OWNER_META_T_TILE, chunk_cols=OWNER_COMB_D_TILE)
            pld.tensor.put(dst=indices_window, peer=peer, src=source_indices, dst_offsets=[row, 0], src_offsets=[0, 0], shape=[PREP_OWNER_ROWS, 16], chunk_rows=OWNER_META_T_TILE, chunk_cols=16)
            pld.tensor.put(dst=weights_window, peer=peer, src=source_weights, dst_offsets=[row, 0], src_offsets=[0, 0], shape=[PREP_OWNER_ROWS, 16], chunk_rows=OWNER_META_T_TILE, chunk_cols=16)
            pld.system.notify(signal, peer, [rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        tasks[0] = published
    with pl.manual_scope():
        published_scalar = tasks[0]
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_prepare_owner_defer", deps=[published_scalar]) as waited:
            for source in pl.range(N_RANKS):
                pld.system.defer_wait(signal, [source, 0], epoch, cmp=pld.WaitCmp.Ge)
        tasks[1] = waited
    with pl.manual_scope():
        waited_scalar = tasks[1]
        with pl.spmd(T // 16, name_hint="prefill_prepare_owner_collect", deps=[waited_scalar]) as copied:
            start = pl.tile.get_block_idx() * 16
            norm_tile = pl.load(norm_window, [lane_base + start, 0], [16, D])
            pl.store(norm_tile, [start, 0], norm)
            scale_tile = pl.load(scale_window, [0, lane_base + start], [1, 16])
            pl.store(scale_tile, [0, start], scale_flat)
            post_tile = pl.load(post_window, [0, (lane_base + start) * HC_MULT], [1, 16 * HC_MULT])
            pl.store(post_tile, [0, start * HC_MULT], post_flat)
            comb_tile = pl.load(comb_window, [lane_base + start, 0], [16, HC_MULT * HC_MULT])
            pl.store(comb_tile, [start, 0], comb)
            indices_tile = pl.load(indices_window, [lane_base + start, 0], [16, 16])
            pl.store(indices_tile, [start, 0], indices)
            weights_tile = pl.load(weights_window, [lane_base + start, 0], [16, 16])
            pl.store(weights_tile, [start, 0], weights)
        tasks[2] = copied
    return tasks[2]
