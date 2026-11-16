# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Migrate six expert tensors through receive windows and two global completion barriers."""

import pypto.language as pl
import pypto.language.distributed as pld

EPLB_ABI_VERSION = 1


def make_eplb_transfer(*, ranks: int, local_slots: int, hidden_size: int, intermediate_size: int):
    """Specialize a raw-storage maintenance program to a resident target weight bank."""
    if min(ranks, local_slots, hidden_size, intermediate_size) <= 0:
        raise ValueError("EPLB transfer dimensions must be positive")
    R = ranks
    C = local_slots
    W = hidden_size * intermediate_size
    I = intermediate_size
    D = hidden_size
    # tiling
    COPY_COL_TILE = 8192

    @pl.jit
    def eplb_transfer_layer(
        w1: pl.InOut[pl.Tensor[[C, W], pl.INT8]],
        w1_scale: pl.InOut[pl.Tensor[[C, I], pl.FP32]],
        w3: pl.InOut[pl.Tensor[[C, W], pl.INT8]],
        w3_scale: pl.InOut[pl.Tensor[[C, I], pl.FP32]],
        w2: pl.InOut[pl.Tensor[[C, W], pl.INT8]],
        w2_scale: pl.InOut[pl.Tensor[[C, D], pl.FP32]],
        sources: pl.Tensor[[R, C, 2], pl.INT32],
        changed: pl.Tensor[[R, C], pl.INT32],
        receive_w1: pld.DistributedTensor[[C, W], pl.INT8],
        receive_w1_scale: pld.DistributedTensor[[C, I], pl.FP32],
        receive_w3: pld.DistributedTensor[[C, W], pl.INT8],
        receive_w3_scale: pld.DistributedTensor[[C, I], pl.FP32],
        receive_w2: pld.DistributedTensor[[C, W], pl.INT8],
        receive_w2_scale: pld.DistributedTensor[[C, D], pl.FP32],
        sources_done: pld.DistributedTensor[[R, 1], pl.INT32],
        installed: pld.DistributedTensor[[R, 1], pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
        epoch: pl.Scalar[pl.INT32],
    ):
        # All resident reads finish before any resident write, including local moves.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="eplb_read_sources") as push_tid:
            for peer in pl.range(R):
                for slot in pl.range(C):
                    move = pl.read(changed, [peer, slot])
                    source_rank = pl.read(sources, [peer, slot, 0])
                    source_slot = pl.read(sources, [peer, slot, 1])
                    if move != 0 and source_rank == my_rank:
                        pld.tensor.put(
                            receive_w1, peer, w1,
                            [slot, 0], [source_slot, 0], [1, W], chunk_cols=COPY_COL_TILE,
                        )
                        pld.tensor.put(
                            receive_w1_scale, peer, w1_scale,
                            [slot, 0], [source_slot, 0], [1, I], chunk_cols=COPY_COL_TILE,
                        )
                        pld.tensor.put(
                            receive_w3, peer, w3,
                            [slot, 0], [source_slot, 0], [1, W], chunk_cols=COPY_COL_TILE,
                        )
                        pld.tensor.put(
                            receive_w3_scale, peer, w3_scale,
                            [slot, 0], [source_slot, 0], [1, I], chunk_cols=COPY_COL_TILE,
                        )
                        pld.tensor.put(
                            receive_w2, peer, w2,
                            [slot, 0], [source_slot, 0], [1, W], chunk_cols=COPY_COL_TILE,
                        )
                        pld.tensor.put(
                            receive_w2_scale, peer, w2_scale,
                            [slot, 0], [source_slot, 0], [1, D], chunk_cols=COPY_COL_TILE,
                        )
            for peer in pl.range(R):
                if peer != my_rank:
                    pld.system.notify(sources_done, peer, [my_rank, 0], 1, op=pld.NotifyOp.AtomicAdd)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="eplb_wait_sources") as read_wait_tid:
            for peer in pl.range(R):
                if peer != my_rank:
                    pld.system.defer_wait(sources_done, [peer, 0], epoch, cmp=pld.WaitCmp.Ge)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="eplb_install", deps=[push_tid, read_wait_tid]) as install_tid:
            for slot in pl.range(C):
                if pl.read(changed, [my_rank, slot]) != 0:
                    pld.tensor.get(w1, my_rank, receive_w1, [slot, 0], [slot, 0], [1, W], chunk_cols=COPY_COL_TILE)
                    pld.tensor.get(
                        w1_scale, my_rank, receive_w1_scale,
                        [slot, 0], [slot, 0], [1, I], chunk_cols=COPY_COL_TILE,
                    )
                    pld.tensor.get(w3, my_rank, receive_w3, [slot, 0], [slot, 0], [1, W], chunk_cols=COPY_COL_TILE)
                    pld.tensor.get(
                        w3_scale, my_rank, receive_w3_scale,
                        [slot, 0], [slot, 0], [1, I], chunk_cols=COPY_COL_TILE,
                    )
                    pld.tensor.get(w2, my_rank, receive_w2, [slot, 0], [slot, 0], [1, W], chunk_cols=COPY_COL_TILE)
                    pld.tensor.get(
                        w2_scale, my_rank, receive_w2_scale,
                        [slot, 0], [slot, 0], [1, D], chunk_cols=COPY_COL_TILE,
                    )
            for peer in pl.range(R):
                if peer != my_rank:
                    pld.system.notify(installed, peer, [my_rank, 0], 1, op=pld.NotifyOp.AtomicAdd)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="eplb_wait_install", deps=[install_tid]):
            for peer in pl.range(R):
                if peer != my_rank:
                    pld.system.defer_wait(installed, [peer, 0], epoch, cmp=pld.WaitCmp.Ge)
        return w1, w1_scale, w3, w3_scale, w2, w2_scale

    @pl.jit.host
    def l3_eplb_transfer_layer(
        w1: pl.InOut[pl.Tensor[[R, C, W], pl.INT8]],
        w1_scale: pl.InOut[pl.Tensor[[R, C, I], pl.FP32]],
        w3: pl.InOut[pl.Tensor[[R, C, W], pl.INT8]],
        w3_scale: pl.InOut[pl.Tensor[[R, C, I], pl.FP32]],
        w2: pl.InOut[pl.Tensor[[R, C, W], pl.INT8]],
        w2_scale: pl.InOut[pl.Tensor[[R, C, D], pl.FP32]],
        sources: pl.Tensor[[R, R, C, 2], pl.INT32],
        changed: pl.Tensor[[R, R, C], pl.INT32],
        epoch: pl.Scalar[pl.INT32],
    ):
        """EPLB HOST ABI v1: six-tensor migration and completion epochs."""
        receive_w1_buf = pld.alloc_window_buffer([C, W], dtype=pl.INT8)
        receive_w1_scale_buf = pld.alloc_window_buffer([C, I], dtype=pl.FP32)
        receive_w3_buf = pld.alloc_window_buffer([C, W], dtype=pl.INT8)
        receive_w3_scale_buf = pld.alloc_window_buffer([C, I], dtype=pl.FP32)
        receive_w2_buf = pld.alloc_window_buffer([C, W], dtype=pl.INT8)
        receive_w2_scale_buf = pld.alloc_window_buffer([C, D], dtype=pl.FP32)
        sources_done_buf = pld.alloc_window_buffer([R, 1], dtype=pl.INT32)
        installed_buf = pld.alloc_window_buffer([R, 1], dtype=pl.INT32)
        for rank in pl.range(R):
            receive_w1 = pld.window(receive_w1_buf, [C, W], dtype=pl.INT8)
            receive_w1_scale = pld.window(receive_w1_scale_buf, [C, I], dtype=pl.FP32)
            receive_w3 = pld.window(receive_w3_buf, [C, W], dtype=pl.INT8)
            receive_w3_scale = pld.window(receive_w3_scale_buf, [C, I], dtype=pl.FP32)
            receive_w2 = pld.window(receive_w2_buf, [C, W], dtype=pl.INT8)
            receive_w2_scale = pld.window(receive_w2_scale_buf, [C, D], dtype=pl.FP32)
            sources_done = pld.window(sources_done_buf, [R, 1], dtype=pl.INT32)
            installed = pld.window(installed_buf, [R, 1], dtype=pl.INT32)
            eplb_transfer_layer(
                w1[rank], w1_scale[rank], w3[rank], w3_scale[rank], w2[rank], w2_scale[rank],
                sources[rank], changed[rank],
                receive_w1, receive_w1_scale, receive_w3, receive_w3_scale, receive_w2, receive_w2_scale,
                sources_done, installed,
                rank, epoch, device=rank,
            )
        return w1, w1_scale, w3, w3_scale, w2, w2_scale

    return l3_eplb_transfer_layer
