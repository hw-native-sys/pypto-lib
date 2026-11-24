# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 prefill output-projection TP weight materialization."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import FLASH as M
from prefill_cp_token_allgather import TP_SIZE


# model config
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS

# TP-sharded output-projection layout
O_PROJ_LOCAL_GROUPS = O_GROUPS // TP_SIZE
O_PROJ_LOCAL_COLS = O_PROJ_LOCAL_GROUPS * O_LORA
O_PROJ_FULL_ROWS = O_GROUPS * O_LORA

# Full-weight scratch and communication windows
O_PROJ_SCRATCH_GROUPS = O_GROUPS
O_PROJ_SCRATCH_RANK = O_LORA
O_PROJ_SCRATCH_INPUT = O_GROUP_IN
O_PROJ_SCRATCH_D = D
O_PROJ_SCRATCH_COLS = O_PROJ_FULL_ROWS
O_PROJ_WO_A_WINDOW_ROWS = O_PROJ_FULL_ROWS if TP_SIZE > 1 else 1
O_PROJ_WO_A_WINDOW_COLS = O_GROUP_IN if TP_SIZE > 1 else 1
# wo_b is group-major [O_GROUPS, D, O_LORA] (decode_o_proj.py's layout) and
# flattens to [GROUPS*D, O_LORA], the form wo_a uses ([GROUPS*O_LORA, O_GROUP_IN]):
# TP shards whole groups rather than a folded [D, O_GROUPS*O_LORA] column range.
O_PROJ_FULL_ROWS_B = O_GROUPS * D
O_PROJ_LOCAL_ROWS_B = O_PROJ_LOCAL_GROUPS * D
O_PROJ_WO_B_WINDOW_ROWS = O_PROJ_FULL_ROWS_B if TP_SIZE > 1 else 1
O_PROJ_WO_B_WINDOW_COLS = O_LORA if TP_SIZE > 1 else 1

# tiling
O_PROJ_WEIGHT_COPY_TILE = 16
O_PROJ_PUSH_SLABS = 4
O_PROJ_A_SLAB_ROWS = O_PROJ_LOCAL_COLS // O_PROJ_PUSH_SLABS
O_PROJ_B_SLAB_ROWS = O_PROJ_LOCAL_ROWS_B // O_PROJ_PUSH_SLABS
# wo_b rows are one group's O_LORA wide, so scale the row tile by the group
# count to keep each copy the byte size of a 16-row tile spanning every group.
O_PROJ_WO_B_COPY_TILE = O_PROJ_WEIGHT_COPY_TILE * O_GROUPS
O_PROJ_WO_B_PUSH_ROWS = O_PROJ_WEIGHT_COPY_TILE * O_PROJ_LOCAL_GROUPS
O_PROJ_A_PACK_K_TILE = 1024
O_PROJ_A_NZ_C0 = 16
O_PROJ_A_PACK_STRIPE_WORDS = O_PROJ_WEIGHT_COPY_TILE * O_PROJ_A_NZ_C0 // 2
O_PROJ_A_PACK_WORDS = O_PROJ_WEIGHT_COPY_TILE * O_PROJ_A_PACK_K_TILE // 2
O_PROJ_A_NZ_ROWS = O_GROUPS * O_GROUP_IN // O_PROJ_A_NZ_C0
O_PROJ_A_NZ_COLS = O_LORA * O_PROJ_A_NZ_C0


@pl.jit.inline
def _o_proj_a_pack_indices() -> pl.Tile[[1, O_PROJ_A_PACK_WORDS], pl.INT32]:
    lane = pl.tile.arange(0, [1, O_PROJ_A_NZ_C0 // 2], dtype=pl.INT32)
    rows = pl.tile.arange(0, [1, O_PROJ_WEIGHT_COPY_TILE], dtype=pl.INT32)
    row_start = pl.tile.mul(rows, O_PROJ_A_PACK_K_TILE // 2)
    zeros = pl.tile.full([O_PROJ_WEIGHT_COPY_TILE, O_PROJ_A_NZ_C0 // 2], dtype=pl.INT32, value=0)
    lanes = pl.tile.col_expand_add(zeros, lane)
    row_major = pl.tile.row_expand_add(lanes, pl.tile.reshape(row_start, [O_PROJ_WEIGHT_COPY_TILE, 1]))
    one_stripe = pl.tile.reshape(row_major, [1, O_PROJ_A_PACK_STRIPE_WORDS])
    strips = pl.tile.arange(0, [1, O_PROJ_A_PACK_K_TILE // O_PROJ_A_NZ_C0], dtype=pl.INT32)
    strip_offsets = pl.tile.mul(strips, O_PROJ_A_NZ_C0 // 2)
    stripe_zeros = pl.tile.full([O_PROJ_A_PACK_K_TILE // O_PROJ_A_NZ_C0, O_PROJ_A_PACK_STRIPE_WORDS], dtype=pl.INT32, value=0)
    repeated = pl.tile.col_expand_add(stripe_zeros, one_stripe)
    offsets = pl.tile.reshape(strip_offsets, [O_PROJ_A_PACK_K_TILE // O_PROJ_A_NZ_C0, 1])
    lookup = pl.tile.row_expand_add(repeated, offsets)
    return pl.tile.reshape(lookup, [1, O_PROJ_A_PACK_WORDS])


@pl.jit.inline
def _o_proj_a_pack_chunk(
    nd: pl.Tile[[O_PROJ_WEIGHT_COPY_TILE, O_PROJ_A_PACK_K_TILE], pl.BF16],
    indices: pl.Tile[[1, O_PROJ_A_PACK_WORDS], pl.INT32],
    tmp: pl.Tile[[1, O_PROJ_A_PACK_WORDS], pl.INT32],
) -> pl.Tile[[O_PROJ_A_PACK_K_TILE // O_PROJ_A_NZ_C0, O_PROJ_WEIGHT_COPY_TILE * O_PROJ_A_NZ_C0], pl.BF16]:
    words = pl.tile.reinterpret_view(nd, pl.INT32, shape=[1, O_PROJ_A_PACK_WORDS])
    ordered = pl.tile.gather(words, indices, tmp)
    return pl.tile.reinterpret_view(
        ordered, pl.BF16, shape=[O_PROJ_A_PACK_K_TILE // O_PROJ_A_NZ_C0, O_PROJ_WEIGHT_COPY_TILE * O_PROJ_A_NZ_C0],
    )


@pl.jit.inline(auto_scope=False)
def pack_o_proj_a_weights(
    wo_a: pl.Tensor[[O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT], pl.BF16],
    wo_a_nz: pl.Tensor[[O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT], pl.BF16, pl.NZ],
    weight_dep: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Pack full resident O-A weights for a standalone projection."""
    raw = pl.reshape(wo_a_nz, [O_PROJ_FULL_ROWS * O_GROUP_IN])
    packed = pl.reshape(raw, [O_PROJ_A_NZ_ROWS, O_PROJ_A_NZ_COLS])
    wo_a_flat = pl.reshape(wo_a, [O_PROJ_FULL_ROWS, O_GROUP_IN])
    with pl.spmd(
        O_PROJ_FULL_ROWS // O_PROJ_WEIGHT_COPY_TILE, name_hint="o_proj_wo_a_pack", deps=[weight_dep],
    ) as packed_ready_tid:
        row = pl.tile.get_block_idx() * O_PROJ_WEIGHT_COPY_TILE
        nz_row = (row // O_LORA) * (O_GROUP_IN // O_PROJ_A_NZ_C0)
        nz_col = (row % O_LORA) * O_PROJ_A_NZ_C0
        indices = _o_proj_a_pack_indices()
        tmp = pl.tile.create([1, O_PROJ_A_PACK_WORDS], dtype=pl.INT32, target_memory=pl.Mem.Vec)
        for k0 in pl.range(0, O_GROUP_IN, O_PROJ_A_PACK_K_TILE):
            nd_tile = pl.load(
                wo_a_flat, [row, k0], [O_PROJ_WEIGHT_COPY_TILE, O_PROJ_A_PACK_K_TILE],
                target_memory=pl.Mem.Vec,
            )
            stripe = nz_row + k0 // O_PROJ_A_NZ_C0
            packed_tile = _o_proj_a_pack_chunk(nd_tile, indices, tmp)
            pl.store(packed_tile, [stripe, nz_col], packed)
    return packed_ready_tid


@pl.jit.inline(auto_scope=False)
def _begin_o_proj_full_weights_gather(
    wo_a_local: pl.Tensor[[O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_local: pl.Tensor[[O_PROJ_LOCAL_GROUPS, D, O_LORA], pl.INT8],
    wo_a_window: pld.DistributedTensor[[O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], pl.BF16],
    wo_b_window: pld.DistributedTensor[[O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], pl.INT8],
    weight_ready: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    weight_consumed: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    weight_epoch: pl.Scalar[pl.INT32],
) -> tuple[pl.Scalar[pl.TASK_ID], pl.Scalar[pl.TASK_ID]]:
    """Push resident TP weight shards and wait for peer publication."""
    previous_epoch = weight_epoch - pl.const(1, pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="o_proj_weight_reuse_wait") as reuse_wait_tid:
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.defer_wait(
                    signal=weight_consumed, offsets=[source_tp, 0],
                    expected=previous_epoch, cmp=pld.WaitCmp.Ge,
                )

    wo_a_local_flat = pl.reshape(wo_a_local, [O_PROJ_LOCAL_COLS, O_GROUP_IN])
    wo_b_local_flat = pl.reshape(wo_b_local, [O_PROJ_LOCAL_ROWS_B, O_LORA])
    with pl.spmd(
        (TP_SIZE - 1) * O_PROJ_PUSH_SLABS, name_hint="o_proj_weight_push", deps=[reuse_wait_tid],
    ) as push_tid:
        worker = pl.tile.get_block_idx()
        peer_slot = worker // O_PROJ_PUSH_SLABS
        slab = worker % O_PROJ_PUSH_SLABS
        peer_tp = (tp_rank + pl.cast(peer_slot, pl.INT32) + 1) % TP_SIZE
        peer = group_base + peer_tp
        a_row = slab * O_PROJ_A_SLAB_ROWS
        b_row = slab * O_PROJ_B_SLAB_ROWS
        pld.tensor.put(
            dst=wo_a_window, peer=peer, src=wo_a_local_flat,
            dst_offsets=[tp_rank * O_PROJ_LOCAL_COLS + a_row, 0], src_offsets=[a_row, 0],
            shape=[O_PROJ_A_SLAB_ROWS, O_GROUP_IN],
            chunk_rows=O_PROJ_WEIGHT_COPY_TILE, chunk_cols=O_GROUP_IN,
        )
        pld.tensor.put(
            dst=wo_b_window, peer=peer, src=wo_b_local_flat,
            dst_offsets=[tp_rank * O_PROJ_LOCAL_ROWS_B + b_row, 0], src_offsets=[b_row, 0],
            shape=[O_PROJ_B_SLAB_ROWS, O_LORA],
            chunk_rows=O_PROJ_WO_B_PUSH_ROWS, chunk_cols=O_LORA,
        )

    # Publish one credit per peer after every slab of both weights is visible.
    with pl.spmd(TP_SIZE - 1, name_hint="o_proj_weight_publish", deps=[push_tid]) as publish_tid:
        peer_slot = pl.tile.get_block_idx()
        peer_tp = (tp_rank + pl.cast(peer_slot, pl.INT32) + 1) % TP_SIZE
        pld.system.notify(
            target=weight_ready, peer=group_base + peer_tp, offsets=[tp_rank, 0],
            value=1, op=pld.NotifyOp.AtomicAdd,
        )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="o_proj_weight_ready_wait") as ready_wait_tid:
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.defer_wait(
                    signal=weight_ready, offsets=[source_tp, 0],
                    expected=weight_epoch, cmp=pld.WaitCmp.Ge,
                )

    return publish_tid, ready_wait_tid


@pl.jit.inline(auto_scope=False)
def _finish_o_proj_full_weights_gather(
    wo_a_local: pl.Tensor[[O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_local: pl.Tensor[[O_PROJ_LOCAL_GROUPS, D, O_LORA], pl.INT8],
    wo_a_full: pl.Tensor[[O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT], pl.BF16],
    wo_a_nz: pl.Tensor[[O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT], pl.BF16, pl.NZ],
    wo_b_full: pl.Tensor[[O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_D, O_LORA], pl.INT8],
    wo_a_window: pld.DistributedTensor[[O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], pl.BF16],
    wo_b_window: pld.DistributedTensor[[O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], pl.INT8],
    weight_consumed: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    order_fence: pl.Tensor[[1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    publish_tid: pl.Scalar[pl.TASK_ID],
    ready_wait_tid: pl.Scalar[pl.TASK_ID],
    norm_ready_tid: pl.Scalar[pl.TASK_ID],
    WRITE_WO_A_ND: pl.constexpr,
) -> pl.Scalar[pl.TASK_ID]:
    """Read the TP shards into NZ scratch and optionally publish the ND copy."""
    wo_a_local_flat = pl.reshape(wo_a_local, [O_PROJ_LOCAL_COLS, O_GROUP_IN])
    wo_b_local_flat = pl.reshape(wo_b_local, [O_PROJ_LOCAL_ROWS_B, O_LORA])
    wo_a_full_flat = pl.reshape(wo_a_full, [O_PROJ_FULL_ROWS, O_GROUP_IN])
    wo_a_raw = pl.reshape(wo_a_nz, [O_PROJ_FULL_ROWS * O_GROUP_IN])
    wo_a_packed = pl.reshape(wo_a_raw, [O_PROJ_A_NZ_ROWS, O_PROJ_A_NZ_COLS])
    with pl.spmd(
        O_PROJ_FULL_ROWS // O_PROJ_WEIGHT_COPY_TILE, name_hint="o_proj_wo_a_readback",
        deps=[publish_tid, ready_wait_tid, norm_ready_tid],
    ) as wo_a_readback_tid:
        order = pl.read(order_fence, [0])
        if order >= 0:
            row = pl.tile.get_block_idx() * O_PROJ_WEIGHT_COPY_TILE
            nz_row = (row // O_LORA) * (O_GROUP_IN // O_PROJ_A_NZ_C0)
            nz_col = (row % O_LORA) * O_PROJ_A_NZ_C0
            indices = _o_proj_a_pack_indices()
            tmp = pl.tile.create([1, O_PROJ_A_PACK_WORDS], dtype=pl.INT32, target_memory=pl.Mem.Vec)
            for k0 in pl.range(0, O_GROUP_IN, O_PROJ_A_PACK_K_TILE):
                if row // O_PROJ_LOCAL_COLS == tp_rank:
                    local_row = row - tp_rank * O_PROJ_LOCAL_COLS
                    nd_tile = pl.load(
                        wo_a_local_flat, [local_row, k0], [O_PROJ_WEIGHT_COPY_TILE, O_PROJ_A_PACK_K_TILE],
                        target_memory=pl.Mem.Vec,
                    )
                else:
                    nd_tile = pl.load(
                        wo_a_window, [row, k0], [O_PROJ_WEIGHT_COPY_TILE, O_PROJ_A_PACK_K_TILE],
                        target_memory=pl.Mem.Vec,
                    )
                if WRITE_WO_A_ND:
                    pl.store(nd_tile, [row, k0], wo_a_full_flat)
                stripe = nz_row + k0 // O_PROJ_A_NZ_C0
                packed_tile = _o_proj_a_pack_chunk(nd_tile, indices, tmp)
                pl.store(packed_tile, [stripe, nz_col], wo_a_packed)

    wo_b_full_flat = pl.reshape(wo_b_full, [O_PROJ_FULL_ROWS_B, O_LORA])
    with pl.spmd(
        O_PROJ_FULL_ROWS_B // O_PROJ_WO_B_COPY_TILE, name_hint="o_proj_wo_b_readback",
        deps=[publish_tid, ready_wait_tid, norm_ready_tid],
    ) as wo_b_readback_tid:
        order = pl.read(order_fence, [0])
        if order >= 0:
            row = pl.tile.get_block_idx() * O_PROJ_WO_B_COPY_TILE
            if row // O_PROJ_LOCAL_ROWS_B == tp_rank:
                tile = pl.load(
                    wo_b_local_flat, [row - tp_rank * O_PROJ_LOCAL_ROWS_B, 0], [O_PROJ_WO_B_COPY_TILE, O_LORA],
                    target_memory=pl.MemorySpace.Vec,
                )
                pl.store(tile, [row, 0], wo_b_full_flat)
            else:
                tile = pl.load(
                    wo_b_window, [row, 0], [O_PROJ_WO_B_COPY_TILE, O_LORA],
                    target_memory=pl.MemorySpace.Vec,
                )
                pl.store(tile, [row, 0], wo_b_full_flat)

    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="o_proj_weight_consumed",
        deps=[wo_a_readback_tid, wo_b_readback_tid],
    ) as weights_ready_tid:
        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=weight_consumed, peer=group_base + peer_tp,
                    offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
                )

    return weights_ready_tid


if TP_SIZE == 1:

    @pl.jit.inline(auto_scope=False)
    def gather_o_proj_full_weights(
        wo_a_local: pl.Tensor[[O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
        wo_b_local: pl.Tensor[[O_PROJ_LOCAL_GROUPS, D, O_LORA], pl.INT8],
        wo_a_full: pl.Tensor[[O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT], pl.BF16],
        wo_a_nz: pl.Tensor[[O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT], pl.BF16, pl.NZ],
        wo_b_full: pl.Tensor[[O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_D, O_LORA], pl.INT8],
        wo_a_window: pld.DistributedTensor[[O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], pl.BF16],
        wo_b_window: pld.DistributedTensor[[O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], pl.INT8],
        weight_ready: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        weight_consumed: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        order_fence: pl.Tensor[[1], pl.INT32],
        group_base: pl.Scalar[pl.INT32],
        tp_rank: pl.Scalar[pl.INT32],
        weight_epoch: pl.Scalar[pl.INT32],
    ) -> pl.Scalar[pl.TASK_ID]:
        """Copy TP1's resident full weights into ND and NZ scratch."""
        wo_a_local_flat = pl.reshape(wo_a_local, [O_PROJ_FULL_ROWS, O_GROUP_IN])
        wo_a_full_flat = pl.reshape(wo_a_full, [O_PROJ_FULL_ROWS, O_GROUP_IN])
        wo_a_raw = pl.reshape(wo_a_nz, [O_PROJ_FULL_ROWS * O_GROUP_IN])
        wo_a_packed = pl.reshape(wo_a_raw, [O_PROJ_A_NZ_ROWS, O_PROJ_A_NZ_COLS])
        with pl.spmd(O_PROJ_FULL_ROWS // O_PROJ_WEIGHT_COPY_TILE, name_hint="o_proj_tp1_wo_a_copy") as wo_a_copy_tid:
            order = pl.read(order_fence, [0])
            if order >= 0:
                row = pl.tile.get_block_idx() * O_PROJ_WEIGHT_COPY_TILE
                nz_row = (row // O_LORA) * (O_GROUP_IN // O_PROJ_A_NZ_C0)
                nz_col = (row % O_LORA) * O_PROJ_A_NZ_C0
                indices = _o_proj_a_pack_indices()
                tmp = pl.tile.create([1, O_PROJ_A_PACK_WORDS], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                for k0 in pl.range(0, O_GROUP_IN, O_PROJ_A_PACK_K_TILE):
                    nd_tile = pl.load(
                        wo_a_local_flat, [row, k0], [O_PROJ_WEIGHT_COPY_TILE, O_PROJ_A_PACK_K_TILE],
                        target_memory=pl.Mem.Vec,
                    )
                    pl.store(nd_tile, [row, k0], wo_a_full_flat)
                    stripe = nz_row + k0 // O_PROJ_A_NZ_C0
                    packed_tile = _o_proj_a_pack_chunk(nd_tile, indices, tmp)
                    pl.store(packed_tile, [stripe, nz_col], wo_a_packed)

        wo_b_local_flat = pl.reshape(wo_b_local, [O_PROJ_FULL_ROWS_B, O_LORA])
        wo_b_full_flat = pl.reshape(wo_b_full, [O_PROJ_FULL_ROWS_B, O_LORA])
        with pl.spmd(O_PROJ_FULL_ROWS_B // O_PROJ_WO_B_COPY_TILE, name_hint="o_proj_tp1_wo_b_copy") as wo_b_copy_tid:
            order = pl.read(order_fence, [0])
            if order >= 0:
                row = pl.tile.get_block_idx() * O_PROJ_WO_B_COPY_TILE
                tile = pl.load(
                    wo_b_local_flat, [row, 0], [O_PROJ_WO_B_COPY_TILE, O_LORA],
                    target_memory=pl.MemorySpace.Vec,
                )
                pl.store(tile, [row, 0], wo_b_full_flat)
        weights_ready_tid = pl.system.task_dummy(deps=[wo_a_copy_tid, wo_b_copy_tid])
        return weights_ready_tid


@pl.jit.inline(auto_scope=False)
def retire_o_proj_weight_signals(
    order_fence: pl.Tensor[[1], pl.INT32],
    weight_ready: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    weight_consumed: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    completed_epochs: pl.Scalar[pl.INT32],
):
    """Retire completed output-projection collective credits."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="o_proj_weight_signal_retire") as retire_tid:
        completed = pl.read(order_fence, [0])
        if completed >= completed_epochs:
            reset_value = pl.cast(0 - completed_epochs, pl.INT32)
            self_rank = group_base + tp_rank
            for source_tp in pl.range(TP_SIZE):
                if source_tp != tp_rank:
                    pld.system.notify(
                        target=weight_ready, peer=self_rank,
                        offsets=[source_tp, 0], value=reset_value, op=pld.NotifyOp.AtomicAdd,
                    )
                    pld.system.notify(
                        target=weight_consumed, peer=self_rank,
                        offsets=[source_tp, 0], value=reset_value, op=pld.NotifyOp.AtomicAdd,
                    )
    return retire_tid
