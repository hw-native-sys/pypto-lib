# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""C4 merged-index and sparse-plan AIV bodies."""

import pypto.language as pl
import pypto.language.distributed as pld

if __package__:
    from .decode_indexer import IDX_TOPK, QUERY_SLOTS, TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH
    from .decode_sparse_attn import (
        CSA_ATTN_K_TILE,
        CSA_CMP_GE_BIAS,
        CSA_CMP_TOPK,
        CSA_COMPRESS_RATIO,
        CSA_PADDED_TOPK,
        CSA_SPARSE_BLOCKS,
        CSA_TOPK,
        NEG_INF,
        WIN,
        WIN_BLOCKS,
    )
else:
    from decode_indexer import IDX_TOPK, QUERY_SLOTS, TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH
    from decode_sparse_attn import (
        CSA_ATTN_K_TILE,
        CSA_CMP_GE_BIAS,
        CSA_CMP_TOPK,
        CSA_COMPRESS_RATIO,
        CSA_PADDED_TOPK,
        CSA_SPARSE_BLOCKS,
        CSA_TOPK,
        NEG_INF,
        WIN,
        WIN_BLOCKS,
    )


PADDED_PLAN_COLS = 32




@pl.jit.inline(auto_scope=False)
def ordered_qk_item(
    descriptors: pl.Tensor[[QUERY_SLOTS, PADDED_PLAN_COLS], pl.INT32],
    ordinal: pl.Scalar[pl.INDEX],
) -> pl.Scalar[pl.INDEX]:
    """Read the original stable valid-first order from per-query descriptors."""
    total_valid = pl.cast(0, pl.INDEX)
    for token in pl.unroll(QUERY_SLOTS):
        total_valid = total_valid + pl.cast(pl.read(descriptors, [token, 1]), pl.INDEX)
    item = ordinal
    if total_valid < QUERY_SLOTS * CSA_SPARSE_BLOCKS:
        desired_valid = ordinal < total_valid
        valid_flag = pl.cast(1,pl.INT32)
        rest = ordinal
        if not desired_valid:
            rest = ordinal - total_valid
            valid_flag = pl.cast(0,pl.INT32)
        count_before = pl.cast(0, pl.INDEX)
        selected = pl.cast(-1, pl.INDEX)
        for token in pl.unroll(QUERY_SLOTS):
            tile_count = pl.cast(pl.read(descriptors, [token, 1]), pl.INDEX)
            if not desired_valid:
                tile_count = CSA_SPARSE_BLOCKS - tile_count
            if count_before <= rest and rest < count_before+tile_count:
                selected = pl.cast(token, pl.INDEX)
            count_before = count_before + tile_count
        if selected >= 0:
            selected_count = pl.cast(0, pl.INDEX)
            for token in pl.unroll(QUERY_SLOTS):
                if token < selected:
                    tile_count = pl.cast(pl.read(descriptors, [token, 1]), pl.INDEX)
                    if not desired_valid:
                        tile_count = CSA_SPARSE_BLOCKS - tile_count
                    selected_count = selected_count + tile_count
            mask = pl.read(descriptors, [selected, 0])
            for block in pl.unroll(CSA_SPARSE_BLOCKS):
                active = pl.cast((mask // (1 << block)) % 2,pl.INT32)
                if active == valid_flag:
                    if selected_count == rest:
                        item = selected * CSA_SPARSE_BLOCKS + block
                    selected_count = selected_count+1
    return item


@pl.jit.inline(auto_scope=False)
def merge_leaf_and_plan(
    leaf_pairs: pl.Tensor[[QUERY_SLOTS * TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH], pl.FP32],
    positions: pl.Tensor[[1, QUERY_SLOTS], pl.INT32],
    window: pl.Tensor[[QUERY_SLOTS, WIN], pl.INT32],
    cmp_indices: pl.Tensor[[QUERY_SLOTS, CSA_CMP_TOPK], pl.INT32],
    sparse_bias: pl.Tensor[[QUERY_SLOTS, CSA_PADDED_TOPK], pl.FP32],
    descriptors: pl.Tensor[[QUERY_SLOTS, PADDED_PLAN_COLS], pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Exact four-way merge followed by the unchanged per-row plan maths."""
    with pl.spmd(QUERY_SLOTS,name_hint="c4_merge_leaf_and_plan",allow_early_resolve=True) as ready:
        token=pl.tile.get_block_idx()
        position = pl.read(positions, [0, token])
        base = token*TOPK_ARENA_ROWS
        top = pl.mrgsort(
            leaf_pairs[base:base+1, :],
            leaf_pairs[base+1:base+2, :],
            leaf_pairs[base+2:base+3, :],
            leaf_pairs[base+3:base+4,:],
        )
        selected = pl.gather(top[:, :TOPK_PAIR_WIDTH],
                             mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        raw = pl.cast(selected, pl.FP32)
        upper = pl.cast(pl.cast((position + 1) // CSA_COMPRESS_RATIO, pl.INT32), pl.FP32)
        upper_row = pl.mul(pl.full([1, IDX_TOPK], dtype=pl.FP32, value=1.0), upper)
        ge = pl.minimum(pl.maximum(pl.add(raw, CSA_CMP_GE_BIAS), 0.0), 1.0)
        lt = pl.minimum(pl.maximum(pl.sub(upper_row, raw), 0.0), 1.0)
        valid = pl.mul(ge, lt)
        mapped = pl.sub(pl.mul(valid, pl.add(raw, 1.0)), 1.0)
        cmp_indices[token:token + 1, :IDX_TOPK] = pl.cast(mapped, pl.INT32)

        win = pl.cast(window[token:token+1,:], pl.FP32)
        win_valid = pl.minimum(pl.maximum(pl.add(win, 1.0), 0.0), 1.0)
        sparse_bias[token:token + 1, :WIN] = pl.mul(pl.sub(win_valid, 1.0), -NEG_INF)
        sparse_bias[token:token + 1, WIN:CSA_TOPK] = pl.mul(
            pl.minimum(mapped, 0.0), -NEG_INF,
        )
        if CSA_PADDED_TOPK > CSA_TOPK:
            sparse_bias[token:token + 1, CSA_TOPK:CSA_PADDED_TOPK] = pl.full(
                [1, CSA_PADDED_TOPK-CSA_TOPK], pl.FP32, value=NEG_INF,
            )

        block_valid = pl.row_max(pl.reshape(valid, [CSA_CMP_TOPK // CSA_ATTN_K_TILE, CSA_ATTN_K_TILE]))
        mask = pl.cast((1 << WIN_BLOCKS) - 1, pl.INT32)
        count = pl.cast(WIN_BLOCKS, pl.INT32)
        for block in pl.unroll(WIN_BLOCKS, CSA_SPARSE_BLOCKS):
            bit = pl.cast(pl.read(block_valid, [block-WIN_BLOCKS, 0]), pl.INT32)
            count = count + bit
            weight = pl.cast(1 << block, pl.INT32)
            mask = mask + bit * weight
        row = pl.tile.full([1, PADDED_PLAN_COLS], dtype=pl.INT32, value=0)
        pl.tile.write(row, [0, 0], mask)
        pl.tile.write(row, [0, 1], count)
        pl.store(row, [token, 0], descriptors)
    return ready

@pl.jit.inline(auto_scope=False)
def merge_owned_leaf(
    leaf_pairs: pl.Tensor[[TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH], pl.FP32],
    topk_ids: pl.Out[pl.Tensor[[1, IDX_TOPK], pl.INT32]],
) -> pl.Scalar[pl.TASK_ID]:
    """Extract a single query's exact selected order using the same four runs."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c4_exact_selected_owned") as selected:
        packed = pl.mrgsort(leaf_pairs[0:1, :], leaf_pairs[1:2, :],
                            leaf_pairs[2:3, :], leaf_pairs[3:4, :])
        topk_ids[0:1, :] = pl.gather(packed[:, :TOPK_PAIR_WIDTH],
                                    mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
    return selected


@pl.jit.inline(auto_scope=False)
def merge_and_publish_owned_leaf(
    leaf_pairs: pl.Tensor[[TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH], pl.FP32],
    topk_ids: pl.Out[pl.Tensor[[1, IDX_TOPK], pl.INT32]],
    ids_window: pld.DistributedTensor[[2 * QUERY_SLOTS, IDX_TOPK], pl.INT32],
    ready: pld.DistributedTensor[[2 * QUERY_SLOTS, 32], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Each peer block merges the same four leaves and sends its own exact ID tile."""
    row = pl.cast((epoch - 1) % 2, pl.INDEX) * QUERY_SLOTS + pl.cast(rank, pl.INDEX)
    with pl.spmd(7, name_hint="c4_owned_merge_publish_peer", allow_early_resolve=True) as sent:
        link = pl.cast(pl.tile.get_block_idx(), pl.INT32)
        packed = pl.mrgsort(leaf_pairs[0:1, :], leaf_pairs[1:2, :],
                            leaf_pairs[2:3, :], leaf_pairs[3:4, :])
        ids = pl.gather(packed[:, :TOPK_PAIR_WIDTH],
                        mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        if link == 0:
            topk_ids[0:1, :] = ids
        peer = link
        if peer >= rank:
            peer = pl.cast(peer + 1, pl.INT32)
        pld.tensor.remote_store(ids, ids_window, peer, [row, 0])
        pld.system.notify(target=ready, peer=peer, offsets=[row, 0],
                          value=epoch, op=pld.NotifyOp.Set)
    return sent


@pl.jit.inline(auto_scope=False)
def plan_from_remote_selection(
    ids_window: pld.DistributedTensor[[2 * QUERY_SLOTS, IDX_TOPK], pl.INT32],
    owned_ids: pl.Tensor[[1, IDX_TOPK], pl.INT32],
    ready: pld.DistributedTensor[[2*QUERY_SLOTS, 32], pl.INT32],
    positions: pl.Tensor[[1, QUERY_SLOTS], pl.INT32],
    window: pl.Tensor[[QUERY_SLOTS, WIN], pl.INT32],
    cmp_indices: pl.Tensor[[QUERY_SLOTS, CSA_CMP_TOPK], pl.INT32],
    sparse_bias: pl.Tensor[[QUERY_SLOTS, CSA_PADDED_TOPK], pl.FP32],
    descriptors: pl.Tensor[[QUERY_SLOTS, PADDED_PLAN_COLS], pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
    current_layer_ready: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Use the same planning arithmetic and each receiving rank's own SWA map."""
    lane = ((epoch - 1) % 2) * QUERY_SLOTS
    with pl.spmd(QUERY_SLOTS, name_hint="c4_plan_from_remote_selection", deps=[current_layer_ready], allow_early_resolve=True) as initialized:
        token = pl.tile.get_block_idx()
        position = pl.read(positions, [0, token])
        if token == rank:
            raw = pl.cast(pl.slice(owned_ids, [1, IDX_TOPK], [0, 0]), pl.FP32)
        else:
            pld.system.wait(signal=ready,offsets=[lane+token,0],expected=epoch,cmp=pld.WaitCmp.Ge)
            pl.system.cacheinvalid()
            raw = pl.cast(pl.slice(ids_window, [1, IDX_TOPK], [lane + token, 0]), pl.FP32)
        upper = pl.cast(pl.cast((position + 1) // CSA_COMPRESS_RATIO, pl.INT32), pl.FP32)
        upper_row = pl.mul(pl.full([1, IDX_TOPK], dtype=pl.FP32, value=1.0), upper)
        ge = pl.minimum(pl.maximum(pl.add(raw, CSA_CMP_GE_BIAS), 0.0), 1.0)
        lt = pl.minimum(pl.maximum(pl.sub(upper_row, raw), 0.0), 1.0)
        valid = pl.mul(ge, lt)
        mapped = pl.sub(pl.mul(valid, pl.add(raw, 1.0)), 1.0)
        cmp_indices[token:token + 1, :IDX_TOPK] = pl.cast(mapped, pl.INT32)

        win = pl.cast(window[token:token+1,:], pl.FP32)
        win_valid = pl.minimum(pl.maximum(pl.add(win, 1.0), 0.0), 1.0)
        sparse_bias[token:token + 1, :WIN] = pl.mul(pl.sub(win_valid, 1.0), -NEG_INF)
        sparse_bias[token:token + 1, WIN:CSA_TOPK] = pl.mul(
            pl.minimum(mapped, 0.0), -NEG_INF,
        )
        if CSA_PADDED_TOPK > CSA_TOPK:
            sparse_bias[token:token + 1, CSA_TOPK:CSA_PADDED_TOPK] = pl.full(
                [1, CSA_PADDED_TOPK-CSA_TOPK], pl.FP32, value=NEG_INF,
            )

        block_valid = pl.row_max(pl.reshape(valid, [CSA_CMP_TOPK // CSA_ATTN_K_TILE, CSA_ATTN_K_TILE]))
        mask = pl.cast((1 << WIN_BLOCKS) - 1, pl.INT32)
        count = pl.cast(WIN_BLOCKS, pl.INT32)
        for block in pl.unroll(WIN_BLOCKS, CSA_SPARSE_BLOCKS):
            bit = pl.cast(pl.read(block_valid, [block-WIN_BLOCKS, 0]), pl.INT32)
            count = count + bit
            weight = pl.cast(1 << block, pl.INT32)
            mask = mask + bit * weight
        row = pl.tile.full([1, PADDED_PLAN_COLS], dtype=pl.INT32, value=0)
        pl.tile.write(row, [0, 0], mask)
        pl.tile.write(row, [0, 1], count)
        pl.store(row, [token, 0], descriptors)
    return initialized
