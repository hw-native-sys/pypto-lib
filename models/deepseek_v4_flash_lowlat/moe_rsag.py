# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto.language as pl

import pypto.language.distributed as pld

from .moe import prepare_moe, moe_local

from .config import FLASH as M
from .decode_shapes import MOE_TOKENS, TP

from .expert_routed import N_BANK

from .hc_post import hc_post


T = MOE_TOKENS

D = M.hidden_size

TOPK = M.num_experts_per_tok

VOCAB = M.vocab_size

HC_MULT = M.hc_mult

MIX_HC = M.mix_hc

HC_DIM = M.hc_dim

MOE_INTER = M.moe_intermediate_size // TP  # this rank's slice of the intermediate dim

N_EXPERTS = M.n_routed_experts

N_RANKS = TP

T_PAD = ((T + 15) // 16) * 16

REDUCE_LANE_ROWS = N_RANKS * T_PAD

REDUCE_WINDOW_ROWS = 2 * REDUCE_LANE_ROWS

REDUCE_SHARD_D = D // N_RANKS

assert D % N_RANKS == 0 and REDUCE_SHARD_D % 512 == 0

REDUCE_D_TILE = 512  # D // this = 8 reduce blocks, each summing [T, 512] tiles


@pl.jit.inline
def clear_moe_signals(
    completion_anchor: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    reduce_signal: pld.DistributedTensor[[N_RANKS, 32], pl.INT32],
    gather_signal: pld.DistributedTensor[[N_RANKS, 32], pl.INT32],
):
    """Clear this rank's all-reduce counters after its final MoE completes."""
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="moe_signal_clear", allow_early_resolve=True
    ) as s6_moe_cleanup_tid:
        # The final MoE output depends on this rank observing every peer's final
        # publish notify, so no peer can issue another one in this forward.
        _completion_anchor = pl.read(completion_anchor, [0, 0, 0])
        zero = pl.cast(0, pl.INT32)
        for src in pl.range(N_RANKS):
            pl.write(reduce_signal, [src, 0], zero)
            pl.write(gather_signal, [src, 0], zero)
    return s6_moe_cleanup_tid


@pl.jit.inline
def all_reduce_ffn(
    ffn_partial: pl.Tensor[[T, D], pl.FP32],
    ffn_out: pl.Tensor[[T, D], pl.BF16],
    reduce_window: pld.DistributedTensor[[REDUCE_WINDOW_ROWS, REDUCE_SHARD_D], pl.FP32],
    reduce_signal: pld.DistributedTensor[[N_RANKS, 32], pl.INT32],
    gather_window: pld.DistributedTensor[[REDUCE_WINDOW_ROWS, REDUCE_SHARD_D], pl.BF16],
    gather_signal: pld.DistributedTensor[[N_RANKS, 32], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    """Exact logical0..TP-1 FP32 order, then one BF16 cast per output element.

    Both independent counters preserve the original +1/Ge epoch protocol and
    self publication. No diagnostic loop/gate/host read enters production.
    """
    lane_base = pl.cast(((moe_epoch - 1) % 2) * REDUCE_LANE_ROWS, pl.INDEX)
    my_row = lane_base + pl.cast(my_rank, pl.INDEX) * T_PAD
    my_column = pl.cast(my_rank, pl.INDEX) * REDUCE_SHARD_D
    with pl.spmd(N_RANKS, name_hint="moe_reduce_publish", allow_early_resolve=True) as publish_tid:
        peer = pl.tile.get_block_idx()
        pld.tensor.put(
            dst=reduce_window,
            peer=peer,
            src=ffn_partial,
            dst_offsets=[my_row, 0],
            src_offsets=[0, peer * REDUCE_SHARD_D],
            shape=[T, REDUCE_SHARD_D],
            chunk_rows=T,
            chunk_cols=REDUCE_SHARD_D,
        )
        pld.system.notify(
            target=reduce_signal, peer=peer, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd
        )
    with pl.spmd(
        REDUCE_SHARD_D // REDUCE_D_TILE,
        name_hint="moe_reduce_wait_fused",
        allow_early_resolve=True,
        deps=[publish_tid],
    ) as reduce_tid:
        for src in pl.range(N_RANKS):
            pld.system.wait(signal=reduce_signal, offsets=[src, 0], expected=moe_epoch, cmp=pld.WaitCmp.Ge)
        d0 = pl.tile.get_block_idx() * REDUCE_D_TILE
        acc = pl.load(reduce_window, [lane_base, d0], [T, REDUCE_D_TILE])
        for src_rank in pl.range(1, N_RANKS):
            src_row = lane_base + src_rank * T_PAD
            acc = pl.add(acc, pl.load(reduce_window, [src_row, d0], [T, REDUCE_D_TILE]))
        reduced = pl.cast(acc, target_type=pl.BF16, mode="rint")
        pl.store(reduced, [0, my_column + d0], ffn_out)
    with pl.spmd(
        N_RANKS, name_hint="moe_bf16_publish", allow_early_resolve=True, deps=[reduce_tid]
    ) as gather_publish_tid:
        peer = pl.tile.get_block_idx()
        pld.tensor.put(
            dst=gather_window,
            peer=peer,
            src=ffn_out,
            dst_offsets=[my_row, 0],
            src_offsets=[0, my_column],
            shape=[T, REDUCE_SHARD_D],
            chunk_rows=T,
            chunk_cols=REDUCE_D_TILE,
        )
        pld.system.notify(
            target=gather_signal, peer=peer, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd
        )
    with pl.spmd(
        D // REDUCE_D_TILE,
        name_hint="moe_bf16_wait_gather",
        allow_early_resolve=True,
        deps=[gather_publish_tid],
    ):
        block = pl.tile.get_block_idx()
        owner = block // (REDUCE_SHARD_D // REDUCE_D_TILE)
        pld.system.wait(signal=gather_signal, offsets=[owner, 0], expected=moe_epoch, cmp=pld.WaitCmp.Ge)
        d0 = (block % (REDUCE_SHARD_D // REDUCE_D_TILE)) * REDUCE_D_TILE
        value = pl.load(gather_window, [lane_base + owner * T_PAD, d0], [T, REDUCE_D_TILE])
        pl.store(value, [0, block * REDUCE_D_TILE], ffn_out)


@pl.jit.inline
def reduce_ffn_tp1(
    ffn_partial: pl.Tensor[[T, D], pl.FP32],
    ffn_out: pl.Tensor[[T, D], pl.BF16],
):
    """Single-rank path: the partial is already the full sum."""
    with pl.spmd(D // REDUCE_D_TILE, name_hint="moe_reduce", allow_early_resolve=True):
        d0 = pl.tile.get_block_idx() * REDUCE_D_TILE
        partial = ffn_partial[0:T, d0 : d0 + REDUCE_D_TILE]
        ffn_out[0:T, d0 : d0 + REDUCE_D_TILE] = pl.cast(partial, target_type=pl.BF16, mode="rint")


@pl.jit.inline(auto_scope=False)
def moe(
    # model inputs
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    # Expert SHARED_EID of each bank is the shared expert (see route_group).
    routed_w1: pl.Tensor[[N_BANK, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_BANK, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_BANK, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_BANK, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_BANK, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_BANK, D], pl.FP32],
    # final output
    x_next: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    # windows
    reduce_window: pld.DistributedTensor[[REDUCE_WINDOW_ROWS, REDUCE_SHARD_D], pl.FP32],
    reduce_signal: pld.DistributedTensor[[N_RANKS, 32], pl.INT32],
    gather_window: pld.DistributedTensor[[REDUCE_WINDOW_ROWS, REDUCE_SHARD_D], pl.BF16],
    gather_signal: pld.DistributedTensor[[N_RANKS, 32], pl.INT32],
    # scalars last: runtime TaskArgs forbids a tensor arg after a scalar arg.
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    # 1-based MoE call id for the shared reduce window (distinct from layer_id).
    moe_epoch: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T, HC_MULT, D], pl.FP32]:
    # Non-output intermediates allocate locally, in their producer's scope.
    (post_ffn, comb_ffn, recv_x, recv_scale, recv_w, work_slot, work_rows,
     slot_expert, route_slot_row) = prepare_moe(
        x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, norm_w, gate_w, gate_bias,
        tid2eid, input_ids, layer_id, num_tokens,
    )

    with pl.scope():
        ffn_partial = moe_local(
            recv_x, recv_scale, recv_w, work_slot, work_rows, slot_expert, route_slot_row,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
            routed_w2, routed_w2_scale, num_tokens,
        )

        ffn_out = pl.create_tensor([T, D], dtype=pl.BF16)
        if N_RANKS == 1:
            reduce_ffn_tp1(ffn_partial, ffn_out)
        else:
            all_reduce_ffn(
                ffn_partial,
                ffn_out,
                reduce_window,
                reduce_signal,
                gather_window,
                gather_signal,
                my_rank,
                moe_epoch,
            )

        hc_post(ffn_out, x_hc, post_ffn, comb_ffn, x_next)
    return x_next
