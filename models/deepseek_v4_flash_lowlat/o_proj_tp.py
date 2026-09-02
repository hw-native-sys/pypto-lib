# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=8  # CI: 8-card TP run; the deployment world size, borrowed via task-submit --device-num
"""DeepSeek-V4 grouped output projection, sharded one group per card.

The projection is `attn_out = sum_g dequant(quant(o_packed[g] @ wo_a[g]) @ wo_b[:, g])`.
`O_GROUPS == TP == 8`, so the group axis is a shard axis with no input collective:
group g is heads [8g, 8g+8), whose `o_packed` rows every card already produced.

Card r runs chain r only, then all-reduces its partial across the eight cards.

There is no separate dequant pass: the per-group activation scale rides the loop
that already has to touch every partial, which here is `oproj_reduce`, on the far
side of the wire. So the raw INT32 partial is published, each rank's scale goes
with it, and the reduce folds `cast * scale` in as it sums -- in rank order, so
the sum reproduces a single card's `g` loop term for term.
"""

import sys as _sys

import pypto.language as pl
import pypto.language.distributed as pld

from config import (
    FLASH as M,
    DECODE_BATCH,
    DECODE_SEQ,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
    TP,
)


# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

# parallelism. Card r owns output group r, so the world size is the group count.
N_RANKS = TP

# tiling -- carried over from decode_sparse_attn_hca.py so the shard is the only
# variable between the two entries.
A_K_TILE = 256  # proj_a cube K frag
MM_T_TILE = 16
T_PAD = ((T + MM_T_TILE - 1) // MM_T_TILE) * MM_T_TILE
B_K_TILE = 256  # proj_b_mm cube K frag


# TP arm: the rank owns one group, so quant is on the serial chain instead of
# running beside seven siblings. Its O_LORA columns are the axis it fans out over.
def _parse_int_argv(name, default):
    for i, tok in enumerate(_sys.argv):
        if tok == name and i + 1 < len(_sys.argv):
            return int(_sys.argv[i + 1])
        if tok.startswith(f"{name}="):
            return int(tok.split("=", 1)[1])
    return default


# Every block recomputes the full-width amax, so a chunk buys only the quantize
# fan-out and pays a full amax for it. Measured on a2a3 with the PINNED ptoas
# v0.57, fastest-rank median: 69.0 / 69.3 / 69.9 us at 1 / 2 / 4 chunks -- the
# fan-out no longer pays. The two-task form it replaced, whose chunks shared one
# amax through GM, read 71.1 / 70.7 / 72.5 there. Re-sweep on any ptoas bump.
QUANT_CHUNKS = _parse_int_argv("--quant-chunks", 1)
# The publish put moves [T, D] FP32 = 128 KB. With chunk_cols == D it goes as a
# single unbuffered chunk; a smaller chunk plus pipeline=True double-buffers the
# VEC staging tile so one chunk is on the wire while the next loads.
PUT_CHUNK_COLS = _parse_int_argv("--put-chunk-cols", D)
PUT_PIPELINE = any(t == "--put-pipeline" for t in _sys.argv)
QUANT_COL_TILE = O_LORA // QUANT_CHUNKS
assert O_LORA % QUANT_CHUNKS == 0, "--quant-chunks must divide O_LORA"
# Both projections are M-starved at T=8 -- 16 rows against a 4096-deep K -- so they
# are bound by how many cores stream weights, not by cube throughput, and the block
# count is the lever. Measured on a2a3, fastest-rank median, at (proj_a blocks,
# proj_b blocks): 69.4 us at (8, 8), 67.3 at (8, 16), 67.1 at (16, 8), 65.0 at
# (16, 16), 68.2 at (32, 16). The two fan-outs are independent and additive, and
# both turn at 24 -- the AIC count -- because a 32-block grid runs two waves.
PROJ_A_MM_N_TILE = 64  # proj_a cube N frag; O_LORA // this = 16 blocks
PROJ_B_D_TILE = 256  # proj_b_mm D chunk per task; D // this = 16 blocks
# proj_b_mm cube N frag; writes INT32 partials. Equal to the D chunk, so each block
# runs one frag: spending the width on blocks beats splitting the frag (16 blocks x
# 2 frags of 128 reads 68.1 us).
PROJ_B_MM_N_TILE = 256

# Diagnostic: emit a leading barrier so host dispatch skew is paid BEFORE the
# measured chain instead of inside the reduce barrier. Parsed off argv because
# the kernel graph is frozen at import.
PRE_SYNC = any(t == "--pre-sync" for t in _sys.argv)
# Diagnostic: stream a buffer larger than L2 before the chain, so the projection
# reads its weights cold. Without it this microbenchmark keeps all 100.7 MB of
# wo_a + wo_b resident in the 192 MB L2 across rounds, which is the opposite of
# a full forward, where the MoE between layers evicts everything attention read.
EVICT_L2 = any(t == "--evict-l2" for t in _sys.argv)
EVICT_MB = 256
EVICT_BYTES = EVICT_MB * 1024 * 1024
EVICT_BLOCKS = 24
EVICT_PER_BLOCK = EVICT_BYTES // EVICT_BLOCKS // 512 * 512
EVICT_CHUNK = 32768
EVICT_TRIPS = (EVICT_PER_BLOCK // EVICT_CHUNK) if EVICT_L2 else 0
# Zero trip count disables the leading barrier without changing the graph.
PRE_SYNC_PEERS = N_RANKS if PRE_SYNC else 0

# All-reduce window: one `[T_PAD, D]` slot per rank per lane.
# Two lanes alternate by epoch so a rank can publish call e while a peer still
# reads call e-1, which keeps one barrier per call enough: reaching call e means
# every peer published e-1, hence finished reading e-2. A layer stack calls this
# projection once per CSA layer off one window, so the lanes are what make the
# reuse safe.
REDUCE_LANE_ROWS = N_RANKS * T_PAD
REDUCE_WINDOW_ROWS = 2 * REDUCE_LANE_ROWS
# One contiguous [T, 1] block per rank, so a peer reads its carrier with a
# direct ND2ND load and no reshape.
SCALE_LANE_ROWS = N_RANKS * T
SCALE_WINDOW_ROWS = 2 * SCALE_LANE_ROWS
# D chunk per reduce block, i.e. D // this = 8 blocks. This grid loads all eight
# rank slots per block, which keeps it element-bound enough that 8 blocks still pay:
# 1024 reads 65.4 us and 2048 reads 68.7 against 64.9 here.
REDUCE_D_TILE = 512
FIRST_EPOCH = 1

assert O_GROUPS == N_RANKS, f"TP-by-group needs one group per rank: O_GROUPS={O_GROUPS}, TP={N_RANKS}"


@pl.jit.inline
def _evict_l2(
    l2_evict: pl.Tensor[[EVICT_BLOCKS, EVICT_PER_BLOCK], pl.INT8],
    l2_evict_sink: pl.Tensor[[EVICT_BLOCKS, EVICT_CHUNK], pl.INT8],
) -> pl.Scalar[pl.TASK_ID]:
    """Stream a buffer larger than L2 so the projection below reads weights cold.

    Trip count is 0 unless --evict-l2, so both variants carry the same graph.
    """
    with pl.spmd(EVICT_BLOCKS, name_hint="l2_evict") as evict_tid:
        eb = pl.tile.get_block_idx()
        for chunk in pl.range(EVICT_TRIPS):
            c0 = chunk * EVICT_CHUNK
            l2_evict_sink[eb : eb + 1, 0:EVICT_CHUNK] = l2_evict[eb : eb + 1, c0 : c0 + EVICT_CHUNK]
    return evict_tid


@pl.jit.inline
def _quant(
    o_r_pad: pl.Tensor[[T_PAD, O_LORA], pl.FP32],
    o_r_i8_pad: pl.Tensor[[T_PAD, O_LORA], pl.INT8],
    act_scale_col: pl.Tensor[[T, 1], pl.FP32],
    pa_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Row amax over the full O_LORA width, then an INT8 quantize of one chunk.

    Blocks of one spmd cannot read each other's partials, so a chunked amax would
    need a second task to combine them. Re-reading every chunk instead keeps the
    quantize a single task: `max` is associative, so each block lands on the same
    amax, and the extra reads cost less than the task edge and its GM round trip.
    """
    with pl.spmd(QUANT_CHUNKS, name_hint="quant", deps=[pa_tid], allow_early_resolve=True) as q_tid:
        qb = pl.tile.get_block_idx()
        q0 = qb * QUANT_COL_TILE
        g_amax = pl.full([1, T], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for pc in pl.unroll(QUANT_CHUNKS):
            p0 = pc * QUANT_COL_TILE
            p_abs = pl.abs(o_r_pad[0:T, p0 : p0 + QUANT_COL_TILE])
            g_amax = pl.maximum(g_amax, pl.reshape(pl.row_max(p_abs), [1, T]))
        g_scale_num = pl.full([1, T], dtype=pl.FP32, value=INT8_SCALE_MAX)
        g_sq_row = pl.div(g_scale_num, g_amax)
        # Every block folds to the same scale; one block records it.
        if qb == 0:
            sc_row = pl.recip(g_sq_row)
            act_scale_col[0:T, 0:1] = pl.reshape(sc_row, [T, 1])
        g_sq_col = pl.reshape(g_sq_row, [T, 1])
        oc_q = o_r_pad[0:T, q0 : q0 + QUANT_COL_TILE]
        oq_scaled = pl.row_expand_mul(oc_q, g_sq_col)
        oq_i32 = pl.cast(oq_scaled, target_type=pl.INT32, mode="rint")
        oq_half = pl.cast(oq_i32, target_type=pl.FP16, mode="round")
        oq_i8 = pl.cast(oq_half, target_type=pl.INT8, mode="trunc")
        o_r_i8_pad[0:T, q0 : q0 + QUANT_COL_TILE] = oq_i8
    return q_tid


@pl.jit.inline
def _proj_chain(
    o_packed: pl.Tensor[[O_GROUPS * T, O_GROUP_IN], pl.BF16],
    wo_a_g: pl.Tensor[[1, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_g: pl.Tensor[[D, O_LORA], pl.INT8],
    o_r_pad: pl.Tensor[[T_PAD, O_LORA], pl.FP32],
    o_r_i8_pad: pl.Tensor[[T_PAD, O_LORA], pl.INT8],
    act_scale_col: pl.Tensor[[T, 1], pl.FP32],
    partial: pl.Tensor[[T_PAD, D], pl.INT32],
    row_base_o: pl.Scalar[pl.INDEX],
    upstream: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """One group's proj_a -> quant -> proj_b chain; returns proj_b's TaskId."""
    with pl.spmd(
        O_LORA // PROJ_A_MM_N_TILE, name_hint="proj_a_mm", deps=[upstream], allow_early_resolve=True
    ) as pa_tid:
        nf = pl.tile.get_block_idx()
        n0 = nf * PROJ_A_MM_N_TILE
        xa0_chunk = pl.slice(o_packed, [MM_T_TILE, A_K_TILE], [row_base_o, 0], valid_shape=[T, A_K_TILE])
        wa0_chunk = wo_a_g[0:1, n0 : n0 + PROJ_A_MM_N_TILE, 0:A_K_TILE]
        acc_a = pl.matmul(xa0_chunk, wa0_chunk, b_trans=True, out_dtype=pl.FP32)
        for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
            k0 = kb * A_K_TILE
            xa_k_chunk = pl.slice(
                o_packed, [MM_T_TILE, A_K_TILE], [row_base_o, k0], valid_shape=[T, A_K_TILE]
            )
            wa_k_chunk = wo_a_g[0:1, n0 : n0 + PROJ_A_MM_N_TILE, k0 : k0 + A_K_TILE]
            acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True)
        # acc_a is 3D (wo_a_g keeps its group axis), which subscript-write cannot express.
        o_r_pad = pl.assemble(o_r_pad, acc_a, [0, n0])

    q_tid = _quant(o_r_pad, o_r_i8_pad, act_scale_col, pa_tid)

    # The padding rows carry no data and depend on nothing, so they are zeroed
    # off the proj_a -> quant -> proj_b chain rather than inside quant, where the
    # fill is the same [T_PAD - T, O_LORA] element count as the real quantize.
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="quant_pad_zero", deps=[upstream], allow_early_resolve=True
    ) as pad_tid:
        zero_half = pl.full([T_PAD - T, O_LORA], dtype=pl.FP16, value=0.0)
        zero_i8 = pl.cast(zero_half, target_type=pl.INT8, mode="trunc")
        o_r_i8_pad[T:T_PAD, 0:O_LORA] = zero_i8

    with pl.spmd(
        D // PROJ_B_D_TILE,
        name_hint="proj_b_mm",
        deps=[q_tid, pad_tid],
        allow_early_resolve=True,
        optimizations=[pl.cross_core_slot(slot_num=2)],
    ) as pb_tid:
        dc = pl.tile.get_block_idx()
        d0 = dc * PROJ_B_D_TILE
        for nf in pl.range(PROJ_B_D_TILE // PROJ_B_MM_N_TILE):
            n0 = d0 + nf * PROJ_B_MM_N_TILE
            acc_b = pl.matmul(
                o_r_i8_pad[:, 0:B_K_TILE],
                wo_b_g[n0 : n0 + PROJ_B_MM_N_TILE, 0:B_K_TILE],
                b_trans=True,
                out_dtype=pl.INT32,
            )
            for kb in pl.pipeline(1, O_LORA // B_K_TILE, stage=2):
                k0 = kb * B_K_TILE
                acc_b = pl.matmul_acc(
                    acc_b,
                    o_r_i8_pad[:, k0 : k0 + B_K_TILE],
                    wo_b_g[n0 : n0 + PROJ_B_MM_N_TILE, k0 : k0 + B_K_TILE],
                    b_trans=True,
                )
            partial[0:MM_T_TILE, n0 : n0 + PROJ_B_MM_N_TILE] = acc_b
    return pb_tid


# === TP-by-group ============================================================
@pl.jit.inline
def o_proj_tp_core(
    o_packed: pl.Tensor[[O_GROUPS * T, O_GROUP_IN], pl.BF16],
    upstream: pl.Scalar[pl.TASK_ID],
    wo_a_shard: pl.Tensor[[1, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_shard: pl.Tensor[[D, O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Tensor[[T, D], pl.BF16],
    reduce_window: pld.DistributedTensor[[REDUCE_WINDOW_ROWS, D], pl.INT32],
    scale_window: pld.DistributedTensor[[SCALE_WINDOW_ROWS, 1], pl.FP32],
    reduce_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    sync_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    # 1-based call id; the signals are monotonic within a dispatch, so waits use
    # `>= oproj_epoch` and `clear_oproj_signals` resets them once at the end.
    oproj_epoch: pl.Scalar[pl.INT32],
):
    """Card `my_rank` runs group `my_rank`'s chain, then all-reduces its partial.

    The dequantized partial is published as FP32 rather than the INT32 accumulator
    because each group carries its own activation scale, which cannot factor out
    of the cross-rank sum. `wo_b_scale` is a shared per-channel factor and is
    applied once, after the sum, so the arithmetic matches a single card's g loop.
    """
    o_r_pad = pl.create_tensor([T_PAD, O_LORA], dtype=pl.FP32)
    o_r_i8_pad = pl.create_tensor([T_PAD, O_LORA], dtype=pl.INT8)
    act_scale_col = pl.create_tensor([T, 1], dtype=pl.FP32)
    partial = pl.create_tensor([T_PAD, D], dtype=pl.INT32)
    barrier_out = pl.array.create(1, pl.TASK_ID)

    lane_base = pl.cast(((oproj_epoch - 1) % 2) * REDUCE_LANE_ROWS, pl.INDEX)
    scale_lane_base = pl.cast(((oproj_epoch - 1) % 2) * SCALE_LANE_ROWS, pl.INDEX)
    my_row = lane_base + pl.cast(my_rank, pl.INDEX) * T_PAD
    my_scale_row = scale_lane_base + pl.cast(my_rank, pl.INDEX) * T
    row_base_o = pl.cast(my_rank, pl.INDEX) * T
    wo_b_scale_2d = pl.reshape(wo_b_scale, [1, D])

    with pl.manual_scope():
        # Anchor the chain on a read of the local attention output, so proj_a
        # carries a real edge instead of being ready at submit.
        # Leading barrier. Emitted unconditionally so both variants carry the
        # same graph shape; --pre-sync only changes the trip count from 0 to
        # N_RANKS. With it on, host dispatch skew is absorbed here and the reduce
        # barrier below measures protocol cost instead of rank start spread.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="oproj_pre_sync") as _sync_tid:
            _sync_anchor = pl.read(o_packed, [0, 0])
            for peer in pl.range(PRE_SYNC_PEERS):
                pld.system.notify(
                    target=sync_signal,
                    peer=peer,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
            for src in pl.range(PRE_SYNC_PEERS):
                pld.system.wait(
                    signal=sync_signal,
                    offsets=[src, 0],
                    expected=oproj_epoch,
                    cmp=pld.WaitCmp.Ge,
                )
        anchor = pl.system.task_dummy(deps=[_sync_tid, upstream])

        pb_tid = _proj_chain(
            o_packed,
            wo_a_shard,
            wo_b_shard,
            o_r_pad,
            o_r_i8_pad,
            act_scale_col,
            partial,
            row_base_o,
            anchor,
        )

        # Dequantize by this group's per-row activation scale. The per-channel
        # weight scale is deliberately NOT applied here -- it rides the reduce so
        # the sum happens on the same quantity a single card's g loop sums.
        # When the dequant rides proj_b's cube task there is no separate grid, so the
        # publish chains straight off proj_b. An Array[TASK_ID] is the sanctioned way
        # to carry a TaskId out of the branch that produced it.
        # No dequant pass. The per-group scale rides the 8-way accumulate, because
        # that loop already touches every partial; sharded, that accumulate is
        # oproj_reduce, on the far side of the wire. So the raw INT32 partial goes
        # out and each source's scale is folded in as the reduce sums it -- term for
        # term a single card's g loop.
        with pl.spmd(N_RANKS, name_hint="oproj_publish", deps=[pb_tid]) as publish_tid:
            peer = pl.tile.get_block_idx()
            pld.tensor.put(
                dst=reduce_window,
                peer=peer,
                src=partial,
                dst_offsets=[my_row, 0],
                src_offsets=[0, 0],
                shape=[T, D],
                chunk_rows=T,
                chunk_cols=PUT_CHUNK_COLS,
                pipeline=PUT_PIPELINE,
            )
            pld.tensor.put(
                dst=scale_window,
                peer=peer,
                src=act_scale_col,
                dst_offsets=[my_scale_row, 0],
                src_offsets=[0, 0],
                shape=[T, 1],
            )
            pld.system.notify(
                target=reduce_signal,
                peer=peer,
                offsets=[my_rank, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )

        # Split from the push so the notify rides the push scope's program order
        # and only the wait holds a core group.
        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="oproj_barrier",
            deps=[pb_tid],
            allow_early_resolve=True,
        ) as barrier_tid:
            for src in pl.range(N_RANKS):
                pld.system.wait(
                    signal=reduce_signal,
                    offsets=[src, 0],
                    expected=oproj_epoch,
                    cmp=pld.WaitCmp.Ge,
                )

        # Sum the rank slots in rank order -- the same term order a single card's
        # g loop uses -- then apply the per-channel weight scale and narrow to
        # BF16. Peer slots are addressed by a loop variable, so the window is
        # read through pl.load; the weight scale is loaded the same way to keep
        # both multiply operands at tile level.
        barrier_out[0] = barrier_tid

    # The reduce is the sole attn_out writer and must sit OUTSIDE the manual scope:
    # manual_scope suppresses auto-dep, so a consumer of attn_out in the caller
    # (hc_post, in the full layer) would carry no edge to this write and would read
    # the buffer while the reduce is still filling it. The barrier's TaskId rides an
    # Array[TASK_ID], the sanctioned way out of a scope that has closed.
    # One block per D tile carrying all T rows: [1, REDUCE_D_TILE] blocks make this
    # 64 tasks whose per-task fixed cost dominates 2 KB of loads.
    with pl.spmd(
        D // REDUCE_D_TILE, name_hint="oproj_reduce", deps=[barrier_out[0]], allow_early_resolve=True
    ) as _reduce_tid:
        d0 = pl.tile.get_block_idx() * REDUCE_D_TILE
        # Seeded from rank 0 rather than a zero tile so the accumulator stays at
        # tile level, and so the term order matches a single card's g loop.
        acc = pl.row_expand_mul(
            pl.cast(
                pl.load(reduce_window, [lane_base, d0], [T, REDUCE_D_TILE]), target_type=pl.FP32, mode="none"
            ),
            pl.load(scale_window, [scale_lane_base, 0], [T, 1]),
        )
        for src_rank in pl.range(1, N_RANKS):
            p_src = pl.cast(
                pl.load(reduce_window, [lane_base + src_rank * T_PAD, d0], [T, REDUCE_D_TILE]),
                target_type=pl.FP32,
                mode="none",
            )
            acc = pl.add(
                acc,
                pl.row_expand_mul(p_src, pl.load(scale_window, [scale_lane_base + src_rank * T, 0], [T, 1])),
            )
        wb_chunk = pl.load(wo_b_scale_2d, [0, d0], [1, REDUCE_D_TILE])
        scaled = pl.col_expand_mul(acc, wb_chunk)
        pl.store(pl.cast(scaled, target_type=pl.BF16, mode="rint"), [0, d0], attn_out)

    return attn_out


@pl.jit.inline
def clear_oproj_signals(
    completion_anchor: pl.Tensor[[T, D], pl.BF16],
    reduce_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    sync_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
):
    """Clear this rank's counters after its final projection in this dispatch."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="oproj_signal_clear"):
        # The final output depends on this rank observing every peer's final
        # notify, so no peer can issue another one in this dispatch.
        _completion_anchor = pl.read(completion_anchor, [0, 0])
        zero = pl.cast(0, pl.INT32)
        for src in pl.range(N_RANKS):
            pl.write(reduce_signal, [src, 0], zero)
            pl.write(sync_signal, [src, 0], zero)


@pl.jit.inline
def o_proj_tp(
    o_packed: pl.Tensor[[O_GROUPS * T, O_GROUP_IN], pl.BF16],
    wo_a_shard: pl.Tensor[[1, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_shard: pl.Tensor[[D, O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    l2_evict: pl.Tensor[[EVICT_BLOCKS, EVICT_PER_BLOCK], pl.INT8],
    l2_evict_sink: pl.Tensor[[EVICT_BLOCKS, EVICT_CHUNK], pl.INT8],
    attn_out: pl.Tensor[[T, D], pl.BF16],
    reduce_window: pld.DistributedTensor[[REDUCE_WINDOW_ROWS, D], pl.INT32],
    scale_window: pld.DistributedTensor[[SCALE_WINDOW_ROWS, 1], pl.FP32],
    reduce_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    sync_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    oproj_epoch: pl.Scalar[pl.INT32],
):
    """Standalone entry: the L2-eviction probe, then the shared projection core."""
    evict_tid = _evict_l2(l2_evict, l2_evict_sink)
    o_proj_tp_core(
        o_packed,
        evict_tid,
        wo_a_shard,
        wo_b_shard,
        wo_b_scale,
        attn_out,
        reduce_window,
        scale_window,
        reduce_signal,
        sync_signal,
        my_rank,
        oproj_epoch,
    )
    clear_oproj_signals(attn_out, reduce_signal, sync_signal)
    return attn_out


# === Test entries ===========================================================
@pl.jit
def o_proj_tp_test(
    o_packed: pl.Tensor[[O_GROUPS * T, O_GROUP_IN], pl.BF16],
    wo_a_shard: pl.Tensor[[1, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_shard: pl.Tensor[[D, O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    l2_evict: pl.Tensor[[EVICT_BLOCKS, EVICT_PER_BLOCK], pl.INT8],
    l2_evict_sink: pl.Tensor[[EVICT_BLOCKS, EVICT_CHUNK], pl.INT8],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    reduce_window: pld.DistributedTensor[[REDUCE_WINDOW_ROWS, D], pl.INT32],
    scale_window: pld.DistributedTensor[[SCALE_WINDOW_ROWS, 1], pl.FP32],
    reduce_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    sync_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    oproj_epoch: pl.Scalar[pl.INT32],
):
    return o_proj_tp(
        o_packed,
        wo_a_shard,
        wo_b_shard,
        wo_b_scale,
        l2_evict,
        l2_evict_sink,
        attn_out,
        reduce_window,
        scale_window,
        reduce_signal,
        sync_signal,
        my_rank,
        oproj_epoch,
    )


@pl.jit.host
def l3_o_proj_tp(
    o_packed: pl.Tensor[[N_RANKS, O_GROUPS * T, O_GROUP_IN], pl.BF16],
    wo_a_shard: pl.Tensor[[N_RANKS, 1, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_shard: pl.Tensor[[N_RANKS, D, O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    l2_evict: pl.Tensor[[N_RANKS, EVICT_BLOCKS, EVICT_PER_BLOCK], pl.INT8],
    l2_evict_sink: pl.Tensor[[N_RANKS, EVICT_BLOCKS, EVICT_CHUNK], pl.INT8],
    attn_out: pl.Out[pl.Tensor[[N_RANKS, T, D], pl.BF16]],
):
    reduce_window_buf = pld.alloc_window_buffer([REDUCE_WINDOW_ROWS, D], dtype=pl.INT32)
    scale_window_buf = pld.alloc_window_buffer([SCALE_WINDOW_ROWS, 1], dtype=pl.FP32)
    reduce_signal_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    sync_signal_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)

    for r in pl.range(pld.world_size()):
        reduce_window = pld.window(reduce_window_buf, [REDUCE_WINDOW_ROWS, D], dtype=pl.INT32)
        scale_window = pld.window(scale_window_buf, [SCALE_WINDOW_ROWS, 1], dtype=pl.FP32)
        reduce_signal = pld.window(reduce_signal_buf, [N_RANKS, 1], dtype=pl.INT32)
        sync_signal = pld.window(sync_signal_buf, [N_RANKS, 1], dtype=pl.INT32)
        o_proj_tp_test(
            o_packed[r],
            wo_a_shard[r],
            wo_b_shard[r],
            wo_b_scale[r],
            l2_evict[r],
            l2_evict_sink[r],
            attn_out[r],
            reduce_window,
            scale_window,
            reduce_signal,
            sync_signal,
            r,
            FIRST_EPOCH,
            device=r,
        )


# === Golden =================================================================
def _golden_o_proj(o_packed, wo_a, wo_b_i8, wo_b_scale):
    """Torch reference, term for term with decode_sparse_attn_hca's grouped o_proj."""
    import torch

    o_model = o_packed.float().reshape(O_GROUPS, T, O_GROUP_IN)
    # o_packed row g*T + t holds group g's token t, so the einsum is per group.
    o_r_g = torch.einsum("gtd,grd->tgr", o_model, wo_a.float())
    amax_g = o_r_g.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_q_g = INT8_SCALE_MAX / amax_g
    o_r_i8_g = torch.round(o_r_g * scale_q_g).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dq_g = 1.0 / scale_q_g
    wo_b_g = wo_b_i8.reshape(D, O_GROUPS, O_LORA)
    out = torch.zeros(T, D, dtype=torch.float32)
    for g in range(O_GROUPS):
        p_g = o_r_i8_g[:, g].to(torch.int32) @ wo_b_g[:, g].to(torch.int32).T
        out = out + p_g.float() * scale_dq_g[:, g]
    out = out * wo_b_scale.float().unsqueeze(0)
    return out.to(torch.bfloat16)


def golden_tp(tensors):
    import torch

    # Rebuild the full weights from the per-rank shards: rank r holds group r.
    wo_a_full = torch.cat([tensors["wo_a_shard"][r] for r in range(N_RANKS)], dim=0)
    wo_b_full = torch.cat([tensors["wo_b_shard"][r] for r in range(N_RANKS)], dim=1)
    for r in range(tensors["o_packed"].shape[0]):
        tensors["attn_out"][r] = _golden_o_proj(
            tensors["o_packed"][r],
            wo_a_full,
            wo_b_full,
            tensors["wo_b_scale"][r],
        )


# === Fixture ================================================================
def _base_tensors():
    """One deterministic weight set, so a re-run compares against the same numbers."""
    import torch

    gen = torch.Generator().manual_seed(20260902)
    o_packed = (torch.randn(O_GROUPS * T, O_GROUP_IN, generator=gen) * 0.1).to(torch.bfloat16)
    wo_a = (torch.randn(O_GROUPS, O_LORA, O_GROUP_IN, generator=gen) / O_GROUP_IN**0.5).to(torch.bfloat16)
    wo_b = torch.randint(-127, 128, (D, O_GROUPS * O_LORA), generator=gen, dtype=torch.int32).to(torch.int8)
    wo_b_scale = (torch.rand(D, generator=gen) * 0.001 + 0.0005).to(torch.float32)
    return o_packed, wo_a, wo_b, wo_b_scale


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    o_packed, wo_a, wo_b, wo_b_scale = _base_tensors()
    stack = lambda x: torch.stack([x] * N_RANKS, dim=0)

    specs = [
        TensorSpec(
            "o_packed",
            [N_RANKS, O_GROUPS * T, O_GROUP_IN],
            torch.bfloat16,
            init_value=lambda: stack(o_packed),
        ),
    ]
    specs += [
        # Card r carries group r's proj_a rows and proj_b columns only.
        TensorSpec(
            "wo_a_shard",
            [N_RANKS, 1, O_LORA, O_GROUP_IN],
            torch.bfloat16,
            init_value=lambda: torch.stack([wo_a[r : r + 1] for r in range(N_RANKS)], dim=0),
            resident="stacked",
        ),
        TensorSpec(
            "wo_b_shard",
            [N_RANKS, D, O_LORA],
            torch.int8,
            init_value=lambda: torch.stack(
                [wo_b[:, r * O_LORA : (r + 1) * O_LORA] for r in range(N_RANKS)], dim=0
            ),
            resident="stacked",
        ),
        TensorSpec("wo_b_scale", [N_RANKS, D], torch.float32, init_value=lambda: stack(wo_b_scale)),
        # Resident so the stream costs bandwidth, not an upload, every round.
        TensorSpec(
            "l2_evict",
            [N_RANKS, EVICT_BLOCKS, EVICT_PER_BLOCK],
            torch.int8,
            init_value=lambda: torch.ones(N_RANKS, EVICT_BLOCKS, EVICT_PER_BLOCK, dtype=torch.int8),
            resident="stacked",
        ),
        TensorSpec(
            "l2_evict_sink",
            [N_RANKS, EVICT_BLOCKS, EVICT_CHUNK],
            torch.int8,
            init_value=lambda: torch.zeros(N_RANKS, EVICT_BLOCKS, EVICT_CHUNK, dtype=torch.int8),
            resident="stacked",
        ),
        TensorSpec("attn_out", [N_RANKS, T, D], torch.bfloat16, is_output=True),
    ]
    return specs


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit
    from pypto.ir.distributed_compiled_program import DistributedConfig

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"]
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=",".join(str(i) for i in range(N_RANKS)),
        help=f"comma-separated device ids; need at least {N_RANKS}",
    )
    parser.add_argument(
        "--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2, 4)
    )
    parser.add_argument("--enable-pmu", nargs="?", const=2, default=0, type=int, choices=[0, 1, 2, 4])
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument(
        "--put-chunk-cols",
        type=int,
        default=PUT_CHUNK_COLS,
        help="Column chunk for the publish put; smaller chunks with "
        "--put-pipeline double-buffer the staging tile.",
    )
    parser.add_argument("--put-pipeline", action="store_true", default=False)
    parser.add_argument(
        "--quant-chunks",
        type=int,
        default=QUANT_CHUNKS,
        help="O_LORA column chunks the quantize fans out over; each recomputes the amax.",
    )
    parser.add_argument(
        "--evict-l2",
        action="store_true",
        default=False,
        help=f"Stream {EVICT_MB} MB before the chain so the projection reads weights cold.",
    )
    parser.add_argument(
        "--pre-sync",
        action="store_true",
        default=False,
        help="Emit a leading barrier so dispatch skew is paid before the measured chain.",
    )
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) >= N_RANKS, f"need at least {N_RANKS} devices, got {device_ids}"

    result = run_jit(
        fn=l3_o_proj_tp,
        specs=build_tensor_specs(),
        golden_fn=golden_tp,
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        compile_cfg=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:N_RANKS],
                num_sub_workers=0,
            ),
        ),
        runtime_cfg=dict(
            platform=args.platform,
            enable_chip_swimlane=args.enable_chip_swimlane,
            enable_pmu=args.enable_pmu,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={"attn_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128)},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
