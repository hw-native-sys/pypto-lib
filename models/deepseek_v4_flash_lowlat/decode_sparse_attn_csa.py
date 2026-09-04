# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 CSA sparse attention with grouped output projection (decode).

Ratio-4 compressed cache plus the sliding window, with the indexer top-k
masking folded in. The SWA and HCA variants live in sibling modules.
"""


import pypto.language as pl
import pypto.language.distributed as pld


from config import (
    FLASH as M,
    DECODE_BATCH,
    DECODE_SEQ,
    BLOCK_SIZE,
    DECODE_CMP_BLOCK_NUM,
    DECODE_ORI_BLOCK_NUM,
    KV_CMP_MAX_BLOCKS,
    KV_ORI_MAX_BLOCKS,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
)


# Dynamic shape variables.
ORI_BLOCK_NUM_DYN = pl.dynamic("ORI_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("CMP_BLOCK_NUM_DYN")

# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
WIN = M.sliding_window
MAX_SEQ_LEN = M.max_position_embeddings
IDX_TOPK = M.index_topk
CMP_TOPK = IDX_TOPK
SOFTMAX_SCALE = M.softmax_scale
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
# Attention is token-sharded: card `my_rank` computes every head of token
# `my_rank`, and the group axis is restored by an all-to-all before o_proj.
H_LOCAL = H
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
COMPRESS_RATIO = 4
CMP_STORAGE_BLOCK_SIZE = BLOCK_SIZE // COMPRESS_RATIO
COMPRESS_RATIO_INV = 1.0 / COMPRESS_RATIO
INDEXER_SCORE_LEN = MAX_SEQ_LEN // 4
CSA_CMP_GE_BIAS = 1.0  # raw + 1, folded for the ge clamp
NEG_INF = -1.0e20

# paged KV cache
ORI_MAX_BLOCKS = KV_ORI_MAX_BLOCKS
ORI_BLOCK_NUM = DECODE_ORI_BLOCK_NUM
CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
CMP_BLOCK_NUM = DECODE_CMP_BLOCK_NUM

# === token shard ===========================================================
# Card r owns token r and computes ALL H heads for it, instead of owning
# H_LOCAL heads of every token. T == N_RANKS is what makes the map one-to-one.
N_RANKS = O_GROUPS
assert T == N_RANKS, f"the token shard needs one token per card, got T={T} N_RANKS={N_RANKS}"
# q arrives head-sharded (card p holds heads [8p, 8p+8) of every token) and a
# token-sharded card needs all 64 heads of its own token, so the two axes are
# transposed by an all-to-all: 8 heads x HEAD_DIM per (src, dst) pair.
Q_SHARD_ROWS = HEADS_PER_GROUP
# o_packed is consumed group-sharded by o_proj, so merge_norm's token-sharded
# output is transposed back the same way: one O_GROUP_IN row per (src, dst).
O_SHARD_COLS = HEADS_PER_GROUP * HEAD_DIM

# tiling
H_TILE = 8               # merge_norm head tile
# One token per card gives the cube a full 64-row M, so every qk/pv box is
# 16 valid rows instead of the 8-in-16 the head shard was forced into.
QK_M_TILE = 16
QK_M_BOX = 16
QK_HEAD_TILES = H // QK_M_TILE
# K tile. The head shard ran 128 with T * 5 = 40 work items; one token leaves 5,
# so the tile shrinks to keep every lane fed off a single token's rows.
ATTN_K_TILE = 32
NUM_QK_CORES = 24        # qk_pv dispatch lanes = a2a3 AIC count; re-sweep for other AIC counts
MM_T_TILE = 16
T_PAD = ((T + MM_T_TILE - 1) // MM_T_TILE) * MM_T_TILE
# proj_b_mm cube N frag; Acc = MM_T_TILE*N*4 = 128KB sits exactly on the a2a3 L0C wall.
# Fused amax+quant token tile. 8 keeps the [1, QUANT_TOKEN_TILE] fp32 amax tile
# 32-byte aligned (8*4=32B, the alloc-tile row floor).
QUANT_TOKEN_TILE = 8
TOPK = WIN + CMP_TOPK
# Floor to 2: a single sparse-K block miscompiles in pypto (S-stride cross-token
# output mixup); a 2-block build with an all-invalid 2nd block is bit-exact.
SPARSE_BLOCKS = max(2, (TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE)
PADDED_TOPK = SPARSE_BLOCKS * ATTN_K_TILE
WIN_BLOCKS = WIN // ATTN_K_TILE   # leading tiles that are pure sliding window
assert WIN % ATTN_K_TILE == 0, "the window must tile whole so a tile is window or compressed, never both"
# One token per card, so a work item is one sparse block of that token.
QK_ITEMS = SPARSE_BLOCKS
# Page-contiguous runs one sliding-window K tile spans: WIN rows capped by the tile,
# plus a worst-case BLOCK_SIZE - 1 head offset, rounded up to pages.
SWA_TILE_WIN_ROWS = min(ATTN_K_TILE, WIN)
SWA_RUNS = (SWA_TILE_WIN_ROWS + 2 * (BLOCK_SIZE - 1)) // BLOCK_SIZE


@pl.jit.inline
def csa_q_all_to_all(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    q_window: pld.DistributedTensor[[H, HEAD_DIM], pl.BF16],
    q_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    tok_epoch: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Transpose q from the head shard onto the token shard: push only.

    Card `my_rank` holds heads [8*my_rank, 8*my_rank+8) of every token and card
    `peer` wants all H heads of token `peer`, so the slice this card owes `peer`
    is one 8-head block of one token -- 8 KB. Issued here rather than next to its
    consumer so the compressor and the indexer, neither of which reads q, cover
    the flight; only the wait sits in front of `qk_pv`.
    """
    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    my_head0 = pl.cast(my_rank, pl.INDEX) * Q_SHARD_ROWS
    # The edge to the q projection is the tensormap read of q_flat: its spmd is a
    # for-form and never names a TaskId. Verify it in deps.json after a build --
    # a missing edge here would race the push against the projection.
    with pl.spmd(N_RANKS, name_hint="csa_q_a2a") as q_push_tid:
        peer = pl.tile.get_block_idx()
        pld.tensor.put(
            dst=q_window,
            peer=peer,
            src=q_flat,
            dst_offsets=[my_head0, 0],
            src_offsets=[peer * H + my_head0, 0],
            shape=[Q_SHARD_ROWS, HEAD_DIM],
            chunk_rows=Q_SHARD_ROWS,
            chunk_cols=HEAD_DIM,
        )
        pld.system.notify(
            target=q_signal,
            peer=peer,
            offsets=[my_rank, 0],
            value=1,
            op=pld.NotifyOp.AtomicAdd,
        )
    return q_push_tid


@pl.jit.inline
def sparse_attn_csa_packed(
    q_window: pld.DistributedTensor[[H, HEAD_DIM], pl.BF16],
    q_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    q_push_tid: pl.Scalar[pl.TASK_ID],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    window_swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    idx_topk: pl.Tensor[[T, INDEXER_SCORE_LEN], pl.INT32],
    position_ids: pl.Tensor[[T, 1], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    o_packed: pl.Tensor[[O_GROUPS * T, O_GROUP_IN], pl.BF16],
    o_window: pld.DistributedTensor[[T, O_SHARD_COLS], pl.BF16],
    o_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    tok_epoch: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Token-sharded sparse decode attention, up to the packed head output.

    Card `my_rank` owns token `my_rank` and computes every head of it, so the
    scattered compressed gather -- the term that bounds `qk_pv`, and the one
    thing here that is per-token rather than per-head -- costs this card one
    token's rows instead of all T. The head axis is restored on the way out.
    """
    # Compressed index contract: -1 invalid, [0, ...) compressed KV slots.
    ori_block_num = pl.tensor.dim(ori_kv, 0)
    ori_kv_flat = pl.reshape(ori_kv, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    my_t = pl.cast(my_rank, pl.INDEX)

    # WAR marker (pypto-lib#481): a scalar-driven gather_row does not mark ori_kv
    # add_inout, so the layer's KV writeback would lose its WAR edge against the
    # qk_pv gather read. add_inout is param-level, so this no-op self-copy suffices.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_touch", allow_early_resolve=True):
        ori_kv_flat[0:T, 0:HEAD_DIM] = ori_kv_flat[0:T, 0:HEAD_DIM]

    # This card plans only its own token, so qk_order holds SPARSE_BLOCKS items.
    sparse_bias = pl.create_tensor([T, PADDED_TOPK], dtype=pl.FP32)
    cmp_sparse_indices = pl.create_tensor([T, CMP_TOPK], dtype=pl.INT32)
    valid_block_mask = pl.create_tensor([T, SPARSE_BLOCKS], dtype=pl.INT32)
    qk_order = pl.create_tensor([QK_ITEMS], dtype=pl.INT32)
    qk_wcur = pl.create_tensor([1], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_slots_build_valid_qk_plan", allow_early_resolve=True) as qk_plan_tid:
        # Compressed slots [0, IDX_TOPK): vectorized masked copy over all T rows, keeping
        # raw iff 0 <= raw < floor((pos + 1) / COMPRESS_RATIO), as out = mask*(raw + 1) - 1.
        c_raw = pl.cast(idx_topk[0:T, 0:IDX_TOPK], target_type=pl.FP32)
        c_pos = pl.cast(position_ids[0:T, 0:1], target_type=pl.FP32)
        c_pos_scaled = pl.mul(pl.add(c_pos, 1.0), COMPRESS_RATIO_INV)
        c_pos_i32 = pl.cast(c_pos_scaled, target_type=pl.INT32, mode="trunc")
        c_pos_q = pl.cast(c_pos_i32, target_type=pl.FP32)
        # Broadcast the per-token bound over IDX_TOPK cols.
        c_upper_b = pl.row_expand_mul(pl.full([T, IDX_TOPK], dtype=pl.FP32, value=1.0), c_pos_q)
        c_ge = pl.minimum(pl.maximum(pl.add(c_raw, CSA_CMP_GE_BIAS), 0.0), 1.0)
        c_lt = pl.minimum(pl.maximum(pl.sub(c_upper_b, c_raw), 0.0), 1.0)
        c_mask = pl.mul(c_ge, c_lt)
        c_out = pl.sub(pl.mul(c_mask, pl.add(c_raw, 1.0)), 1.0)
        cmp_sparse_indices[0:T, 0:IDX_TOPK] = pl.cast(c_out, target_type=pl.INT32)
        # Window blocks are always live; compressed blocks from the mask.
        for c_t0 in pl.range(T):
            for c_wb in pl.range(WIN_BLOCKS):
                pl.write(valid_block_mask, [c_t0, c_wb], pl.cast(1, pl.INT32))
        for c_sb in pl.range(WIN_BLOCKS, SPARSE_BLOCKS):
            c_s0 = (c_sb - WIN_BLOCKS) * ATTN_K_TILE
            c_blk_valid = pl.row_max(c_mask[:, c_s0 : c_s0 + ATTN_K_TILE])
            for c_dt in pl.range(T):
                c_valid = pl.cast(pl.read(c_blk_valid, [c_dt, 0]), target_type=pl.INT32)
                pl.write(valid_block_mask, [c_dt, c_sb], c_valid)

        # Additive softmax bias (0 valid / NEG_INF invalid) that qk_pv adds onto the
        # scaled scores, so invalid lanes exp to ~0 with no per-block mask multiply.
        v_win_f = pl.cast(window_swa_indices[0:T, 0:WIN], target_type=pl.FP32)
        # Index contract: raw == -1 invalid, raw >= 0 valid. min(idx, 0) is -1 for
        # invalid / 0 for valid; * -NEG_INF gives NEG_INF / 0.
        v_win_valid = pl.minimum(pl.maximum(pl.add(v_win_f, 1.0), 0.0), 1.0)
        sparse_bias[0:T, 0:WIN] = pl.mul(pl.sub(v_win_valid, 1.0), -NEG_INF)
        sparse_bias[0:T, WIN:TOPK] = pl.mul(pl.minimum(c_out, 0.0), -NEG_INF)
        if PADDED_TOPK > TOPK:
            sparse_bias[0:T, TOPK:PADDED_TOPK] = pl.full([T, PADDED_TOPK - TOPK], dtype=pl.FP32, value=NEG_INF)

        pl.write(qk_wcur, [0], pl.cast(0, pl.INT32))
        # Pass 1: this card's non-empty tiles to the front of qk_order.
        for plan_sb in pl.range(SPARSE_BLOCKS):
            if pl.read(valid_block_mask, [my_t, plan_sb]) > 0:
                plan_w = pl.read(qk_wcur, [0])
                pl.write(qk_order, [plan_w], pl.cast(plan_sb, pl.INT32))
                pl.write(qk_wcur, [0], pl.cast(plan_w + 1, pl.INT32))
        # Pass 2: empty tiles appended to the tail.
        for plan_sb in pl.range(SPARSE_BLOCKS):
            if pl.read(valid_block_mask, [my_t, plan_sb]) <= 0:
                plan_w = pl.read(qk_wcur, [0])
                pl.write(qk_order, [plan_w], pl.cast(plan_sb, pl.INT32))
                pl.write(qk_wcur, [0], pl.cast(plan_w + 1, pl.INT32))

    # Land the transposed q before qk_pv reads it. Staged into a plain tensor so
    # the cube slices an ordinary operand; the copy is one [H, HEAD_DIM] tile.
    # Flagged so qk_pv can be staged speculatively: this is a producer-side hint
    # that releases the CONSUMER, not a request to early-dispatch the wait itself,
    # so it does not put a spinning wait ahead of the traffic it waits on. Without
    # it qk_pv has an unflagged direct producer and cannot be early-dispatched at
    # all, the plan scope's own flag notwithstanding.
    q_tok = pl.create_tensor([H, HEAD_DIM], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_q_a2a_wait", deps=[q_push_tid],
               allow_early_resolve=True) as q_wait_tid:
        for q_src in pl.range(N_RANKS):
            pld.system.wait(
                signal=q_signal,
                offsets=[q_src, 0],
                expected=tok_epoch,
                cmp=pld.WaitCmp.Ge,
            )
        pl.store(pl.load(q_window, [0, 0], [H, HEAD_DIM]), [0, 0], q_tok)

    # One lane per core over this card's own token's K tiles. Each lane gathers a
    # tile into L1 and runs every head against it, so the H axis rides the tile
    # that is already resident instead of costing another gather.
    cmp_block_num = pl.tensor.dim(cmp_kv, 0)
    cmp_kv_flat = pl.reshape(cmp_kv, [cmp_block_num * CMP_STORAGE_BLOCK_SIZE, HEAD_DIM])
    sparse_blk_mi = pl.create_tensor([SPARSE_BLOCKS * H, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([SPARSE_BLOCKS * H, 1], dtype=pl.FP32)
    sparse_blk_oi = pl.create_tensor([SPARSE_BLOCKS * H, HEAD_DIM], dtype=pl.FP32)

    with pl.spmd(NUM_QK_CORES, name_hint="qk_pv", deps=[qk_plan_tid, q_wait_tid], allow_early_resolve=True) as qk_tid:
        qk_core = pl.tile.get_block_idx()
        qk_lane_iters = (QK_ITEMS - qk_core + NUM_QK_CORES - 1) // NUM_QK_CORES
        for qk_it in pl.range(qk_lane_iters):
            qk_flat = qk_core + qk_it * NUM_QK_CORES
            qk_sb = pl.cast(pl.read(qk_order, [qk_flat]), pl.INDEX)
            qk_b = my_t // S
            qk_s0 = qk_sb * ATTN_K_TILE
            qk_bias_row = sparse_bias[my_t : my_t + 1, qk_s0 : qk_s0 + ATTN_K_TILE]
            qk_block_valid = pl.read(valid_block_mask, [my_t, qk_sb])
            qk_row = qk_sb * H
            if qk_block_valid > 0:
                qk_kv = pl.create_l1([ATTN_K_TILE, HEAD_DIM], pl.BF16)
                # Sliding-window rows of this tile: all ATTN_K_TILE of them for a
                # window tile, none for a compressed tile.
                qk_win_rows = pl.min(pl.max(WIN - qk_s0, 0), ATTN_K_TILE)
                if qk_win_rows > 0:
                    # Window rows are consecutive absolute positions and paged KV keeps a
                    # page's positions consecutive, so they form SWA_RUNS page-contiguous
                    # runs -- one multi-row gather each, row count via valid_shape.
                    qk_pos = pl.cast(pl.read(position_ids, [my_t, 0]), pl.INDEX)
                    qk_win_len = pl.min(qk_pos + 1, WIN)
                    qk_win_start = qk_pos - qk_win_len + 1
                    qk_run_rows = pl.min(pl.max(qk_win_len - qk_s0, 0), qk_win_rows)
                    qk_head = (qk_win_start + qk_s0) % BLOCK_SIZE
                    for qk_run in pl.unroll(SWA_RUNS):
                        qk_run_lo = pl.max(qk_run * BLOCK_SIZE - qk_head, 0)
                        qk_run_hi = pl.min((qk_run + 1) * BLOCK_SIZE - qk_head, qk_run_rows)
                        if qk_run_hi > qk_run_lo:
                            qk_run_raw = pl.read(window_swa_indices, [my_t, qk_s0 + qk_run_lo])
                            # An unmapped page (-1) falls back to row 0 like the tail below
                            # -- every such slot is NEG_INF-masked by sparse_bias.
                            qk_run_src = pl.cast(pl.max(qk_run_raw, 0), pl.INDEX)
                            qk_kv = pl.gather_row(qk_kv, ori_kv_flat, [qk_run_lo, 0], [qk_run_src, 0],
                                                  [ATTN_K_TILE, HEAD_DIM],
                                                  valid_shape=[qk_run_hi - qk_run_lo, HEAD_DIM])
                    qk_tail_n = qk_win_rows - qk_run_rows
                    if qk_tail_n > 0:
                        # Slots past the visible window still need finite data so their
                        # NEG_INF-biased lanes exp to ~0 instead of reading stale L1.
                        qk_kv = pl.gather_row(qk_kv, ori_kv_flat, [qk_run_rows, 0], [0, 0],
                                              [ATTN_K_TILE, HEAD_DIM], valid_shape=[qk_tail_n, HEAD_DIM])
                # Compressed rows stay per-row: the indexer top-k slots are scattered.
                for qk_r in pl.range(qk_win_rows, ATTN_K_TILE):
                    qk_cmp_k = qk_s0 + qk_r - WIN
                    if qk_cmp_k < CMP_TOPK:
                        qk_ridx = pl.read(cmp_sparse_indices, [my_t, qk_cmp_k])
                        if qk_ridx >= 0:
                            qk_slot = qk_ridx
                            qk_cblk = pl.cast(
                                pl.read(
                                    cmp_block_table,
                                    [qk_b, qk_slot // CMP_STORAGE_BLOCK_SIZE],
                                ),
                                pl.INDEX,
                            )
                            qk_csrc = qk_cblk * CMP_STORAGE_BLOCK_SIZE + qk_slot % CMP_STORAGE_BLOCK_SIZE
                            qk_kv = pl.gather_row(qk_kv, cmp_kv_flat, [qk_r, 0], [qk_csrc, 0], [1, HEAD_DIM])
                        else:
                            qk_kv = pl.gather_row(qk_kv, ori_kv_flat, [qk_r, 0], [0, 0], [1, HEAD_DIM])
                    else:
                        qk_kv = pl.gather_row(qk_kv, ori_kv_flat, [qk_r, 0], [0, 0], [1, HEAD_DIM])

                # All H heads run against the resident tile, QK_M_TILE rows per
                # matmul. One token per card fills the cube's 16-row box exactly,
                # so no box carries padding rows.
                for qk_hb in pl.pipeline(QK_HEAD_TILES, stage=2):
                    qk_h0 = qk_hb * QK_M_TILE
                    qk_q_tile = q_tok[qk_h0 : qk_h0 + QK_M_TILE, 0:HEAD_DIM]
                    qk_raw = pl.matmul(qk_q_tile, qk_kv, b_trans=True, out_dtype=pl.FP32)
                    qk_scaled = pl.mul(qk_raw, SOFTMAX_SCALE)
                    # Per-block bias broadcast-added in one op.
                    qk_scores = pl.col_expand_add(qk_scaled, qk_bias_row)
                    qk_mi = pl.row_max(qk_scores)
                    # Invalid lanes (NEG_INF bias) exp to ~0; all-invalid blocks die in
                    # the merge alpha/beta, so no mask multiply is needed.
                    qk_exp = pl.exp(pl.row_expand_sub(qk_scores, qk_mi))
                    qk_li = pl.row_sum(qk_exp)
                    qk_exp_bf16 = pl.cast(qk_exp, target_type=pl.BF16, mode="rint")
                    qk_oi = pl.matmul(qk_exp_bf16, qk_kv, out_dtype=pl.FP32)
                    qk_hrow = qk_row + qk_h0
                    sparse_blk_mi[qk_hrow : qk_hrow + QK_M_TILE, 0 : 1] = qk_mi
                    sparse_blk_li[qk_hrow : qk_hrow + QK_M_TILE, 0 : 1] = qk_li
                    sparse_blk_oi[qk_hrow : qk_hrow + QK_M_TILE, 0 : HEAD_DIM] = qk_oi
            else:
                qk_oi_zero = pl.full([QK_M_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
                for qk_hb in pl.range(QK_HEAD_TILES):
                    qk_hrow = qk_row + qk_hb * QK_M_TILE
                    for qk_hr in pl.range(QK_M_TILE):
                        pl.write(sparse_blk_mi, [qk_hrow + qk_hr, 0], -3.0e38)
                        pl.write(sparse_blk_li, [qk_hrow + qk_hr, 0], 0.0)
                    sparse_blk_oi[qk_hrow : qk_hrow + QK_M_TILE, 0 : HEAD_DIM] = qk_oi_zero

    # Head-invariant interleaved cos and sign-folded sin for this card's token.
    # The conjugate (inverse) rotation is out[j] = x[j]*cos_il[j] + x[j^1]*sign[j]*sin_il[j].
    rope_cos_il = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    # j^1 lane-swap index for merge_norm's rotation gather. Shaped [H_TILE, ROPE_DIM]
    # because gather's index must match its source rows.
    rope_swap_idx = pl.create_tensor([H_TILE, ROPE_DIM], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_cs", allow_early_resolve=True):
        sw_ones = pl.full([H_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        sw_idx_f = pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32)
        sw_col = pl.col_expand_mul(sw_ones, sw_idx_f)
        sw_dup_i32 = pl.cast(pl.mul(sw_col, 0.5), target_type=pl.INT32, mode="trunc")
        sw_dup_f = pl.cast(sw_dup_i32, target_type=pl.FP32)
        sw_lane = pl.sub(sw_col, pl.mul(sw_dup_f, 2.0))                                           # j%2
        sw_swap_f = pl.sub(pl.add(sw_col, 1.0), pl.mul(sw_lane, 2.0))                             # j^1
        rope_swap_idx[0:H_TILE, 0:ROPE_DIM] = pl.cast(sw_swap_f, target_type=pl.INT32)

        cs_ones = pl.full([T, ROPE_DIM], dtype=pl.FP32, value=1.0)
        cs_idx_f = pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32)
        cs_col = pl.col_expand_mul(cs_ones, cs_idx_f)
        cs_dup_i32 = pl.cast(pl.mul(cs_col, 0.5), target_type=pl.INT32, mode="trunc")
        cs_dup_f = pl.cast(cs_dup_i32, target_type=pl.FP32)
        cs_dup_idx = pl.cast(cs_dup_f, target_type=pl.INT32)                                      # j>>1
        cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))                                           # j%2
        cs_sign = pl.neg(pl.sub(pl.mul(cs_lane, 2.0), 1.0))                                       # [+1,-1,...] (conjugate)
        cs_cos = pl.cast(freqs_cos[0:T, 0:HALF_ROPE], target_type=pl.FP32)
        cs_sin = pl.cast(freqs_sin[0:T, 0:HALF_ROPE], target_type=pl.FP32)
        rope_cos_il[0:T, 0:ROPE_DIM] = pl.gather(cs_cos, dim=-1, index=cs_dup_idx)
        cs_sin_il = pl.gather(cs_sin, dim=-1, index=cs_dup_idx)
        rope_sin_signed[0:T, 0:ROPE_DIM] = pl.mul(cs_sin_il, cs_sign)

    # Online-softmax merge across this token's sparse-K tiles, sink-norm, then fused
    # inverse RoPE, one spmd block per head tile. A block covers exactly one output
    # group, so its store is one contiguous O_GROUP_IN row of the push buffer.
    o_tok = pl.create_tensor([O_GROUPS, O_SHARD_COLS], dtype=pl.BF16)
    with pl.spmd(H // H_TILE, name_hint="merge_norm", allow_early_resolve=True) as merge_tid:
        m_g = pl.tile.get_block_idx()
        m_h0 = m_g * H_TILE
        m_mi = sparse_blk_mi[m_h0 : m_h0 + H_TILE, 0 : 1]
        m_li = sparse_blk_li[m_h0 : m_h0 + H_TILE, 0 : 1]
        m_oi = sparse_blk_oi[m_h0 : m_h0 + H_TILE, 0 : HEAD_DIM]

        for m_sb in pl.pipeline(1, SPARSE_BLOCKS, stage=2):
            m_row = m_sb * H + m_h0
            m_cur_mi = sparse_blk_mi[m_row : m_row + H_TILE, 0 : 1]
            m_cur_li = sparse_blk_li[m_row : m_row + H_TILE, 0 : 1]
            m_cur_oi = sparse_blk_oi[m_row : m_row + H_TILE, 0 : HEAD_DIM]
            m_mi_new = pl.maximum(m_mi, m_cur_mi)
            m_alpha = pl.exp(pl.sub(m_mi, m_mi_new))
            m_beta = pl.exp(pl.sub(m_cur_mi, m_mi_new))
            m_li = pl.add(pl.mul(m_alpha, m_li), pl.mul(m_beta, m_cur_li))
            m_oi = pl.add(pl.row_expand_mul(m_oi, m_alpha), pl.row_expand_mul(m_cur_oi, m_beta))
            m_mi = m_mi_new

        n_sink_bias = pl.reshape(attn_sink[m_h0 : m_h0 + H_TILE], [H_TILE, 1])
        n_sink_tile = pl.add(pl.sub(m_mi, m_mi), n_sink_bias)
        n_denom = pl.add(m_li, pl.exp(pl.sub(n_sink_tile, m_mi)))
        n_full = pl.row_expand_div(m_oi, n_denom)[0 : H_TILE, 0 : HEAD_DIM]
        n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode="rint")

        # Inverse RoPE on this head-tile's fp32 rope segment. cos_il / sign*sin are
        # head-invariant for this card's token, so col_expand them over the H_TILE
        # head rows; rope_swap_idx (j^1) pairs the interleaved real/imag lanes.
        m_rope = n_full[0 : H_TILE, NOPE_DIM : HEAD_DIM]
        m_cos_il = rope_cos_il[my_t : my_t + 1, 0 : ROPE_DIM]
        m_sin_signed = rope_sin_signed[my_t : my_t + 1, 0 : ROPE_DIM]
        m_swapped = pl.gather(m_rope, dim=-1, index=rope_swap_idx[0:H_TILE, 0:ROPE_DIM])
        m_rot = pl.add(pl.col_expand_mul(m_rope, m_cos_il), pl.col_expand_mul(m_swapped, m_sin_signed))
        n_rope_bf16 = pl.cast(m_rot, target_type=pl.BF16, mode="rint")
        n_full_bf16 = pl.concat(n_bf16[:, : NOPE_DIM], n_rope_bf16)

        for n_hi in pl.unroll(H_TILE):
            n_col = n_hi * HEAD_DIM
            # Nope and inverse-RoPE halves concatenated on chip: one contiguous store.
            o_tok[m_g : m_g + 1, n_col : n_col + HEAD_DIM] = n_full_bf16[n_hi : n_hi + 1, :]

        # Transpose this group back onto the shard o_proj consumes, from the block
        # that just produced it. Group m_g's only consumer is card m_g, so the push
        # needs nothing the block does not already hold, and a block that finishes
        # early puts its 8 KB on the wire while its siblings are still merging --
        # a separate grid after merge_norm would serialize all eight behind the
        # slowest one.
        pld.tensor.put(
            dst=o_window,
            peer=m_g,
            src=o_tok,
            dst_offsets=[my_t, 0],
            src_offsets=[m_g, 0],
            shape=[1, O_SHARD_COLS],
            chunk_rows=1,
            chunk_cols=O_SHARD_COLS,
        )
        pld.system.notify(
            target=o_signal,
            peer=m_g,
            offsets=[my_rank, 0],
            value=1,
            op=pld.NotifyOp.AtomicAdd,
        )

    # Land this card's group into the rows its own o_proj shard reads. Split from
    # the push so the notify rides merge_norm's program order.
    o_base = pl.cast(my_rank, pl.INDEX) * T
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_o_a2a_wait", deps=[merge_tid],
               allow_early_resolve=True) as o_wait_tid:
        for o_src in pl.range(N_RANKS):
            pld.system.wait(
                signal=o_signal,
                offsets=[o_src, 0],
                expected=tok_epoch,
                cmp=pld.WaitCmp.Ge,
            )
        o_landed = pl.load(o_window, [0, 0], [T, O_SHARD_COLS])
        pl.store(o_landed, [o_base, 0], o_packed)

    return o_wait_tid




# The replicated single-card `sparse_attn_csa` / `sparse_attn_test` entries are
# dropped in this variant: the token shard is defined by the layer's comm
# windows, so there is no one-card form of it to unit-test.

def golden_sparse_attn(tensors):
    """Torch reference: sparse_attn decode path followed by grouped o_proj."""
    import torch

    q = tensors["q"].float()
    ori_kv = tensors["ori_kv"].float()
    window_swa_indices = tensors["window_swa_indices"]
    cmp_kv = tensors["cmp_kv"].float()
    cmp_block_table = tensors["cmp_block_table"]
    # Compressed slots: keep raw indexer topk iff 0 <= raw < floor((pos + 1) / COMPRESS_RATIO), else -1.
    raw = tensors["idx_topk"][:, :CMP_TOPK].to(torch.int64)
    bound = ((tensors["position_ids"][:, 0].to(torch.int64) + 1) // COMPRESS_RATIO).unsqueeze(1)
    keep = (raw >= 0) & (raw < bound)
    cmp_sparse_indices = torch.where(keep, raw, torch.full_like(raw, -1)).to(torch.int32)
    attn_sink = tensors["attn_sink"].float()
    cos = tensors["freqs_cos"].float()
    sin = tensors["freqs_sin"].float()
    wo_a = tensors["wo_a"].float()
    wo_b_i8 = tensors["wo_b"]
    wo_b_scale = tensors["wo_b_scale"].float()

    o = torch.zeros(T, H, HEAD_DIM)

    # Per-query-token attention. The window prefix is driven by window_swa_indices;
    # cmp_sparse_indices contains compressed-cache slots only.
    for t in range(T):
        b = t // S
        kv_rows = []
        valid = []

        for raw in window_swa_indices[t].tolist():
            slot = int(raw)
            if slot >= 0:
                blk_id = slot // BLOCK_SIZE
                intra = slot % BLOCK_SIZE
                kv_rows.append(ori_kv[blk_id, intra, 0])
                valid.append(True)
            else:
                kv_rows.append(torch.zeros(HEAD_DIM, dtype=ori_kv.dtype))
                valid.append(False)

        for raw in cmp_sparse_indices[t].tolist():
            if raw < 0:
                kv_rows.append(torch.zeros(HEAD_DIM, dtype=ori_kv.dtype))
                valid.append(False)
                continue
            cmp_slot = int(raw)
            block_id = int(cmp_block_table[b, cmp_slot // CMP_STORAGE_BLOCK_SIZE].item())
            kv_rows.append(cmp_kv[block_id, cmp_slot % CMP_STORAGE_BLOCK_SIZE, 0])
            valid.append(True)

        if not any(valid):
            continue

        pad_k = PADDED_TOPK - TOPK
        if pad_k:
            kv_rows.extend(torch.zeros(HEAD_DIM, dtype=ori_kv.dtype) for _ in range(pad_k))
            valid.extend(False for _ in range(pad_k))

        kv_b = torch.stack(kv_rows, dim=0)
        valid_b = torch.tensor(valid, dtype=torch.bool)
        q_t = q[t]

        block_mi = []
        block_li = []
        block_oi = []
        for tile_start in range(0, PADDED_TOPK, ATTN_K_TILE):
            kv_tile = kv_b[tile_start:tile_start + ATTN_K_TILE]
            valid_tile = valid_b[tile_start:tile_start + ATTN_K_TILE]
            scores = (q_t @ kv_tile.T) * SOFTMAX_SCALE
            scores = scores.masked_fill(~valid_tile.unsqueeze(0), NEG_INF)
            mi = scores.max(dim=-1, keepdim=True).values
            exp_scores = torch.exp(scores - mi).masked_fill(~valid_tile.unsqueeze(0), 0.0)
            li = exp_scores.sum(dim=-1, keepdim=True)
            oi = exp_scores.to(torch.bfloat16).float() @ kv_tile.to(torch.bfloat16).float()
            block_mi.append(mi)
            block_li.append(li)
            block_oi.append(oi)

        score_max = block_mi[0]
        li = block_li[0]
        oi_num = block_oi[0]
        for mi_cur, li_cur, oi_cur in zip(block_mi[1:], block_li[1:], block_oi[1:]):
            score_max_new = torch.maximum(score_max, mi_cur)
            alpha = torch.exp(score_max - score_max_new)
            beta = torch.exp(mi_cur - score_max_new)
            li = alpha * li + beta * li_cur
            oi_num = alpha * oi_num + beta * oi_cur
            score_max = score_max_new

        denom = li + torch.exp(attn_sink.unsqueeze(-1) - score_max)
        o[t] = oi_num / denom

    rope_pair = o[..., NOPE_DIM:].unflatten(-1, (-1, 2))
    rope_even = rope_pair[..., 0]
    rope_odd = rope_pair[..., 1]
    cos_half = cos[:, :HALF_ROPE].unsqueeze(1)
    sin_half = sin[:, :HALF_ROPE].unsqueeze(1)
    inv_even = (rope_even * cos_half + rope_odd * sin_half).to(torch.bfloat16).float()
    inv_odd = (rope_odd * cos_half - rope_even * sin_half).to(torch.bfloat16).float()
    o_rope = torch.stack([inv_even, inv_odd], dim=-1).flatten(-2)
    o = torch.cat([o[..., :NOPE_DIM], o_rope], dim=-1).to(torch.bfloat16)

    seq_per_batch = T // B
    o_model = o.float().view(B, seq_per_batch, O_GROUPS, O_GROUP_IN)
    o_r = torch.einsum("bsgd,grd->bsgr", o_model, wo_a)
    # PER-GROUP INT8 activation quant: one amax per O_LORA group. Each group's INT32
    # partial is dequantized by its OWN per-row act scale before the groups are summed
    # (the per-group scale cannot factor out of the K-sum), then the channel scale.
    o_r_g = o_r.reshape(T, O_GROUPS, O_LORA)
    amax_g = o_r_g.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)   # [T, G, 1]
    scale_q_g = INT8_SCALE_MAX / amax_g
    o_r_i8_g = torch.round(o_r_g * scale_q_g).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dq_g = 1.0 / scale_q_g                                              # [T, G, 1]
    wo_b_g = wo_b_i8.reshape(D, O_GROUPS, O_LORA)
    out = torch.zeros(T, D, dtype=torch.float32)
    for g in range(O_GROUPS):
        p_g = o_r_i8_g[:, g].to(torch.int32) @ wo_b_g[:, g].to(torch.int32).T   # [T, D]
        out = out + p_g.float() * scale_dq_g[:, g]                             # per-row group scale
    out = out * wo_b_scale.unsqueeze(0)                                        # per-channel weight scale

    tensors["attn_out"][:] = out.to(torch.bfloat16)

def build_tensor_specs(
    causal_regression_fixture: bool = False,
    short_window_fixture: bool = False,
    mixed_topk_fixture: bool = False,
    cache_window_replacement_fixture: bool = False,
):
    """Build deterministic demo tensors for the CSA standalone harness."""
    import torch
    from golden import TensorSpec
    from utils import block_table, quant_w_per_channel, swa_indices_and_lens
    from utils import build_rope_tables, materialize_token_rope_tables

    cmp_valid = IDX_TOPK
    shared_freqs_cos, shared_freqs_sin = build_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)
    rope_positions = torch.arange(T, dtype=torch.int32)
    shared_rope_cos, shared_rope_sin = materialize_token_rope_tables(shared_freqs_cos, shared_freqs_sin, rope_positions)

    def init_q():
        """Initialize the query tensor used by the decode attention stage."""
        q = torch.rand(T, H, HEAD_DIM) - 0.5
        if causal_regression_fixture:
            q[0].fill_(1.0)
        return q

    def init_ori_kv():
        """Initialize the sliding-window KV cache pages."""
        kv = torch.rand(ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM) - 0.5
        if causal_regression_fixture:
            kv[0, WIN - 1, 0].fill_(8.0)
        if cache_window_replacement_fixture:
            kv[0, 16, 0].fill_(0.0)
            kv[0, 16, 0, 0] = 4.0
        return kv

    def init_window_swa_indices():
        """Lower the window through the same producer the model uses.

        Indexing the block table by window slot instead of absolute position
        would keep every row of a WIN == BLOCK_SIZE window inside one page, so
        the fixture could not tell a correct page-run split from a broken one.
        Going through swa_indices_and_lens straddles a page boundary whenever
        init_position_ids is not page-aligned, which it is not.
        """
        positions = init_position_ids().reshape(B, S)
        return swa_indices_and_lens(
            positions,
            init_window_block_table(),
            block_size=BLOCK_SIZE,
            window=WIN,
        )[0].contiguous()

    def init_cmp_kv():
        """Initialize the compressed-cache KV pages."""
        return torch.rand(CMP_BLOCK_NUM, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM) - 0.5

    def init_attn_sink():
        """Initialize the per-head sink logits to zero."""
        return torch.zeros(H)

    def init_window_block_table():
        """Build the demo block table for the sliding-window cache pages."""
        return block_table(batch=B, table_blocks=ORI_MAX_BLOCKS, physical_blocks=ORI_BLOCK_NUM)

    def init_cmp_block_table():
        """Build the demo block table for the compressed-cache pages."""
        rows = torch.arange(CMP_MAX_BLOCKS, dtype=torch.int32) % CMP_BLOCK_NUM
        return rows.unsqueeze(0).expand(B, -1).clone()

    def init_cmp_sparse_indices():
        """Build the compressed sparse index list."""
        indices = torch.full((T, CMP_TOPK), -1, dtype=torch.int32)
        indices[:, :cmp_valid] = torch.arange(cmp_valid, dtype=torch.int32).unsqueeze(0).expand(T, -1)
        if short_window_fixture:
            indices[:, :] = -1
            indices[:, :17] = torch.arange(17, dtype=torch.int32).unsqueeze(0).expand(T, -1)
        if mixed_topk_fixture:
            indices[:, :] = -1
            mixed_cmp_valid = min(cmp_valid, IDX_TOPK)
            if mixed_cmp_valid:
                indices[:, :mixed_cmp_valid] = torch.arange(mixed_cmp_valid, dtype=torch.int32).unsqueeze(0).expand(T, -1)
        if cache_window_replacement_fixture:
            indices[:, :] = -1
        if causal_regression_fixture:
            indices[0, :] = -1
        return indices

    def init_idx_topk():
        """Raw indexer topk feeding sparse_attn's compressed-slot masking. Only the
        first CMP_TOPK cols are read; identity mask here (see init_position_ids), so
        the masked output equals this fixture pattern."""
        topk = torch.full((T, INDEXER_SCORE_LEN), -1, dtype=torch.int32)
        topk[:, :CMP_TOPK] = init_cmp_sparse_indices()
        return topk

    def init_position_ids():
        """Large enough that floor((pos + 1) / COMPRESS_RATIO) >= CMP_TOPK, so the
        per-token bound never clips the fixture slots (mask reduces to raw >= 0)."""
        return torch.full((T, 1), COMPRESS_RATIO * CMP_TOPK, dtype=torch.int32)

    def init_cos():
        """Build the split-half cosine table used by the inverse-RoPE reference."""
        return shared_rope_cos.clone()

    def init_sin():
        """Build the split-half sine table used by the inverse-RoPE reference."""
        return shared_rope_sin.clone()

    def init_wo_a():
        """Initialize the grouped first-stage output-projection weights."""
        return (torch.rand(O_GROUPS, O_LORA, O_GROUP_IN) - 0.5) / (O_GROUP_IN ** 0.5)

    wo_b_bf16 = ((torch.rand(D, O_GROUPS * O_LORA) - 0.5) / ((O_GROUPS * O_LORA) ** 0.5)).to(torch.bfloat16)
    wo_b_i8, wo_b_scale = quant_w_per_channel(wo_b_bf16)

    def init_wo_b():
        """Initialize the second-stage output-projection weights in per-channel INT8 form."""
        return wo_b_i8

    def init_wo_b_scale():
        """Initialize the dequant scales paired with the INT8 second-stage weights."""
        return wo_b_scale

    return [
        TensorSpec("q", [T, H, HEAD_DIM], torch.bfloat16, init_value=init_q),
        TensorSpec("ori_kv", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_ori_kv),
        TensorSpec("window_swa_indices", [T, WIN], torch.int32, init_value=init_window_swa_indices),
        TensorSpec("cmp_kv", [CMP_BLOCK_NUM, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table", [B, CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("idx_topk", [T, INDEXER_SCORE_LEN], torch.int32, init_value=init_idx_topk),
        TensorSpec("position_ids", [T, 1], torch.int32, init_value=init_position_ids),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("freqs_cos", [T, ROPE_DIM], torch.bfloat16, init_value=init_cos),
        TensorSpec("freqs_sin", [T, ROPE_DIM], torch.bfloat16, init_value=init_sin),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=init_wo_b),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=init_wo_b_scale),
        TensorSpec("attn_out", [T, D], torch.bfloat16, is_output=True),
    ]


# No `__main__` entry: this variant is exercised only through decode_layer.py,
# which owns the comm windows the token shard is built on.
