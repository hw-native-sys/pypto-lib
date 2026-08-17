# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 sparse attention with grouped output projection (decode)."""


import pypto.language as pl

from config import (
    ACTIVE as M,
    DECODE_BATCH,
    DECODE_SEQ,
    BLOCK_SIZE,
    DECODE_CMP_BLOCK_NUM,
    DECODE_ORI_BLOCK_NUM,
    KV_CMP_MAX_BLOCKS,
    KV_ORI_MAX_BLOCKS,
)


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
TOPK_FULL = WIN + IDX_TOPK           # sparse-K columns: window block + indexer topk
CMP_TOPK = IDX_TOPK
SOFTMAX_SCALE = M.softmax_scale
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
O_GROUPS_PER_SPMD = 2

# kernel-local
SUPPORTED_COMPRESS_RATIOS = (0, 4, 128)
DEFAULT_COMPRESS_RATIO = 4
# CSA compressed-slot masking (folded in from the CSA orchestrator): raw indexer
# topk -> per-token bound floor((pos + 1) / COMPRESS_RATIO).
MAX_SEQ_LEN = M.max_position_embeddings
INDEXER_SCORE_LEN = MAX_SEQ_LEN // 4
COMPRESS_RATIO_INV = 1.0 / DEFAULT_COMPRESS_RATIO
CSA_CMP_GE_BIAS = 1.0  # raw + 1, folded for the ge clamp
ORI_MAX_BLOCKS = KV_ORI_MAX_BLOCKS
ORI_BLOCK_NUM = DECODE_ORI_BLOCK_NUM
CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
CMP_BLOCK_NUM = DECODE_CMP_BLOCK_NUM

# tiling
ROPE_OUT_TOK_TILE = 8
H_TILE = 16
# merge_norm gathers with a single-row flat index so the a5 gather lowering's
# per-block row-base chain (tile.ci/divs/muls/add) hits its rows==1 early-out.
# The gather source must be CONTIGUOUS for a flat index to address it correctly,
# so the rope half is produced as its own tile rather than sliced out of n_full.
MERGE_SWAP_FLAT_LEN = H_TILE * ROPE_DIM
ROPE_DIM_F = float(ROPE_DIM)
ROPE_DIM_INV = 1.0 / ROPE_DIM
# qk_pv cube-batch tile (M for the QK/PV matmuls). Batching QK_M_TILE head rows
# per matmul extracts the shared KV tile L1->L0 once per QK_M_TILE/H_TILE
# head-tiles (2x reuse at 32) instead of per H_TILE head-tile, then slices the
# [QK_M_TILE, ...] result back into H_TILE-row stores so the sparse_blk_* layout
# and merge_norm stay unchanged. 32 keeps the [32,128] softmax inside the 192KB
# Vec budget without a cross-core split. 64 is infeasible without further work
# (its [64,128] softmax and co-resident QK+PV L0C accumulators overflow Vec/L0C).
QK_M_TILE = 32
ATTN_K_TILE = 128
# qk_pv dispatch width = the a2a3 AIC (MIX-cluster) count. A runtime pre-pass
# (qk_plan, below) load-balances the T*SPARSE_BLOCKS work items across these lanes,
# replacing the old fixed strided (token, block-lane) NSPLIT split whose imbalance
# grew with per-token variance in the valid-block count. Platform-specific: this is
# the 24-wide dispatch a2a3 targets; re-sweep NUM_QK_CORES for other AIC counts.
NUM_QK_CORES = 24
# MXFP8 output-projection tiles. quant_mx and matmul_mx intentionally live in
# separate tasks and exchange the quantized values/scales through GM.
MX_BLOCK_K = 32
MX_K_TILE = 64
MX_K_SCALE_TILE = MX_K_TILE // MX_BLOCK_K
MX_N_TILE = 256
MM_T_TILE = 16
T_PAD = ((T + MM_T_TILE - 1) // MM_T_TILE) * MM_T_TILE
A_K_TILES = O_GROUP_IN // MX_K_TILE
A_N_TILES = O_LORA // MX_N_TILE
B_GROUP_K_TILES = O_LORA // MX_K_TILE
B_K_TILES = (O_GROUPS * O_LORA) // MX_K_TILE
B_N_TILES = D // MX_N_TILE
WO_A_SCALE_ROWS_PER_GROUP = A_N_TILES * A_K_TILES * MX_K_SCALE_TILE
WO_A_SCALE_ROWS = O_GROUPS * WO_A_SCALE_ROWS_PER_GROUP
WO_B_SCALE_ROWS = B_N_TILES * B_K_TILES * MX_K_SCALE_TILE
NEG_INF = -1.0e20

assert T % 2 == 0
assert T % ROPE_OUT_TOK_TILE == 0  # rope-pack loop tiles tokens by ROPE_OUT_TOK_TILE
assert H % 4 == 0
assert QK_M_TILE % H_TILE == 0
assert H % QK_M_TILE == 0
assert H % O_GROUPS == 0
assert O_GROUP_IN % MX_K_TILE == 0
assert O_LORA % MX_K_TILE == 0 and O_LORA % MX_N_TILE == 0
assert D % MX_N_TILE == 0


def get_standalone_cmp_valid(compress_ratio: int) -> int:
    """Map demo compress-ratio modes to the valid compressed-cache tail length."""
    if compress_ratio == 0:
        return 0
    if compress_ratio == 4:
        return IDX_TOPK
    if compress_ratio == 128:
        return MAX_SEQ_LEN // compress_ratio
    raise ValueError(f"Unsupported compress_ratio={compress_ratio}; expected one of {SUPPORTED_COMPRESS_RATIOS}")


# CSA/full sparse-K width. SWA and HCA use explicit sibling modules so a
# combined decode layer can import all three variants in one Python process
# without relying on import-time config mutation and module-cache order.
TOPK = WIN + CMP_TOPK
# Floor to 2: a single sparse-K block miscompiles in pypto (S-stride cross-token
# output mixup); a 2-block build with an all-invalid 2nd block is bit-exact.
SPARSE_BLOCKS = max(2, (TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE)
PADDED_TOPK = SPARSE_BLOCKS * ATTN_K_TILE
assert WIN <= TOPK <= TOPK_FULL, f"TOPK ({TOPK}) must be in [WIN={WIN}, TOPK_FULL={TOPK_FULL}]"
# qk_pv work items: one per (token, sparse block), load-balanced across NUM_QK_CORES
# lanes by the qk_plan pre-pass (non-empty tiles first, empty tiles appended).
QK_ITEMS = T * SPARSE_BLOCKS


@pl.jit.inline
def sparse_attn(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    window_swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    idx_topk: pl.Tensor[[T, INDEXER_SCORE_LEN], pl.INT32],
    position_ids: pl.Tensor[[T, 1], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[WO_A_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[WO_B_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    attn_out: pl.Tensor[[T, D], pl.BF16],
):
    """Run sparse decode attention, inverse RoPE, and grouped output projection."""
    # Gather the sliding-window + compressed-cache rows. Compressed index contract:
    #   -1              invalid
    #   [0, ...)        compressed KV slots
    ori_kv_flat = pl.reshape(ori_kv, [ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    sparse_bias = pl.create_tensor([T, PADDED_TOPK], dtype=pl.FP32)

    # WAR marker (pypto-lib#481): the fused gather reads ori_kv inside qk_pv, but a
    # scalar-driven gather_row does not by itself mark the param add_inout (and an
    # in-qk_pv self-copy collides with the gather's tensor view). One no-op self-copy
    # marks ori_kv add_inout before qk_pv, so the enclosing layer's in-place KV-cache
    # writeback gets its WAR edge against the gather read. add_inout is a param-level
    # property, so a single tile touch suffices -- no per-token fan-out.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_touch"):
        ori_kv_flat[0:T, 0:HEAD_DIM] = ori_kv_flat[0:T, 0:HEAD_DIM]

    # qk_pv gathers window/compressed rows into one L1 matmul operand. Invalid
    # lanes gather a finite row and are zeroed out by the NEG_INF softmax bias.
    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    o_packed = pl.create_tensor([O_GROUPS * T, O_GROUP_IN], dtype=pl.BF16)
    sparse_blk_mi = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, 1], dtype=pl.FP32)
    sparse_blk_oi = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, HEAD_DIM], dtype=pl.FP32)

    # Load-balanced qk_pv planning (qk_plan): a single scalar task compacts the
    # T*SPARSE_BLOCKS (token, sparse-block) work items into qk_order[] -- non-empty
    # tiles (valid_block_mask > 0) first, empty tiles appended -- via one running
    # write cursor. qk_pv then dispatches NUM_QK_CORES lanes; lane c walks its items
    # strided by NUM_QK_CORES (qk_order[c], qk_order[c + NC], ...). Because the
    # non-empty tiles occupy the front of qk_order, they spread one-per-lane before
    # any lane takes a second -- the heavy tiles balance evenly across cores while the
    # cheap empty tiles fill the tail slots. Replaces the fixed strided (token,
    # block-lane) NSPLIT mapping, whose imbalance grew with per-token variance in the
    # valid-block count. The T/SPARSE_BLOCKS scan loops are trace-time unrolled (small
    # constants) so the cursor read-modify-write is an explicit sequential chain.
    # cmp_sparse_indices holds compressed-cache slots (invalid = -1); valid_block_mask
    # flags non-empty sparse blocks. Both feed qk_pv, so they stay GM scratch here.
    cmp_sparse_indices = pl.create_tensor([T, CMP_TOPK], dtype=pl.INT32)
    valid_block_mask = pl.create_tensor([T, SPARSE_BLOCKS], dtype=pl.INT32)
    qk_order = pl.create_tensor([QK_ITEMS], dtype=pl.INT32)
    qk_wcur = pl.create_tensor([1], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_slots_build_valid_qk_plan", allow_early_resolve=True) as qk_plan_tid:
        # Compressed slots [0, IDX_TOPK): vectorized masked copy over all T rows, keeping
        # raw iff 0 <= raw < floor((pos + 1) / COMPRESS_RATIO), as out = mask*(raw + 1) - 1.
        c_raw = pl.cast(idx_topk[0:T, 0:IDX_TOPK], target_type=pl.FP32)
        c_pos = pl.cast(position_ids[0:T, 0:1], target_type=pl.FP32)
        c_pos_q = pl.cast(pl.cast(pl.mul(pl.add(c_pos, 1.0), COMPRESS_RATIO_INV), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        # Broadcast the per-token bound over IDX_TOPK cols.
        c_upper_b = pl.row_expand_mul(pl.full([T, IDX_TOPK], dtype=pl.FP32, value=1.0), c_pos_q)
        c_ge = pl.minimum(pl.maximum(pl.add(c_raw, CSA_CMP_GE_BIAS), 0.0), 1.0)
        c_lt = pl.minimum(pl.maximum(pl.sub(c_upper_b, c_raw), 0.0), 1.0)
        c_mask = pl.mul(c_ge, c_lt)
        c_out = pl.sub(pl.mul(c_mask, pl.add(c_raw, 1.0)), 1.0)
        cmp_sparse_indices[0:T, 0:IDX_TOPK] = pl.cast(c_out, target_type=pl.INT32)
        # Block 0 (sliding-window) is always live; blocks 1.. from the compressed mask.
        for c_t0 in pl.range(T):
            pl.write(valid_block_mask, [c_t0, 0], pl.cast(1, pl.INT32))
        for c_sb in pl.range(1, SPARSE_BLOCKS):
            c_s0 = (c_sb - 1) * ATTN_K_TILE
            c_blk_valid = pl.row_max(c_mask[:, c_s0 : c_s0 + ATTN_K_TILE])
            for c_dt in pl.range(T):
                c_valid = pl.cast(pl.read(c_blk_valid, [c_dt, 0]), target_type=pl.INT32)
                pl.write(valid_block_mask, [c_dt, c_sb], c_valid)

        # Additive softmax bias (0 valid / NEG_INF invalid) that qk_pv adds onto the
        # scaled scores, so invalid lanes exp to ~0 with no per-block mask multiply.
        v_win_f = pl.cast(window_swa_indices[0:T, 0:WIN], target_type=pl.FP32)
        # Index contract (line 138): raw == -1 invalid, raw >= 0 valid. min(idx, 0)
        # is -1 for invalid / 0 for valid; * -NEG_INF gives NEG_INF / 0. Bit-exact,
        # 2 vector ops instead of the add/max/min/sub clamp chain. c_out is the just-
        # computed post-mask compressed slots (integer-valued), reused directly.
        v_win_valid = pl.minimum(pl.maximum(pl.add(v_win_f, 1.0), 0.0), 1.0)
        sparse_bias[0:T, 0:WIN] = pl.mul(pl.sub(v_win_valid, 1.0), -NEG_INF)
        sparse_bias[0:T, WIN:TOPK] = pl.mul(pl.minimum(c_out, 0.0), -NEG_INF)
        if PADDED_TOPK > TOPK:
            sparse_bias[0:T, TOPK:PADDED_TOPK] = pl.full([T, PADDED_TOPK - TOPK], dtype=pl.FP32, value=NEG_INF)

        pl.write(qk_wcur, [0], pl.cast(0, pl.INT32))
        # Pass 1: non-empty tiles to the front of qk_order.
        for plan_t in pl.unroll(T):
            for plan_sb in pl.unroll(SPARSE_BLOCKS):
                if pl.read(valid_block_mask, [plan_t, plan_sb]) > 0:
                    plan_w = pl.read(qk_wcur, [0])
                    pl.write(qk_order, [plan_w], pl.cast(plan_t * SPARSE_BLOCKS + plan_sb, pl.INT32))
                    pl.write(qk_wcur, [0], pl.cast(plan_w + 1, pl.INT32))
        # Pass 2: empty tiles appended to the tail.
        for plan_t in pl.unroll(T):
            for plan_sb in pl.unroll(SPARSE_BLOCKS):
                if pl.read(valid_block_mask, [plan_t, plan_sb]) <= 0:
                    plan_w = pl.read(qk_wcur, [0])
                    pl.write(qk_order, [plan_w], pl.cast(plan_t * SPARSE_BLOCKS + plan_sb, pl.INT32))
                    pl.write(qk_wcur, [0], pl.cast(plan_w + 1, pl.INT32))

    # One lane per core. Each lane walks its planned items -- the (token, block)
    # work is derived per item, and qk_pv directly gathers window/compressed KV
    # rows into the shared matmul operand.
    with pl.spmd(NUM_QK_CORES, name_hint="qk_pv", deps=[qk_plan_tid], allow_early_resolve=True) as qk_tid:
        qk_core = pl.tile.get_block_idx()
        # Items for this lane: qk_core, qk_core + NUM_QK_CORES, ...  The per-lane
        # count is derived from the lane index (no stored per-core count); a lane
        # with index >= QK_ITEMS runs zero iterations.
        qk_lane_iters = (QK_ITEMS - qk_core + NUM_QK_CORES - 1) // NUM_QK_CORES
        for qk_it in pl.range(qk_lane_iters):
            qk_flat = qk_core + qk_it * NUM_QK_CORES
            qk_item = pl.cast(pl.read(qk_order, [qk_flat]), pl.INDEX)
            qk_t = qk_item // SPARSE_BLOCKS
            qk_sb = qk_item - qk_t * SPARSE_BLOCKS
            qk_b = qk_t // S
            qk_token_base = qk_t * (H // H_TILE) * SPARSE_BLOCKS * H_TILE
            qk_s0 = qk_sb * ATTN_K_TILE
            qk_bias_row = sparse_bias[qk_t : qk_t + 1, qk_s0 : qk_s0 + ATTN_K_TILE]
            qk_block_valid = pl.read(valid_block_mask, [qk_t, qk_sb])
            if qk_block_valid > 0:
                qk_kv = pl.create_l1([ATTN_K_TILE, HEAD_DIM], pl.BF16)
                for qk_r in pl.range(ATTN_K_TILE):
                    qk_k = qk_s0 + qk_r
                    if qk_k < WIN:
                        qk_win_slot_i32 = pl.read(window_swa_indices, [qk_t, qk_k])
                        if qk_win_slot_i32 >= 0:
                            qk_win_slot = pl.cast(qk_win_slot_i32, pl.INDEX)
                            qk_kv = pl.gather_row(qk_kv, ori_kv_flat, [qk_r, 0], [qk_win_slot, 0], [1, HEAD_DIM])
                        else:
                            qk_kv = pl.gather_row(qk_kv, ori_kv_flat, [qk_r, 0], [0, 0], [1, HEAD_DIM])
                    else:
                        qk_cmp_k = qk_k - WIN
                        if qk_cmp_k < CMP_TOPK:
                            qk_ridx = pl.read(cmp_sparse_indices, [qk_t, qk_cmp_k])
                            if qk_ridx >= 0:
                                qk_slot = qk_ridx
                                qk_cblk = pl.cast(pl.read(cmp_block_table, [qk_b, qk_slot // BLOCK_SIZE]), pl.INDEX)
                                qk_csrc = qk_cblk * BLOCK_SIZE + qk_slot % BLOCK_SIZE
                                qk_kv = pl.gather_row(qk_kv, cmp_kv_flat, [qk_r, 0], [qk_csrc, 0], [1, HEAD_DIM])
                            else:
                                qk_kv = pl.gather_row(qk_kv, ori_kv_flat, [qk_r, 0], [0, 0], [1, HEAD_DIM])
                        else:
                            qk_kv = pl.gather_row(qk_kv, ori_kv_flat, [qk_r, 0], [0, 0], [1, HEAD_DIM])

                # Cube-batch QK_M_TILE head rows per QK/PV matmul so the shared KV
                # tile is extracted L1->L0 once per QK_M_TILE/H_TILE head-tiles
                # (2x reuse at QK_M_TILE=32) instead of per head-tile. The
                # [QK_M_TILE, ...] softmax result is sliced back into H_TILE-row
                # stores at the SAME offsets as the per-head-tile path
                # (qk_h_idx == qk_hb * (QK_M_TILE // H_TILE) + qk_sub), so the
                # sparse_blk_* layout and merge_norm are bit-identical.
                for qk_hb in pl.pipeline(H // QK_M_TILE, stage=2):
                    qk_h0 = qk_hb * QK_M_TILE
                    qk_head_row = qk_t * H + qk_h0
                    qk_q_tile = q_flat[qk_head_row : qk_head_row + QK_M_TILE, 0 : HEAD_DIM]
                    qk_raw = pl.matmul(qk_q_tile, qk_kv, b_trans=True, out_dtype=pl.FP32)
                    qk_scaled = pl.mul(qk_raw, SOFTMAX_SCALE)
                    # Broadcast-add the per-block bias directly (col_expand_add) instead
                    # of col_expand into a dead pl.full(0) base + a separate add.
                    qk_scores = pl.col_expand_add(qk_scaled, qk_bias_row)
                    qk_mi = pl.row_max(qk_scores)
                    # Invalid lanes (NEG_INF bias, zero kv rows) exp to ~0; all-invalid
                    # blocks die in the merge alpha/beta -- no mask multiply needed.
                    qk_exp = pl.exp(pl.row_expand_sub(qk_scores, qk_mi))
                    qk_li = pl.row_sum(qk_exp)
                    qk_exp_bf16 = pl.cast(qk_exp, target_type=pl.BF16, mode="rint")
                    qk_oi = pl.matmul(qk_exp_bf16, qk_kv, out_dtype=pl.FP32)
                    for qk_sub in pl.unroll(QK_M_TILE // H_TILE):
                        qk_h_idx = qk_hb * (QK_M_TILE // H_TILE) + qk_sub
                        qk_r0 = qk_sub * H_TILE
                        qk_blk_base = qk_token_base + qk_h_idx * SPARSE_BLOCKS * H_TILE
                        qk_row = qk_blk_base + qk_sb * H_TILE
                        sparse_blk_mi[qk_row : qk_row + H_TILE, 0 : 1] = qk_mi[qk_r0 : qk_r0 + H_TILE, 0 : 1]
                        sparse_blk_li[qk_row : qk_row + H_TILE, 0 : 1] = qk_li[qk_r0 : qk_r0 + H_TILE, 0 : 1]
                        sparse_blk_oi[qk_row : qk_row + H_TILE, 0 : HEAD_DIM] = qk_oi[qk_r0 : qk_r0 + H_TILE, 0 : HEAD_DIM]
            else:
                qk_oi_zero = pl.full([H_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
                for qk_h_idx in pl.range(H // H_TILE):
                    qk_blk_base = qk_token_base + qk_h_idx * SPARSE_BLOCKS * H_TILE
                    qk_row = qk_blk_base + qk_sb * H_TILE
                    for qk_hr in pl.range(H_TILE):
                        pl.write(sparse_blk_mi, [qk_row + qk_hr, 0], -3.0e38)
                        pl.write(sparse_blk_li, [qk_row + qk_hr, 0], 0.0)
                    sparse_blk_oi[qk_row : qk_row + H_TILE, 0 : HEAD_DIM] = qk_oi_zero

    # Precompute the head-invariant interleaved cos and sign*sin once: they depend
    # only on (token, column), not head, so building them per head would repeat the
    # same dup-gather H times on the bottleneck Vec engine. sign is folded into sin
    # (multiply by +/-1). The conjugate (inverse) rotation is:
    #   out[j] = x[j]*cos_il[j] + x[j^1]*sign[j]*sin_il[j]
    # Hoisted ABOVE merge_norm (which now fuses the rotation): independent of qk_pv,
    # so it overlaps it and is off merge_norm's critical path.
    rope_cos_il = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    # The j^1 lane-swap index is block-invariant too; merge_norm rebuilt the whole
    # arange/trunc/lane chain on each of its T*(H//H_TILE) blocks. Build it here.
    rope_swap_idx_m = pl.create_tensor([1, MERGE_SWAP_FLAT_LEN], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_cs", allow_early_resolve=True):
        sw_col_m = pl.cast(
            pl.arange(0, [1, MERGE_SWAP_FLAT_LEN], dtype=pl.INT32), target_type=pl.FP32)
        sw_rowbase_m = pl.mul(
            pl.cast(pl.cast(pl.mul(sw_col_m, ROPE_DIM_INV), target_type=pl.INT32, mode="trunc"),
                    target_type=pl.FP32),
            ROPE_DIM_F)                                                                           # r*ROPE_DIM
        sw_j_m = pl.sub(sw_col_m, sw_rowbase_m)                                                   # j
        sw_dup_f_m = pl.cast(pl.cast(pl.mul(sw_j_m, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        sw_lane_m = pl.sub(sw_j_m, pl.mul(sw_dup_f_m, 2.0))                                       # j%2
        sw_jx_m = pl.sub(pl.add(sw_j_m, 1.0), pl.mul(sw_lane_m, 2.0))                              # j^1
        rope_swap_idx_m[0:1, 0:MERGE_SWAP_FLAT_LEN] = pl.cast(
            pl.add(sw_rowbase_m, sw_jx_m), target_type=pl.INT32)
        cs_col = pl.col_expand_mul(
            pl.full([T, ROPE_DIM], dtype=pl.FP32, value=1.0),
            pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32))
        cs_dup_f = pl.cast(pl.cast(pl.mul(cs_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        cs_dup_idx = pl.cast(cs_dup_f, target_type=pl.INT32)                                      # j>>1
        cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))                                           # j%2
        cs_sign = pl.neg(pl.sub(pl.mul(cs_lane, 2.0), 1.0))                                       # [+1,-1,...] (conjugate)
        cs_cos = pl.cast(freqs_cos[0:T, 0:HALF_ROPE], target_type=pl.FP32)
        cs_sin = pl.cast(freqs_sin[0:T, 0:HALF_ROPE], target_type=pl.FP32)
        rope_cos_il[0:T, 0:ROPE_DIM] = pl.gather(cs_cos, dim=-1, index=cs_dup_idx)
        rope_sin_signed[0:T, 0:ROPE_DIM] = pl.mul(pl.gather(cs_sin, dim=-1, index=cs_dup_idx), cs_sign)

    # Online-softmax merge across sparse-K tiles, sink-norm, then fused inverse RoPE.
    # One spmd block per (token, head-tile) -- T*(H//H_TILE) blocks -- so the merge
    # fans out over that many AIVs instead of T blocks each running a serial head-tile
    # loop. The inverse-RoPE rotation + rope-column pack is fused in (was a separate
    # "rope" spmd reading an attn_rope_stage GM round-trip): the head-tile's fp32 rope
    # segment is rotated in UB and packed straight into o_packed's rope columns.
    # The MXFP8 projection below reads o_packed through ordinary GM dependencies.
    with pl.spmd(T * (H // H_TILE), name_hint="merge_norm") as _merge_tid:
        m_idx = pl.tile.get_block_idx()
        m_t = m_idx // (H // H_TILE)
        m_h_idx = m_idx - m_t * (H // H_TILE)
        m_h0 = m_h_idx * H_TILE
        m_blk_base = m_idx * SPARSE_BLOCKS * H_TILE
        m_mi = sparse_blk_mi[m_blk_base : m_blk_base + H_TILE, 0 : 1]
        m_li = sparse_blk_li[m_blk_base : m_blk_base + H_TILE, 0 : 1]
        m_oi = sparse_blk_oi[m_blk_base : m_blk_base + H_TILE, 0 : HEAD_DIM]

        if SPARSE_BLOCKS > 1:
            for m_sb in pl.pipeline(1, SPARSE_BLOCKS, stage=2):
                m_row = m_blk_base + m_sb * H_TILE
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
        n_nope = pl.row_expand_div(m_oi[0 : H_TILE, 0 : NOPE_DIM], n_denom)
        n_nope_bf16 = pl.cast(n_nope, target_type=pl.BF16, mode="rint")

        m_swap_flat = rope_swap_idx_m[0:1, 0:MERGE_SWAP_FLAT_LEN]
        m_rope = pl.row_expand_div(m_oi[0 : H_TILE, NOPE_DIM : HEAD_DIM], n_denom)
        m_cos_il = rope_cos_il[m_t : m_t + 1, 0 : ROPE_DIM]
        m_sin_signed = rope_sin_signed[m_t : m_t + 1, 0 : ROPE_DIM]
        m_rope_flat = pl.reshape(m_rope, [1, MERGE_SWAP_FLAT_LEN])
        m_swapped = pl.reshape(pl.gather(m_rope_flat, dim=-1, index=m_swap_flat), [H_TILE, ROPE_DIM])
        m_rot = pl.add(pl.col_expand_mul(m_rope, m_cos_il), pl.col_expand_mul(m_swapped, m_sin_signed))
        n_rope_bf16 = pl.cast(m_rot, target_type=pl.BF16, mode="rint")
        n_full_bf16 = pl.concat(n_nope_bf16, n_rope_bf16)

        for n_hi in pl.range(H_TILE):
            n_gh = m_h0 + n_hi
            n_g = n_gh // HEADS_PER_GROUP
            n_hh = n_gh - n_g * HEADS_PER_GROUP
            n_pack_row = n_g * T + m_t
            n_col = n_hh * HEAD_DIM
            o_packed[n_pack_row : n_pack_row + 1, n_col : n_col + HEAD_DIM] = n_full_bf16[n_hi : n_hi + 1, 0 : HEAD_DIM]

    # ========================================================================
    # Grouped MXFP8 output projection. Each quant stage writes values and E8M0
    # scales to GM; the following matmul_mx stage reads that materialized form.
    # K=64 partials are reduced in FP32 between the two projections.
    # ========================================================================
    wo_a_flat = pl.reshape(wo_a, [O_GROUPS * O_GROUP_IN, O_LORA])
    o_a_mx = pl.create_tensor([O_GROUPS * T_PAD, O_GROUP_IN], dtype=pl.FP8E4M3FN)
    o_a_scale_store = pl.create_tensor(
        [O_GROUPS * A_K_TILES, MM_T_TILE * MX_K_SCALE_TILE], dtype=pl.FP8E8M0
    )
    proj_a_partial = pl.create_tensor(
        [O_GROUPS * A_K_TILES * MM_T_TILE, O_LORA], dtype=pl.FP32
    )
    o_r_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.FP32)
    o_b_mx = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.FP8E4M3FN)
    o_b_scale_store = pl.create_tensor(
        [O_GROUPS * B_GROUP_K_TILES, MM_T_TILE * MX_K_SCALE_TILE], dtype=pl.FP8E8M0
    )
    proj_b_partial = pl.create_tensor(
        [O_GROUPS * B_GROUP_K_TILES * MM_T_TILE, D], dtype=pl.FP32
    )
    proj_b_group = pl.create_tensor([T_PAD, O_GROUPS * D], dtype=pl.FP32)

    for quant_idx in pl.parallel(O_GROUPS * A_K_TILES):
        g = quant_idx // A_K_TILES
        kb = quant_idx - g * A_K_TILES
        k0 = kb * MX_K_TILE
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_a_mx_quant"):
            a_src = pl.load(
                o_packed,
                [g * T, k0],
                [MM_T_TILE, MX_K_TILE],
                valid_shape=[T, MX_K_TILE],
                target_memory=pl.Mem.Vec,
            )
            a_src = pl.fillpad(a_src, pad_value=pl.PadValue.zero)
            a_q, a_scale = pl.quant_mx(a_src, layout=pl.MX_A_ZZ)
            pl.store(a_q, [g * T_PAD, k0], o_a_mx)
            a_scale_flat = pl.reshape(a_scale, [1, MM_T_TILE * MX_K_SCALE_TILE])
            pl.store(a_scale_flat, [quant_idx, 0], o_a_scale_store)

    for task_idx in pl.parallel(O_GROUPS * A_K_TILES * A_N_TILES):
        g = task_idx // (A_K_TILES * A_N_TILES)
        local_idx = task_idx % (A_K_TILES * A_N_TILES)
        kb = local_idx // A_N_TILES
        nb = local_idx - kb * A_N_TILES
        k0 = kb * MX_K_TILE
        n0 = nb * MX_N_TILE
        a_scale_idx = g * A_K_TILES + kb
        a_scale_slice = o_a_scale_store[a_scale_idx : a_scale_idx + 1, :]
        a_scale_mx = pl.tensor.view(
            a_scale_slice, [MM_T_TILE, MX_K_SCALE_TILE], layout=pl.MX_A_ZZ
        )
        wa_scale_offset = (
            g * WO_A_SCALE_ROWS_PER_GROUP
            + (nb * A_K_TILES + kb) * MX_K_SCALE_TILE
        )
        wa_scale_slice = wo_a_scale[
            wa_scale_offset : wa_scale_offset + MX_K_SCALE_TILE, :
        ]
        wa_scale_mx = pl.tensor.view(
            wa_scale_slice, [MX_K_SCALE_TILE, MX_N_TILE], layout=pl.MX_B_NN
        )
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_a_mx_mm"):
            a_k = pl.move(
                pl.load(o_a_mx, [g * T_PAD, k0], [MM_T_TILE, MX_K_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.Left,
            )
            a_scale_k = pl.move(
                pl.load(a_scale_mx, [0, 0], [MM_T_TILE, MX_K_SCALE_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.LeftScale,
            )
            wa_k = pl.move(
                pl.load(
                    wo_a_flat,
                    [g * O_GROUP_IN + k0, n0],
                    [MX_K_TILE, MX_N_TILE],
                    target_memory=pl.Mem.Mat,
                ),
                target_memory=pl.Mem.Right,
            )
            wa_scale_k = pl.move(
                pl.load(wa_scale_mx, [0, 0], [MX_K_SCALE_TILE, MX_N_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.RightScale,
            )
            a_partial = pl.matmul_mx(a_k, a_scale_k, wa_k, wa_scale_k)
            pl.store(
                a_partial,
                [(g * A_K_TILES + kb) * MM_T_TILE, n0],
                proj_a_partial,
            )

    for task_idx in pl.parallel(O_GROUPS * A_N_TILES):
        g = task_idx // A_N_TILES
        nb = task_idx - g * A_N_TILES
        n0 = nb * MX_N_TILE
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_a_mx_reduce"):
            a_sum = pl.tile.full([MM_T_TILE, MX_N_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.pipeline(A_K_TILES, stage=2):
                a_part = pl.load(
                    proj_a_partial,
                    [(g * A_K_TILES + kb) * MM_T_TILE, n0],
                    [MM_T_TILE, MX_N_TILE],
                    target_memory=pl.Mem.Vec,
                )
                a_sum = pl.add(a_sum, a_part)
            pl.store(a_sum, [0, g * O_LORA + n0], o_r_pad)

    for quant_idx in pl.parallel(O_GROUPS * B_GROUP_K_TILES):
        g = quant_idx // B_GROUP_K_TILES
        kb = quant_idx - g * B_GROUP_K_TILES
        k0 = kb * MX_K_TILE
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_b_mx_quant"):
            b_src = pl.load(
                o_r_pad,
                [0, g * O_LORA + k0],
                [MM_T_TILE, MX_K_TILE],
                target_memory=pl.Mem.Vec,
            )
            b_q, b_scale = pl.quant_mx(b_src, layout=pl.MX_A_ZZ)
            pl.store(b_q, [0, g * O_LORA + k0], o_b_mx)
            b_scale_flat = pl.reshape(b_scale, [1, MM_T_TILE * MX_K_SCALE_TILE])
            pl.store(b_scale_flat, [quant_idx, 0], o_b_scale_store)

    for task_idx in pl.parallel(O_GROUPS * B_GROUP_K_TILES * B_N_TILES):
        g = task_idx // (B_GROUP_K_TILES * B_N_TILES)
        local_idx = task_idx % (B_GROUP_K_TILES * B_N_TILES)
        kb = local_idx // B_N_TILES
        nb = local_idx - kb * B_N_TILES
        global_kb = g * B_GROUP_K_TILES + kb
        n0 = nb * MX_N_TILE
        b_scale_slice = o_b_scale_store[global_kb : global_kb + 1, :]
        b_scale_mx = pl.tensor.view(
            b_scale_slice, [MM_T_TILE, MX_K_SCALE_TILE], layout=pl.MX_A_ZZ
        )
        wb_scale_offset = (nb * B_K_TILES + global_kb) * MX_K_SCALE_TILE
        wb_scale_slice = wo_b_scale[
            wb_scale_offset : wb_scale_offset + MX_K_SCALE_TILE, :
        ]
        wb_scale_mx = pl.tensor.view(
            wb_scale_slice, [MX_K_SCALE_TILE, MX_N_TILE], layout=pl.MX_B_NN
        )
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_b_mx_mm"):
            b_k = pl.move(
                pl.load(
                    o_b_mx,
                    [0, g * O_LORA + kb * MX_K_TILE],
                    [MM_T_TILE, MX_K_TILE],
                    target_memory=pl.Mem.Mat,
                ),
                target_memory=pl.Mem.Left,
            )
            b_scale_k = pl.move(
                pl.load(b_scale_mx, [0, 0], [MM_T_TILE, MX_K_SCALE_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.LeftScale,
            )
            wb_k = pl.move(
                pl.load(
                    wo_b,
                    [global_kb * MX_K_TILE, n0],
                    [MX_K_TILE, MX_N_TILE],
                    target_memory=pl.Mem.Mat,
                ),
                target_memory=pl.Mem.Right,
            )
            wb_scale_k = pl.move(
                pl.load(wb_scale_mx, [0, 0], [MX_K_SCALE_TILE, MX_N_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.RightScale,
            )
            b_partial = pl.matmul_mx(b_k, b_scale_k, wb_k, wb_scale_k)
            pl.store(
                b_partial,
                [global_kb * MM_T_TILE, n0],
                proj_b_partial,
            )

    for task_idx in pl.parallel(O_GROUPS * B_N_TILES):
        g = task_idx // B_N_TILES
        nb = task_idx - g * B_N_TILES
        n0 = nb * MX_N_TILE
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_b_mx_reduce"):
            b_sum = pl.tile.full([MM_T_TILE, MX_N_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.pipeline(B_GROUP_K_TILES, stage=2):
                b_part = pl.load(
                    proj_b_partial,
                    [(g * B_GROUP_K_TILES + kb) * MM_T_TILE, n0],
                    [MM_T_TILE, MX_N_TILE],
                    target_memory=pl.Mem.Vec,
                )
                b_sum = pl.add(b_sum, b_part)
            pl.store(b_sum, [0, g * D + n0], proj_b_group)

    for nb in pl.parallel(B_N_TILES):
        n0 = nb * MX_N_TILE
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_b_group_reduce"):
            out_sum = pl.tile.full([MM_T_TILE, MX_N_TILE], dtype=pl.FP32, value=0.0)
            for g in pl.pipeline(O_GROUPS, stage=2):
                group_part = pl.load(
                    proj_b_group,
                    [0, g * D + n0],
                    [MM_T_TILE, MX_N_TILE],
                    target_memory=pl.Mem.Vec,
                )
                out_sum = pl.add(out_sum, group_part)
            out_valid = pl.set_validshape(out_sum, T, MX_N_TILE)
            out_bf16 = pl.cast(out_valid, target_type=pl.BF16, mode="rint")
            pl.store(out_bf16, [0, n0], attn_out)

    return attn_out

@pl.jit
def sparse_attn_test(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    window_swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    idx_topk: pl.Tensor[[T, INDEXER_SCORE_LEN], pl.INT32],
    position_ids: pl.Tensor[[T, 1], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[WO_A_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[WO_B_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    sparse_attn(
        q,
        ori_kv,
        window_swa_indices,
        cmp_kv,
        cmp_block_table,
        idx_topk,
        position_ids,
        attn_sink,
        freqs_cos,
        freqs_sin,
        wo_a,
        wo_a_scale,
        wo_b,
        wo_b_scale,
        attn_out,
    )
    return attn_out


def golden_sparse_attn(tensors):
    """Torch reference: sparse_attn decode path followed by grouped o_proj."""
    import torch

    q = tensors["q"].float()
    ori_kv = tensors["ori_kv"].float()
    window_swa_indices = tensors["window_swa_indices"]
    cmp_kv = tensors["cmp_kv"].float()
    cmp_block_table = tensors["cmp_block_table"]
    # Compressed slots: keep raw indexer topk iff 0 <= raw < floor((pos + 1) /
    # COMPRESS_RATIO), else -1 -- the masking sparse_attn now folds in internally.
    raw = tensors["idx_topk"][:, :CMP_TOPK].to(torch.int64)
    bound = ((tensors["position_ids"][:, 0].to(torch.int64) + 1) // DEFAULT_COMPRESS_RATIO).unsqueeze(1)
    keep = (raw >= 0) & (raw < bound)
    cmp_sparse_indices = torch.where(keep, raw, torch.full_like(raw, -1)).to(torch.int32)
    attn_sink = tensors["attn_sink"].float()
    cos = tensors["freqs_cos"].float()
    sin = tensors["freqs_sin"].float()
    from expert_shared import _dynamic_mxfp8_matmul, _unpack_b_scale_tiled

    wo_a = tensors["wo_a"]
    wo_a_scale_packed = tensors["wo_a_scale"]
    wo_b = tensors["wo_b"]
    wo_b_scale = _unpack_b_scale_tiled(
        tensors["wo_b_scale"], O_GROUPS * O_LORA, D
    )

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
            blk_id = int(cmp_block_table[b, cmp_slot // BLOCK_SIZE].item())
            intra = cmp_slot % BLOCK_SIZE
            kv_rows.append(cmp_kv[blk_id, intra, 0])
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
    o_model = o.float().view(T, O_GROUPS, O_GROUP_IN)
    o_r_g = torch.empty(T, O_GROUPS, O_LORA, dtype=torch.float32)
    out = torch.zeros(T, D, dtype=torch.float32)
    for g in range(O_GROUPS):
        wa_scale0 = g * WO_A_SCALE_ROWS_PER_GROUP
        wa_scale = _unpack_b_scale_tiled(
            wo_a_scale_packed[
                wa_scale0 : wa_scale0 + WO_A_SCALE_ROWS_PER_GROUP
            ],
            O_GROUP_IN,
            O_LORA,
        )
        o_r_g[:, g] = _dynamic_mxfp8_matmul(o_model[:, g], wo_a[g], wa_scale)
        k0 = g * O_LORA
        out += _dynamic_mxfp8_matmul(
            o_r_g[:, g],
            wo_b[k0 : k0 + O_LORA],
            wo_b_scale[k0 // MX_BLOCK_K : (k0 + O_LORA) // MX_BLOCK_K],
        )

    tensors["attn_out"][:] = out.to(torch.bfloat16)

def build_tensor_specs(
    compress_ratio: int = DEFAULT_COMPRESS_RATIO,
    causal_regression_fixture: bool = False,
    short_window_fixture: bool = False,
    mixed_topk_fixture: bool = False,
    cache_window_replacement_fixture: bool = False,
):
    """Build deterministic demo tensors for the merged standalone harness."""
    import torch
    from decode_metadata import block_table
    from golden import TensorSpec
    from rope_tables import build_deepseek_v4_rope_tables, materialize_token_rope_tables

    cmp_valid = get_standalone_cmp_valid(compress_ratio)
    shared_freqs_cos, shared_freqs_sin = build_deepseek_v4_rope_tables(M, compress_ratio, dtype=torch.bfloat16)
    shared_rope_cos, shared_rope_sin = materialize_token_rope_tables(
        shared_freqs_cos,
        shared_freqs_sin,
        torch.arange(T, dtype=torch.int32),
    )

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
        """Build physical cache-row indices for standalone window raw slots."""
        tbl = init_window_block_table()
        indices = torch.full((T, WIN), -1, dtype=torch.int32)
        for t in range(T):
            b = t // S
            for raw in range(WIN):
                blk = int(tbl[b, raw // BLOCK_SIZE].item())
                if blk >= 0:
                    indices[t, raw] = blk * BLOCK_SIZE + raw % BLOCK_SIZE
        return indices

    def init_cmp_kv():
        """Initialize the compressed-cache KV pages."""
        return torch.rand(CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM) - 0.5

    def init_attn_sink():
        """Initialize the per-head sink logits to zero."""
        return torch.zeros(H)

    def init_window_block_table():
        """Build the demo block table for the sliding-window cache pages."""
        return block_table(
            batch=B,
            table_blocks=ORI_MAX_BLOCKS,
            physical_blocks=ORI_BLOCK_NUM,
        )

    def init_cmp_block_table():
        """Build the demo block table for the compressed-cache pages."""
        return block_table(
            batch=B,
            table_blocks=CMP_MAX_BLOCKS,
            physical_blocks=CMP_BLOCK_NUM,
        )

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
        return torch.full((T, 1), DEFAULT_COMPRESS_RATIO * CMP_TOPK, dtype=torch.int32)

    def init_cos():
        """Build the split-half cosine table used by the inverse-RoPE reference."""
        return shared_rope_cos.clone()

    def init_sin():
        """Build the split-half sine table used by the inverse-RoPE reference."""
        return shared_rope_sin.clone()

    from expert_shared import _gen_mxfp8_weight_kn

    wo_a_values = []
    wo_a_scales = []
    for _ in range(O_GROUPS):
        wa_value, wa_scale = _gen_mxfp8_weight_kn(
            (O_GROUP_IN, O_LORA),
            dequant_std=0.25 / O_GROUP_IN ** 0.5,
            chan_cv=0.25,
        )
        wo_a_values.append(wa_value)
        wo_a_scales.append(wa_scale)
    wo_a_fp8 = torch.stack(wo_a_values)
    wo_a_scale = torch.cat(wo_a_scales, dim=0)
    wo_b_fp8, wo_b_scale = _gen_mxfp8_weight_kn(
        (O_GROUPS * O_LORA, D),
        dequant_std=0.25 / (O_GROUPS * O_LORA) ** 0.5,
        chan_cv=0.25,
    )

    def init_wo_a():
        """Initialize grouped MXFP8 first-stage output-projection weights."""
        return wo_a_fp8

    def init_wo_a_scale():
        """Initialize tiled E8M0 scales for the grouped first-stage weights."""
        return wo_a_scale

    def init_wo_b():
        """Initialize the second-stage MXFP8 output-projection weights."""
        return wo_b_fp8

    def init_wo_b_scale():
        """Initialize tiled E8M0 scales for the second-stage weights."""
        return wo_b_scale

    return [
        TensorSpec("q", [T, H, HEAD_DIM], torch.bfloat16, init_value=init_q),
        TensorSpec("ori_kv", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_ori_kv),
        TensorSpec("window_swa_indices", [T, WIN], torch.int32, init_value=init_window_swa_indices),
        TensorSpec("cmp_kv", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table", [B, CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("idx_topk", [T, INDEXER_SCORE_LEN], torch.int32, init_value=init_idx_topk),
        TensorSpec("position_ids", [T, 1], torch.int32, init_value=init_position_ids),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("freqs_cos", [T, ROPE_DIM], torch.bfloat16, init_value=init_cos),
        TensorSpec("freqs_sin", [T, ROPE_DIM], torch.bfloat16, init_value=init_sin),
        TensorSpec("wo_a", [O_GROUPS, O_GROUP_IN, O_LORA], torch.float8_e4m3fn, init_value=init_wo_a),
        TensorSpec("wo_a_scale", [WO_A_SCALE_ROWS, MX_N_TILE], torch.float8_e8m0fnu, init_value=init_wo_a_scale),
        TensorSpec("wo_b", [O_GROUPS * O_LORA, D], torch.float8_e4m3fn, init_value=init_wo_b),
        TensorSpec("wo_b_scale", [WO_B_SCALE_ROWS, MX_N_TILE], torch.float8_e8m0fnu, init_value=init_wo_b_scale),
        TensorSpec("attn_out", [T, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a5",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    # --compress-ratio only selects which compressed-tail data pattern to validate;
    # the pruned widths are covered by the swa/hca variant tests.
    parser.add_argument("--compress-ratio", type=int, default=DEFAULT_COMPRESS_RATIO,
                        choices=list(SUPPORTED_COMPRESS_RATIOS))
    parser.add_argument("--causal-regression-fixture", action="store_true", default=False,
                        help="Amplify the S=2 future-window-slot regression; use with --compress-ratio 0.")
    parser.add_argument("--short-window-fixture", action="store_true", default=False,
                        help="Use a short-window topk row with valid prefix + -1 padding.")
    parser.add_argument("--mixed-topk-fixture", action="store_true", default=False,
                        help="Use -1-padded window slots with valid compressed raw indices.")
    parser.add_argument("--cache-window-replacement-fixture", action="store_true", default=False,
                        help="Place a sentinel row inside the cache window prefix.")
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2))
    parser.add_argument("--enable-dep-gen", action="store_true", default=False,
                        help="Capture PTO2 dependency edges (deps.json); the swimlane "
                             "converter draws fanout/fanin arrows from the sibling file.")
    parser.add_argument("--enable-pmu", nargs="?", const=2, default=0, type=int, choices=[0, 1, 2, 4])
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    compress_ratio = args.compress_ratio
    print(f"compress_ratio={compress_ratio} "
          f"-> TOPK={TOPK} SPARSE_BLOCKS={SPARSE_BLOCKS} PADDED_TOPK={PADDED_TOPK}", flush=True)

    result = run_jit(
        fn=sparse_attn_test,
        specs=build_tensor_specs(
            compress_ratio,
            args.causal_regression_fixture,
            args.short_window_fixture,
            args.mixed_topk_fixture,
            args.cache_window_replacement_fixture,
        ),
        golden_fn=golden_sparse_attn,
        golden_data=args.golden_data,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
            enable_dep_gen=args.enable_dep_gen,
            enable_pmu=args.enable_pmu,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "attn_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
