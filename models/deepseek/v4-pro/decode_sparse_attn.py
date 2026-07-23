# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 sparse attention with grouped MXFP8 output projection (decode).

FA / RoPE / merge stay BF16 (存8算16): ``cmp_kv`` is C8 (e4m3 + group-64 FP32
scale interim); window ``ori_kv`` remains BF16 until MLAProlog writes C8.
MLAEpilog ``o_a_proj`` / ``o_b_proj`` are Hybrid MXFP8 W8A8 (e4m3 + e8m0, block=32).
"""


import pypto.language as pl

from config import (
    FLASH as M,
    DECODE_BATCH,
    DECODE_SEQ,
    BLOCK_SIZE,
    DECODE_CMP_BLOCK_NUM,
    DECODE_ORI_BLOCK_NUM,
    KV_CMP_MAX_BLOCKS,
    KV_ORI_MAX_BLOCKS,
    MX_BLOCK_K,
)
from kv_c8_common import KV_SCALE_COLS
from mx_quant_common import (
    ATOL_RTOL,
    MX_KV_GROUP,
    dequant_kv_c8_fp32_scale,
    dynamic_mx_quant_e4m3,
    gen_mxfp8_weight_kn,
    golden_kv_c8_quant_row_fp32_scale,
    mx_matmul_fp8,
    unpack_scale_b_nn,
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
O_GROUP_IN_SCALE = O_GROUP_IN // MX_BLOCK_K
O_LORA_TOTAL = O_GROUPS * O_LORA
O_LORA_TOTAL_SCALE = O_LORA_TOTAL // MX_BLOCK_K

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
# proj_a cube K-frag. 256 (not 128) keeps the B-cache-line floor: B is K-contiguous
# under b_trans, so K*2B(bf16) = 512B == the a2a3 L2 line (K=128 was 256B, half a
# line -> wasted MTE2 DMA). At 256 the cube's L0A/L0B operand staging hits 100%
# (the wall); 512 would spill it for no gain (swept: K=512 net-negative).
A_K_TILE = 256
# proj_a is a pure-cube MX matmul scope (proj_a_mm) writing the fp32 GM intermediate
# o_r_pad (cf. expert_shared gate/up decouple), consumed directly by proj_b_mm.
PROJ_A_MM_N_TILE = 128   # cube N frag; Right FP8 tile = 256*128 = 32KB (< 64KB wall).
MM_T_TILE = 16
T_PAD = ((T + MM_T_TILE - 1) // MM_T_TILE) * MM_T_TILE
B_K_TILE = 256  # proj_b_mm cube K frag; 256*256 FP8 Right tile = 64KB (MX wall).
# proj_b is decoupled into a pure-cube MX GEMM scope (proj_b_mm) and a pure-vector
# group-sum scope (proj_b_act) meeting through grouped FP32 partials in GM.
PROJ_B_MM_N_TILE = 256    # cube N frag; Acc = 16*256*4 = 16KB FP32.
PROJ_B_ACT_N_TILE = 512   # vector N frag for the O_GROUPS-way FP32 accumulate+cast.
PA_NFRAGS = O_LORA // PROJ_A_MM_N_TILE   # proj_a cube N-frags per group
# so the per-group split does not multiply the task count by N-frags. A 512-column
# chunk produces 8 * (4096 / 512) = 64 balanced cube blocks.
PROJ_B_D_CHUNK = 512
PB_DCHUNKS = D // PROJ_B_D_CHUNK
# proj_b_act uses one block per 512-column output region, eight blocks in total.
PROJ_B_ACT_T_TILE = 8    # inner token tile for proj_b_act O_GROUPS-way FP32 accumulate
PROJ_B_ACT_TBLK = 8      # proj_b_act token block per task
PB_ACT_NREG = D // PROJ_B_ACT_N_TILE
PB_ACT_TBLKS = T // PROJ_B_ACT_TBLK
NEG_INF = -1.0e20

assert T % 2 == 0
assert T % ROPE_OUT_TOK_TILE == 0  # rope-pack loop tiles tokens by ROPE_OUT_TOK_TILE
assert H % 4 == 0
assert QK_M_TILE % H_TILE == 0
assert H % QK_M_TILE == 0
assert H % O_GROUPS == 0
assert O_GROUP_IN % MX_BLOCK_K == 0
assert O_LORA % MX_BLOCK_K == 0
assert A_K_TILE % MX_BLOCK_K == 0
assert B_K_TILE % MX_BLOCK_K == 0
assert (O_GROUPS * O_LORA) % B_K_TILE == 0
assert D % PROJ_B_MM_N_TILE == 0, "proj_b_mm cube N-loop must cover D"
assert D % PROJ_B_D_CHUNK == 0, "proj_b D-chunk loop must cover D"
assert PROJ_B_D_CHUNK % PROJ_B_MM_N_TILE == 0, "proj_b inner N-frag loop must cover the D-chunk"
assert T % PROJ_B_ACT_TBLK == 0 and PROJ_B_ACT_TBLK % PROJ_B_ACT_T_TILE == 0
assert D % PROJ_B_ACT_N_TILE == 0, "proj_b_act vector N-loop must cover D"
assert O_LORA % B_K_TILE == 0, "proj_b group K-loop covers O_LORA in B_K_TILE iters"


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
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.FP8E4M3FN],
    cmp_kv_scale: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, KV_SCALE_COLS], pl.FP32],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    idx_topk: pl.Tensor[[T, INDEXER_SCORE_LEN], pl.INT32],
    position_ids: pl.Tensor[[T, 1], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[O_GROUPS, O_GROUP_IN_SCALE, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[O_LORA_TOTAL, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[O_LORA_TOTAL_SCALE, D], pl.FP8E8M0],
    attn_out: pl.Tensor[[T, D], pl.BF16],
):
    """Run sparse decode attention, inverse RoPE, and grouped output projection."""
    # Gather the sliding-window + compressed-cache rows. Compressed index contract:
    #   -1              invalid
    #   [0, ...)        compressed KV slots
    ori_kv_flat = pl.reshape(ori_kv, [ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    cmp_kv_scale_flat = pl.reshape(cmp_kv_scale, [CMP_BLOCK_NUM * BLOCK_SIZE, KV_SCALE_COLS])
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
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_slots_build_valid_qk_plan") as qk_plan_tid:
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

    # Gather SWA (BF16 ori_kv) + compressed C8 (e4m3×FP32 scale → BF16) into a dense
    # sparse-K workspace. qk_pv then only loads BF16 tiles (存8算16), avoiding FP8
    # ops inside the cube MIX scope (L1 FP8 assemble hits blayout walls on a5).
    csa_kv_flat = pl.create_tensor([T * PADDED_TOPK, HEAD_DIM], dtype=pl.BF16)
    with pl.spmd(T * SPARSE_BLOCKS, name_hint="gather_kv", deps=[qk_plan_tid]) as gather_tid:
        g_item = pl.tile.get_block_idx()
        g_t = g_item // SPARSE_BLOCKS
        g_sb = g_item - g_t * SPARSE_BLOCKS
        g_b = g_t // S
        g_s0 = g_sb * ATTN_K_TILE
        g_dst0 = g_t * PADDED_TOPK + g_s0
        if g_sb == 0:
            for g_r in pl.range(ATTN_K_TILE):
                g_k = g_s0 + g_r
                if g_k < WIN:
                    g_win_slot_i32 = pl.read(window_swa_indices, [g_t, g_k])
                    if g_win_slot_i32 >= 0:
                        g_win_slot = pl.cast(g_win_slot_i32, pl.INDEX)
                        csa_kv_flat[g_dst0 + g_r : g_dst0 + g_r + 1, 0:HEAD_DIM] = ori_kv_flat[
                            g_win_slot : g_win_slot + 1, 0:HEAD_DIM
                        ]
                    else:
                        csa_kv_flat[g_dst0 + g_r : g_dst0 + g_r + 1, 0:HEAD_DIM] = pl.full(
                            [1, HEAD_DIM], dtype=pl.BF16, value=0.0
                        )
                else:
                    csa_kv_flat[g_dst0 + g_r : g_dst0 + g_r + 1, 0:HEAD_DIM] = pl.full(
                        [1, HEAD_DIM], dtype=pl.BF16, value=0.0
                    )
        else:
            # Per-row C8→BF16: expand group scales [1,8]→[1,512] (32B-aligned tiles).
            g_scale_idx = pl.cast(
                pl.div(
                    pl.cast(pl.arange(0, [1, HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32),
                    64.0,
                ),
                target_type=pl.INT32,
                mode="trunc",
            )
            for g_r in pl.range(ATTN_K_TILE):
                g_k = g_s0 + g_r
                g_cmp_k = g_k - WIN
                g_dst = g_dst0 + g_r
                if g_cmp_k < CMP_TOPK:
                    g_ridx = pl.read(cmp_sparse_indices, [g_t, g_cmp_k])
                    if g_ridx >= 0:
                        g_cblk = pl.cast(pl.read(cmp_block_table, [g_b, g_ridx // BLOCK_SIZE]), pl.INDEX)
                        g_csrc = g_cblk * BLOCK_SIZE + g_ridx % BLOCK_SIZE
                        g_row_fp8 = cmp_kv_flat[g_csrc : g_csrc + 1, 0:HEAD_DIM]
                        g_row_sc = cmp_kv_scale_flat[g_csrc : g_csrc + 1, 0:KV_SCALE_COLS]
                        g_sc_exp = pl.gather(g_row_sc, dim=-1, index=g_scale_idx)
                        g_row_f = pl.cast(g_row_fp8, target_type=pl.FP32)
                        g_dq = pl.mul(g_row_f, g_sc_exp)
                        csa_kv_flat[g_dst : g_dst + 1, 0:HEAD_DIM] = pl.cast(
                            g_dq, target_type=pl.BF16, mode="rint"
                        )
                    else:
                        csa_kv_flat[g_dst : g_dst + 1, 0:HEAD_DIM] = pl.full(
                            [1, HEAD_DIM], dtype=pl.BF16, value=0.0
                        )
                else:
                    csa_kv_flat[g_dst : g_dst + 1, 0:HEAD_DIM] = pl.full(
                        [1, HEAD_DIM], dtype=pl.BF16, value=0.0
                    )

    # One lane per core. Each lane walks its planned items and loads pre-gathered BF16 KV.
    with pl.spmd(NUM_QK_CORES, name_hint="qk_pv", deps=[gather_tid]) as qk_tid:
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
            qk_token_base = qk_t * (H // H_TILE) * SPARSE_BLOCKS * H_TILE
            qk_s0 = qk_sb * ATTN_K_TILE
            qk_bias_row = sparse_bias[qk_t : qk_t + 1, qk_s0 : qk_s0 + ATTN_K_TILE]
            qk_block_valid = pl.read(valid_block_mask, [qk_t, qk_sb])
            if qk_block_valid > 0:
                qk_base = qk_t * PADDED_TOPK + qk_s0
                qk_kv = csa_kv_flat[qk_base : qk_base + ATTN_K_TILE, 0:HEAD_DIM]

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
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_cs"):
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
    # with-form spmd so the dispatch TaskId (merge_tid) can be an explicit dep of
    # the manual-scope proj_a tasks below (which read merge_norm's o_packed cols).
    with pl.spmd(T * (H // H_TILE), name_hint="merge_norm") as merge_tid:
        m_idx = pl.tile.get_block_idx()
        m_t = m_idx // (H // H_TILE)
        m_h_idx = m_idx - m_t * (H // H_TILE)
        m_h0 = m_h_idx * H_TILE
        m_blk_base = m_idx * SPARSE_BLOCKS * H_TILE
        m_mi = sparse_blk_mi[m_blk_base : m_blk_base + H_TILE, 0 : 1]
        m_li = sparse_blk_li[m_blk_base : m_blk_base + H_TILE, 0 : 1]
        m_oi = sparse_blk_oi[m_blk_base : m_blk_base + H_TILE, 0 : HEAD_DIM]

        # Guarded so the SWA (SPARSE_BLOCKS == 1) specialization uses the
        # single block's stats directly instead of an empty merge loop.
        if SPARSE_BLOCKS > 1:
            for m_sb in pl.range(1, SPARSE_BLOCKS):
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
        n_full = pl.row_expand_div(m_oi, n_denom)[0 : H_TILE, 0 : HEAD_DIM]
        n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode="rint")

        # Inverse RoPE on this head-tile's fp32 rope segment. cos_il / sign*sin are
        # head-invariant for token m_t, so col_expand them over the H_TILE head rows;
        # swap_idx (j^1) pairs the interleaved real/imag lanes. Rounded to bf16 (golden
        # also rounds inverse-RoPE to bf16) and packed into o_packed's rope columns.
        m_col = pl.col_expand_mul(
            pl.full([H_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0),
            pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32))
        m_dup_f = pl.cast(pl.cast(pl.mul(m_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        m_lane = pl.sub(m_col, pl.mul(m_dup_f, 2.0))                                              # j%2
        m_swap_idx = pl.cast(pl.sub(pl.add(m_col, 1.0), pl.mul(m_lane, 2.0)), target_type=pl.INT32)  # j^1
        m_rope = n_full[0 : H_TILE, NOPE_DIM : HEAD_DIM]
        m_cos_il = rope_cos_il[m_t : m_t + 1, 0 : ROPE_DIM]
        m_sin_signed = rope_sin_signed[m_t : m_t + 1, 0 : ROPE_DIM]
        m_swapped = pl.gather(m_rope, dim=-1, index=m_swap_idx)
        m_rot = pl.add(pl.col_expand_mul(m_rope, m_cos_il), pl.col_expand_mul(m_swapped, m_sin_signed))
        n_rope_bf16 = pl.cast(m_rot, target_type=pl.BF16, mode="rint")

        for n_hi in pl.range(H_TILE):
            n_gh = m_h0 + n_hi
            n_g = n_gh // HEADS_PER_GROUP
            n_hh = n_gh - n_g * HEADS_PER_GROUP
            n_pack_row = n_g * T + m_t
            n_col = n_hh * HEAD_DIM
            o_packed[n_pack_row : n_pack_row + 1, n_col : n_col + NOPE_DIM] = n_bf16[n_hi : n_hi + 1, 0 : NOPE_DIM]
            o_packed[n_pack_row : n_pack_row + 1, n_col + NOPE_DIM : n_col + HEAD_DIM] = n_rope_bf16[n_hi : n_hi + 1, 0 : ROPE_DIM]

    # ========================================================================
    # Back-to-back grouped MXFP8 output projection (manual scope).
    #
    # Per-group pipeline: proj_a_mm[g] -> proj_b_mm[g] -> proj_b_act sums all
    # group FP32 partials and casts to BF16. Dyn MX quant runs inside each K-loop.
    # ========================================================================
    o_r_pad = pl.create_tensor([T_PAD, O_LORA_TOTAL], dtype=pl.FP32)
    partials = pl.create_tensor([T_PAD, O_GROUPS * D], dtype=pl.FP32)
    proj_b_tids = pl.array.create(O_GROUPS, pl.TASK_ID)
    wo_a_kn = pl.reshape(wo_a, [O_GROUPS * O_GROUP_IN, O_LORA])
    wo_a_scale_kn = pl.reshape(wo_a_scale, [O_GROUPS * O_GROUP_IN // MX_BLOCK_K, O_LORA])

    with pl.manual_scope():
        for g in pl.parallel(O_GROUPS):
            row_base_o = g * T
            out_col_g = g * O_LORA
            col_g = g * O_LORA
            wa_row0 = g * O_GROUP_IN

            with pl.spmd(
                PA_NFRAGS,
                name_hint="proj_a_mm",
                deps=[merge_tid],
            ) as pa_tid:
                nf = pl.tile.get_block_idx()
                n0 = nf * PROJ_A_MM_N_TILE
                acc_a = pl.create_tile(
                    [MM_T_TILE, PROJ_A_MM_N_TILE], dtype=pl.FP32, target_memory=pl.Mem.Acc
                )
                for k0 in pl.pipeline(0, O_GROUP_IN, A_K_TILE, stage=2):
                    xa_tile = pl.load(
                        o_packed,
                        [row_base_o, k0],
                        [MM_T_TILE, A_K_TILE],
                        valid_shapes=[T, A_K_TILE],
                        target_memory=pl.Mem.Vec,
                    )
                    xa_f = pl.cast(xa_tile, target_type=pl.FP32, mode="none")
                    xa_q, xa_s = pl.mx_quant(xa_f, mode="mxfp8_e4m3")
                    wa_tile = pl.load(
                        wo_a_kn,
                        [wa_row0 + k0, n0],
                        [A_K_TILE, PROJ_A_MM_N_TILE],
                        target_memory=pl.Mem.Mat,
                    )
                    was_tile = pl.load(
                        wo_a_scale_kn,
                        [(wa_row0 + k0) // MX_BLOCK_K, n0],
                        [A_K_TILE // MX_BLOCK_K, PROJ_A_MM_N_TILE],
                        target_memory=pl.Mem.Mat,
                        mx_layout="mx_b_nn",
                    )
                    la = pl.move(
                        pl.move(xa_q, target_memory=pl.Mem.Mat), target_memory=pl.Mem.Left
                    )
                    las = pl.move(
                        pl.move(xa_s, target_memory=pl.Mem.Mat), target_memory=pl.Mem.LeftScale
                    )
                    rb = pl.move(wa_tile, target_memory=pl.Mem.Right)
                    rbs = pl.move(was_tile, target_memory=pl.Mem.RightScale)
                    las = pl.tget_scale_addr(las, la)
                    rbs = pl.tget_scale_addr(rbs, rb)
                    acc_a = pl.matmul_mx_acc(acc_a, la, las, rb, rbs)
                pl.store(acc_a, [0, out_col_g + n0], o_r_pad)

            with pl.spmd(PB_DCHUNKS, name_hint="proj_b_mm", deps=[pa_tid]) as pb_tid:
                dc = pl.tile.get_block_idx()
                d0 = dc * PROJ_B_D_CHUNK
                for nf in pl.range(PROJ_B_D_CHUNK // PROJ_B_MM_N_TILE):
                    n0 = d0 + nf * PROJ_B_MM_N_TILE
                    acc_b = pl.create_tile(
                        [MM_T_TILE, PROJ_B_MM_N_TILE], dtype=pl.FP32, target_memory=pl.Mem.Acc
                    )
                    for k0 in pl.pipeline(col_g, col_g + O_LORA, B_K_TILE, stage=2):
                        or_tile = pl.load(
                            o_r_pad,
                            [0, k0],
                            [MM_T_TILE, B_K_TILE],
                            valid_shapes=[T, B_K_TILE],
                            target_memory=pl.Mem.Vec,
                        )
                        or_q, or_s = pl.mx_quant(or_tile, mode="mxfp8_e4m3")
                        wb_tile = pl.load(
                            wo_b,
                            [k0, n0],
                            [B_K_TILE, PROJ_B_MM_N_TILE],
                            target_memory=pl.Mem.Mat,
                        )
                        wbs_tile = pl.load(
                            wo_b_scale,
                            [k0 // MX_BLOCK_K, n0],
                            [B_K_TILE // MX_BLOCK_K, PROJ_B_MM_N_TILE],
                            target_memory=pl.Mem.Mat,
                            mx_layout="mx_b_nn",
                        )
                        ob_la = pl.move(
                            pl.move(or_q, target_memory=pl.Mem.Mat), target_memory=pl.Mem.Left
                        )
                        ob_las = pl.move(
                            pl.move(or_s, target_memory=pl.Mem.Mat), target_memory=pl.Mem.LeftScale
                        )
                        ob_rb = pl.move(wb_tile, target_memory=pl.Mem.Right)
                        ob_rbs = pl.move(wbs_tile, target_memory=pl.Mem.RightScale)
                        ob_las = pl.tget_scale_addr(ob_las, ob_la)
                        ob_rbs = pl.tget_scale_addr(ob_rbs, ob_rb)
                        acc_b = pl.matmul_mx_acc(acc_b, ob_la, ob_las, ob_rb, ob_rbs)
                    pl.store(acc_b, [0, g * D + n0], partials)
            proj_b_tids[g] = pb_tid

    with pl.spmd(
        PB_ACT_NREG * PB_ACT_TBLKS,
        name_hint="proj_b_act",
        deps=[proj_b_tids[i] for i in range(O_GROUPS)],
    ) as _act_tid:
        act_idx = pl.tile.get_block_idx()
        nreg = act_idx // PB_ACT_TBLKS
        tblk = act_idx - nreg * PB_ACT_TBLKS
        ob_n0 = nreg * PROJ_B_ACT_N_TILE
        t0 = tblk * PROJ_B_ACT_TBLK
        for b_tb in pl.range(t0, t0 + PROJ_B_ACT_TBLK, PROJ_B_ACT_T_TILE):
            acc = pl.full([PROJ_B_ACT_T_TILE, PROJ_B_ACT_N_TILE], dtype=pl.FP32, value=0.0)
            for act_g in pl.pipeline(O_GROUPS, stage=2):
                p_col0 = act_g * D + ob_n0
                p_g = partials[b_tb : b_tb + PROJ_B_ACT_T_TILE, p_col0 : p_col0 + PROJ_B_ACT_N_TILE]
                acc = pl.add(acc, p_g)
            out_bf16 = pl.cast(acc, target_type=pl.BF16, mode="rint")
            attn_out[b_tb : b_tb + PROJ_B_ACT_T_TILE, ob_n0 : ob_n0 + PROJ_B_ACT_N_TILE] = out_bf16

    return attn_out

@pl.jit
def sparse_attn_test(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    window_swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.FP8E4M3FN],
    cmp_kv_scale: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, KV_SCALE_COLS], pl.FP32],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    idx_topk: pl.Tensor[[T, INDEXER_SCORE_LEN], pl.INT32],
    position_ids: pl.Tensor[[T, 1], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[O_GROUPS, O_GROUP_IN_SCALE, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[O_LORA_TOTAL, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[O_LORA_TOTAL_SCALE, D], pl.FP8E8M0],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    sparse_attn(
        q,
        ori_kv,
        window_swa_indices,
        cmp_kv,
        cmp_kv_scale,
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
    cmp_kv_fp8 = tensors["cmp_kv"]
    cmp_kv_scale = tensors["cmp_kv_scale"]
    cmp_kv = dequant_kv_c8_fp32_scale(
        cmp_kv_fp8.view(-1, HEAD_DIM),
        cmp_kv_scale.view(-1, KV_SCALE_COLS),
    ).view(CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
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
    wo_a = tensors["wo_a"]
    wo_a_scale = tensors["wo_a_scale"]
    wo_b = tensors["wo_b"]
    wo_b_scale = tensors["wo_b_scale"]

    def _b_scale(s):
        return unpack_scale_b_nn(s)

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
    o_model = o.float().view(B, seq_per_batch, O_GROUPS, O_GROUP_IN)
    out = torch.zeros(T, D, dtype=torch.float32)
    for g in range(O_GROUPS):
        o_g = o_model[:, :, g, :].reshape(T, O_GROUP_IN)
        o_r_q, o_r_s = dynamic_mx_quant_e4m3(o_g)
        o_r = mx_matmul_fp8(o_r_q, o_r_s, wo_a[g], _b_scale(wo_a_scale[g]))
        o_r_q2, o_r_s2 = dynamic_mx_quant_e4m3(o_r)
        wo_b_g = wo_b[g * O_LORA : (g + 1) * O_LORA, :]
        wo_b_s_g = wo_b_scale[g * O_LORA // MX_BLOCK_K : (g + 1) * O_LORA // MX_BLOCK_K, :]
        out = out + mx_matmul_fp8(o_r_q2, o_r_s2, wo_b_g, _b_scale(wo_b_s_g))

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
        """Initialize compressed-cache KV pages as C8 (e4m3 + FP32 group-64 scale)."""
        return cmp_kv_fp8

    def init_cmp_kv_scale():
        """Companion FP32 scales written with init_cmp_kv."""
        return cmp_kv_scale_fp32

    # Pre-quantize demo cmp_kv once so fp8 + scale stay paired.
    _cmp_bf16 = (torch.rand(CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM) - 0.5).to(torch.bfloat16)
    cmp_kv_fp8 = torch.empty(CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.float8_e4m3fn)
    cmp_kv_scale_fp32 = torch.empty(CMP_BLOCK_NUM, BLOCK_SIZE, 1, KV_SCALE_COLS, dtype=torch.float32)
    for _b in range(CMP_BLOCK_NUM):
        for _r in range(BLOCK_SIZE):
            _q, _s = golden_kv_c8_quant_row_fp32_scale(_cmp_bf16[_b, _r, 0])
            cmp_kv_fp8[_b, _r, 0] = _q
            cmp_kv_scale_fp32[_b, _r, 0] = _s
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

    wo_a_stacked = []
    wo_a_scale_stacked = []
    for _g in range(O_GROUPS):
        wa, was = gen_mxfp8_weight_kn(
            (O_GROUP_IN, O_LORA),
            dequant_std=1.0 / (O_GROUP_IN ** 0.5),
            chan_cv=0.50,
        )
        wo_a_stacked.append(wa)
        wo_a_scale_stacked.append(was)
    wo_a_tensor = torch.stack(wo_a_stacked, dim=0)
    wo_a_scale_tensor = torch.stack(wo_a_scale_stacked, dim=0)

    def init_wo_a():
        """Initialize grouped o_a MXFP8 weights [G, K, N]."""
        return wo_a_tensor

    def init_wo_a_scale():
        return wo_a_scale_tensor

    wo_b, wo_b_scale = gen_mxfp8_weight_kn(
        (O_LORA_TOTAL, D),
        dequant_std=1.0 / (O_LORA_TOTAL ** 0.5),
        chan_cv=0.50,
    )

    def init_wo_b():
        return wo_b

    def init_wo_b_scale():
        return wo_b_scale

    return [
        TensorSpec("q", [T, H, HEAD_DIM], torch.bfloat16, init_value=init_q),
        TensorSpec("ori_kv", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_ori_kv),
        TensorSpec("window_swa_indices", [T, WIN], torch.int32, init_value=init_window_swa_indices),
        TensorSpec("cmp_kv", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.float8_e4m3fn, init_value=init_cmp_kv),
        TensorSpec("cmp_kv_scale", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, KV_SCALE_COLS], torch.float32, init_value=init_cmp_kv_scale),
        TensorSpec("cmp_block_table", [B, CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("idx_topk", [T, INDEXER_SCORE_LEN], torch.int32, init_value=init_idx_topk),
        TensorSpec("position_ids", [T, 1], torch.int32, init_value=init_position_ids),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("freqs_cos", [T, ROPE_DIM], torch.bfloat16, init_value=init_cos),
        TensorSpec("freqs_sin", [T, ROPE_DIM], torch.bfloat16, init_value=init_sin),
        TensorSpec("wo_a", [O_GROUPS, O_GROUP_IN, O_LORA], torch.float8_e4m3fn, init_value=init_wo_a),
        TensorSpec("wo_a_scale", [O_GROUPS, O_GROUP_IN_SCALE, O_LORA], torch.float8_e8m0fnu, init_value=init_wo_a_scale),
        TensorSpec("wo_b", [O_LORA_TOTAL, D], torch.float8_e4m3fn, init_value=init_wo_b),
        TensorSpec("wo_b_scale", [O_LORA_TOTAL_SCALE, D], torch.float8_e8m0fnu, init_value=init_wo_b_scale),
        TensorSpec("attn_out", [T, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
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
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    compress_ratio = args.compress_ratio
    print(f"compress_ratio={compress_ratio} "
          f"-> TOPK={TOPK} SPARSE_BLOCKS={SPARSE_BLOCKS} PADDED_TOPK={PADDED_TOPK}", flush=True)

    oproj_tol = ATOL_RTOL["oproj_mxfp8"]
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
            "attn_out": ratio_allclose(atol=oproj_tol["atol"], rtol=oproj_tol["rtol"]),
        },
        compile_only=args.compile_only,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
