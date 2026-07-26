# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 sparse attention with grouped MXFP8 output projection (decode, HCA).

FA / RoPE / merge stay BF16 (存8算16): ``cmp_kv`` is C8 (e4m3 + group-64 FP32
scale interim); window ``ori_kv`` remains BF16.
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
    unpack_scale_b_nn_tiled,
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
TOPK_FULL = WIN + IDX_TOPK           # full sparse-K width (window + indexer topk)
SOFTMAX_SCALE = M.softmax_scale
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

# kernel-local
SUPPORTED_COMPRESS_RATIOS = (0, 4, 128)
DEFAULT_COMPRESS_RATIO = 128
ORI_MAX_BLOCKS = KV_ORI_MAX_BLOCKS
ORI_BLOCK_NUM = DECODE_ORI_BLOCK_NUM
CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
CMP_BLOCK_NUM = DECODE_CMP_BLOCK_NUM

# tiling
VALID_TOKEN_TILE = 8
GATHER_FILL_TILE = 128
ROPE_OUT_TOK_TILE = 8
# A5 FP32 tgather corrupts the last row of an exactly-8-row box; slice gathers to 4.
ROPE_GATHER_T_TILE = 4
# Sparse-K gather runs in its own spmd grid (cf. decode_sparse_attn_swa) that
# materializes the per-token KV rows into a contiguous GM buffer, so qk_pv loads
# a plain [ATTN_K_TILE, HEAD_DIM] slice instead of running a 128-iteration
# scalar-read + gather_row loop ahead of its cube work.
# Gather segments per token. 4 keeps the grid at T*4 = 32 blocks, which co-resides
# with the 16 qproj_dequant blocks in one AIV wave -- the gather must overlap the
# Q chain, and a wider grid queues behind it instead.
GATHER_SEGS = 4
# Every segment carries BOTH a slice of the window and a slice of the compressed
# tail. The two have opposite per-row costs (bulk-run window vs scattered per-row
# topk), so splitting them across separate tasks makes the compressed ones the
# straggler that gates qk_pv; interleaving them balances every block instead.
# Window sub-tile probed for physical contiguity: a decode window is a run of
# consecutive logical positions, so a sub-tile that stays inside one paged block
# maps to consecutive cache rows and moves in ONE bulk DMA. Only the sub-tile
# straddling a block boundary falls back to the per-row copy (~0.44us/row vs
# ~0.02us/row bulk).
GATHER_RUN = 16
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
# proj_a cube K-frag. 256 (not 128) keeps the B-cache-line floor: B is K-contiguous
# under b_trans, so K*2B(bf16) = 512B == the a2a3 L2 line (K=128 was 256B, half a
# line -> wasted MTE2 DMA). At 256 the cube's L0A/L0B operand staging hits 100%
# (the wall); 512 would spill it for no gain (swept: K=512 net-negative).
A_K_TILE = 256
# proj_a is a pure-cube matmul scope (proj_a_mm) writing the fp32 GM intermediate
# o_r (cf. expert_routed w2 decouple), consumed directly by the fused amax+quant
# scope below; the decouple frees the cube N-frag from any vector-side UB constraint.
PROJ_A_MM_N_TILE = 128
MM_T_TILE = 16
T_PAD = ((T + MM_T_TILE - 1) // MM_T_TILE) * MM_T_TILE
B_K_TILE = 256
# proj_b_mm writes grouped INT32 partials; proj_b_act dequantizes and sums them.
PROJ_B_MM_N_TILE = 256
PROJ_B_ACT_N_TILE = 512   # vector N frag for the decoupled per-group dequant+sum (proj_b_act
                          # now sums O_GROUPS INT32 partials, each x its group act scale, then
                          # x the per-channel weight scale -> BF16). 512 (not 1024) keeps the
                          # O_GROUPS-way accumulate inside UB and gives D/512 = 8 vector tasks.
# Fused per-group amax+quant processes the full [8, 1024] row tile.
QUANT_TOKEN_TILE = 8
# Per-group back-to-back o_proj (manual-scope, qwen3-style fine-grained deps):
# proj_a[g] -> quant[g] (PER-GROUP amax, no global barrier) -> proj_b[g] pipeline.
PA_NFRAGS = O_LORA // PROJ_A_MM_N_TILE   # proj_a cube N-frags per group
# Each proj_b group has eight blocks with two N-fragments per block.
PROJ_B_D_CHUNK = 512
PB_DCHUNKS = D // PROJ_B_D_CHUNK
# proj_b_act is split per (D-region, token-block) so the O_GROUPS-way dequant+sum spreads
# over vector cores.
PROJ_B_ACT_T_TILE = 8    # inner token tile for the proj_b_act O_GROUPS-way INT32->FP32 accumulate
PROJ_B_ACT_TBLK = 8      # proj_b_act token block per task
PB_ACT_NREG = D // PROJ_B_ACT_N_TILE
PB_ACT_TBLKS = T // PROJ_B_ACT_TBLK
NEG_INF = -1.0e20

assert T % VALID_TOKEN_TILE == 0
assert T % 2 == 0
assert T % ROPE_GATHER_T_TILE == 0
assert H_TILE % ROPE_GATHER_T_TILE == 0
assert H % 4 == 0
assert QK_M_TILE % H_TILE == 0
assert H % QK_M_TILE == 0
assert T % QUANT_TOKEN_TILE == 0
assert H % O_GROUPS == 0
assert O_LORA % PROJ_A_MM_N_TILE == 0, "proj_a cube N-grid must cover O_LORA"
assert (O_GROUPS * O_LORA) % B_K_TILE == 0
assert D % PROJ_B_MM_N_TILE == 0, "proj_b_mm cube N-loop must cover D"
assert D % PROJ_B_D_CHUNK == 0, "proj_b D-chunk loop must cover D"
assert PROJ_B_D_CHUNK % PROJ_B_MM_N_TILE == 0, "proj_b inner N-frag loop must cover the D-chunk"
assert T % PROJ_B_ACT_TBLK == 0 and PROJ_B_ACT_TBLK % PROJ_B_ACT_T_TILE == 0
assert D % PROJ_B_ACT_N_TILE == 0, "proj_b_act vector N-loop must cover D"
assert O_LORA % B_K_TILE == 0, "proj_b group K-loop covers O_LORA in B_K_TILE iters"
assert O_GROUP_IN % MX_BLOCK_K == 0
assert O_LORA % MX_BLOCK_K == 0
assert (O_GROUPS * O_LORA) % MX_BLOCK_K == 0

_A_K_CHUNKS = O_GROUP_IN // A_K_TILE
_B_K_CHUNKS = O_LORA // B_K_TILE
_A_KS = A_K_TILE // MX_BLOCK_K
_B_KS = B_K_TILE // MX_BLOCK_K
_WO_A_SCALE_ROWS_PER_G = PA_NFRAGS * _A_K_CHUNKS * _A_KS
_WO_B_NUM_N = D // PROJ_B_MM_N_TILE
_WO_B_NUM_K = (O_GROUPS * O_LORA) // B_K_TILE
_WO_B_SCALE_ROWS = _WO_B_NUM_N * _WO_B_NUM_K * _B_KS
_A_SLOTS = O_GROUPS * PA_NFRAGS * _A_K_CHUNKS
_B_SLOTS = O_GROUPS * PB_DCHUNKS * _B_K_CHUNKS
_MX_WS_SLOTS = _A_SLOTS + _B_SLOTS
assert _WO_A_SCALE_ROWS_PER_G == PA_NFRAGS * (O_GROUP_IN // MX_BLOCK_K)
assert _A_K_CHUNKS == 16  # pl.unroll literal in proj_a_mm
assert _B_K_CHUNKS == 4   # pl.unroll literal in proj_b_mm


def get_standalone_cmp_valid(compress_ratio: int) -> int:
    """Map demo compress-ratio modes to the valid compressed-cache tail length."""
    if compress_ratio == 0:
        return 0
    if compress_ratio == 4:
        return IDX_TOPK
    if compress_ratio == 128:
        return MAX_SEQ_LEN // compress_ratio
    raise ValueError(f"Unsupported compress_ratio={compress_ratio}; expected one of {SUPPORTED_COMPRESS_RATIOS}")


CMP_TOPK = min(MAX_SEQ_LEN // 128, IDX_TOPK)
# HCA sparse-K width: cache-first window slots + the deterministic
# ratio-128 compressed tail.
TOPK = WIN + CMP_TOPK
# Floor to 2: a single sparse-K block miscompiles in pypto (S-stride cross-token
# output mixup); a 2-block build with an all-invalid 2nd block is bit-exact.
SPARSE_BLOCKS = max(2, (TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE)
PADDED_TOPK = SPARSE_BLOCKS * ATTN_K_TILE
GATHER_WIN_ROWS = WIN // GATHER_SEGS
GATHER_CMP_ROWS = (PADDED_TOPK - WIN) // GATHER_SEGS
assert WIN <= TOPK <= TOPK_FULL, f"TOPK ({TOPK}) must be in [WIN={WIN}, TOPK_FULL={TOPK_FULL}]"
assert WIN == ATTN_K_TILE, f"HCA window tile requires WIN ({WIN}) == ATTN_K_TILE ({ATTN_K_TILE})"
assert WIN % GATHER_SEGS == 0 and (PADDED_TOPK - WIN) % GATHER_SEGS == 0
assert GATHER_WIN_ROWS % GATHER_RUN == 0, "window bulk-copy runs must tile the window slice"
assert BLOCK_SIZE % GATHER_RUN == 0, "a contiguous run must not straddle two paged blocks by construction"


@pl.jit.inline
def sparse_attn_hca(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    window_swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.FP8E4M3FN],
    cmp_kv_scale: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, KV_SCALE_COLS], pl.FP32],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, CMP_TOPK], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[O_GROUPS, _WO_A_SCALE_ROWS_PER_G, PROJ_A_MM_N_TILE], pl.FP8E8M0],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[_WO_B_SCALE_ROWS, PROJ_B_MM_N_TILE], pl.FP8E8M0],
    attn_out: pl.Tensor[[T, D], pl.BF16],
):
    """Run sparse decode attention, inverse RoPE, and grouped output projection."""
    # Gather the historical/current window + compressed-cache rows.
    # Compressed index contract:
    #   -1              invalid
    #   [0, ...)        compressed KV slots
    ori_kv_flat = pl.reshape(ori_kv, [ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    cmp_kv_scale_flat = pl.reshape(cmp_kv_scale, [CMP_BLOCK_NUM * BLOCK_SIZE, KV_SCALE_COLS])
    sparse_bias = pl.create_tensor([T, PADDED_TOPK], dtype=pl.FP32)

    # Additive softmax bias (0 valid / NEG_INF invalid) that qk_pv adds onto the
    # scaled scores, so invalid lanes exp to ~0 with no per-block mask multiply.
    # Window bias can be vectorized from INT32 indices; compressed bias cannot —
    # vector cast of cmp_sparse_indices → bias wrongly marks t>0 slots NEG_INF on
    # A5 (same bug as decode_sparse_attn). Init compressed lanes to NEG_INF and
    # fill 0.0 / NEG_INF via scalar writes inside hca_gather_kv.
    with pl.spmd(T // VALID_TOKEN_TILE, name_hint="build_valid") as valid_tid:
        v_blk = pl.tile.get_block_idx()
        v_t0 = v_blk * VALID_TOKEN_TILE
        v_win_f = pl.cast(window_swa_indices[v_t0 : v_t0 + VALID_TOKEN_TILE, 0 : WIN], target_type=pl.FP32)
        v_win_valid = pl.minimum(pl.maximum(pl.add(v_win_f, 1.0), 0.0), 1.0)
        sparse_bias[v_t0 : v_t0 + VALID_TOKEN_TILE, 0 : WIN] = pl.mul(pl.sub(v_win_valid, 1.0), -NEG_INF)
        sparse_bias[v_t0 : v_t0 + VALID_TOKEN_TILE, WIN : PADDED_TOPK] = pl.full(
            [VALID_TOKEN_TILE, PADDED_TOPK - WIN], dtype=pl.FP32, value=NEG_INF)

    # Sparse-K gather, hoisted out of qk_pv into its own grid, writing one token's
    # sparse-K rows into the contiguous hca_kv_flat buffer. Every block carries a
    # GATHER_WIN_ROWS slice of the window AND a GATHER_CMP_ROWS slice of the
    # compressed tail, so the cheap bulk runs and the costly scattered rows are
    # spread evenly. Invalid (-1) and padded lanes are zero-filled to match the
    # golden's zero rows; the NEG_INF bias then kills them in the softmax.
    hca_kv_flat = pl.create_tensor([T * PADDED_TOPK, HEAD_DIM], dtype=pl.BF16)
    with pl.spmd(T * GATHER_SEGS, name_hint="hca_gather_kv", deps=[valid_tid]) as gather_tid:
        g_task = pl.tile.get_block_idx()
        g_t = g_task // GATHER_SEGS
        g_seg = g_task - g_t * GATHER_SEGS
        g_b = g_t // S
        g_row0 = g_t * PADDED_TOPK

        # Window slice: probe each sub-tile's first/last slot. Endpoints that are
        # GATHER_RUN-1 apart mean the whole run sits in one paged block.
        g_wk0 = g_seg * GATHER_WIN_ROWS
        for g_sub in pl.range(GATHER_WIN_ROWS // GATHER_RUN):
            g_sk0 = g_wk0 + g_sub * GATHER_RUN
            g_sdst = g_row0 + g_sk0
            g_first = pl.read(window_swa_indices, [g_t, g_sk0])
            g_last = pl.read(window_swa_indices, [g_t, g_sk0 + GATHER_RUN - 1])
            # A -1 slot anywhere in the run pins g_run_ok below the match value,
            # so an invalid or block-straddling run takes the per-row path.
            g_run_ok = (g_last - g_first) + pl.min(g_first, 0) * GATHER_RUN
            if g_run_ok == GATHER_RUN - 1:
                g_run_src = pl.cast(g_first, pl.INDEX)
                hca_kv_flat[g_sdst : g_sdst + GATHER_RUN, 0:HEAD_DIM] = ori_kv_flat[
                    g_run_src : g_run_src + GATHER_RUN, 0:HEAD_DIM
                ]
            else:
                for g_dr in pl.range(GATHER_RUN):
                    g_wdst = g_sdst + g_dr
                    g_win_slot_i32 = pl.read(window_swa_indices, [g_t, g_sk0 + g_dr])
                    if g_win_slot_i32 >= 0:
                        g_win_slot = pl.cast(g_win_slot_i32, pl.INDEX)
                        hca_kv_flat[g_wdst : g_wdst + 1, 0:HEAD_DIM] = ori_kv_flat[
                            g_win_slot : g_win_slot + 1, 0:HEAD_DIM
                        ]
                    else:
                        hca_kv_flat[g_wdst : g_wdst + 1, 0:HEAD_DIM] = pl.full(
                            [1, HEAD_DIM], dtype=pl.BF16, value=0.0)

        # Compressed slice: topk slots are scattered; per-row C8→BF16 dequant
        # (group-64 FP32 scales expanded via gather for 32B row alignment).
        g_ck0 = g_seg * GATHER_CMP_ROWS
        g_cdst0 = g_row0 + WIN + g_ck0
        g_scale_idx = pl.cast(
            pl.div(
                pl.cast(pl.arange(0, [1, HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32),
                64.0,
            ),
            target_type=pl.INT32,
            mode="trunc",
        )
        for g_dr in pl.range(GATHER_CMP_ROWS):
            g_dst = g_cdst0 + g_dr
            g_cmp_k = g_ck0 + g_dr
            g_k = WIN + g_cmp_k
            if g_cmp_k < CMP_TOPK:
                g_ridx = pl.read(cmp_sparse_indices, [g_t, g_cmp_k])
                if g_ridx >= 0:
                    pl.write(sparse_bias, [g_t, g_k], 0.0)
                    g_cblk = pl.cast(pl.read(cmp_block_table, [g_b, g_ridx // BLOCK_SIZE]), pl.INDEX)
                    g_csrc = g_cblk * BLOCK_SIZE + g_ridx % BLOCK_SIZE
                    g_row_fp8 = cmp_kv_flat[g_csrc : g_csrc + 1, 0:HEAD_DIM]
                    g_row_sc = cmp_kv_scale_flat[g_csrc : g_csrc + 1, 0:KV_SCALE_COLS]
                    g_sc_exp = pl.gather(g_row_sc, dim=-1, index=g_scale_idx)
                    g_row_f = pl.cast(g_row_fp8, target_type=pl.FP32)
                    g_dq = pl.mul(g_row_f, g_sc_exp)
                    hca_kv_flat[g_dst : g_dst + 1, 0:HEAD_DIM] = pl.cast(
                        g_dq, target_type=pl.BF16, mode="rint"
                    )
                else:
                    pl.write(sparse_bias, [g_t, g_k], NEG_INF)
                    hca_kv_flat[g_dst : g_dst + 1, 0:HEAD_DIM] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
            else:
                hca_kv_flat[g_dst : g_dst + 1, 0:HEAD_DIM] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)

    # qk_pv writes per-tile (mi, li, oi) to GM; merge_norm reads them back. Not
    # fused on a2a3: the PV output (Acc) -> online rescale (Vec) needs an
    # unsupported tmov, and a [H_TILE, HEAD_DIM] carry overflows the Vec buffer.
    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    o_packed = pl.create_tensor([O_GROUPS * T, O_GROUP_IN], dtype=pl.BF16)
    sparse_blk_mi = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, 1], dtype=pl.FP32)
    sparse_blk_oi = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, HEAD_DIM], dtype=pl.FP32)

    with pl.spmd(T * SPARSE_BLOCKS, name_hint="qk_pv", deps=[gather_tid]) as qk_tid:
        qk_item = pl.tile.get_block_idx()
        qk_t = qk_item // SPARSE_BLOCKS
        qk_sb = qk_item - qk_t * SPARSE_BLOCKS
        qk_token_base = qk_t * (H // H_TILE) * SPARSE_BLOCKS * H_TILE
        # Sparse-block OUTER / head-tile INNER: both head-batches' QK (b_trans)
        # and PV consume the SAME pre-gathered KV tile.
        qk_s0 = qk_sb * ATTN_K_TILE
        qk_bias_row = sparse_bias[qk_t : qk_t + 1, qk_s0 : qk_s0 + ATTN_K_TILE]
        qk_base = qk_t * PADDED_TOPK + qk_s0
        qk_kv = hca_kv_flat[qk_base : qk_base + ATTN_K_TILE, 0:HEAD_DIM]

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
            qk_scores = pl.col_expand_add(qk_scaled, qk_bias_row)
            qk_mi = pl.maximum(pl.row_max(qk_scores), -1.0e20)
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

    # Precompute the head-invariant interleaved cos and sign*sin once: they depend
    # only on (token, column), not head, so building them per head would repeat the
    # same dup-gather H times on the bottleneck Vec engine. sign is folded into sin
    # (multiply by +/-1). The conjugate (inverse) rotation is:
    #   out[j] = x[j]*cos_il[j] + x[j^1]*sign[j]*sin_il[j]
    # Hoisted ABOVE merge_norm (which now fuses the rotation): independent of qk_pv,
    # so it overlaps it and is off merge_norm's critical path.
    rope_cos_il = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_cs") as rope_tid:
        # Slice T=8 gathers into ROPE_GATHER_T_TILE-row tiles (A5 tgather 8-row bug).
        cs_col = pl.col_expand_mul(
            pl.full([ROPE_GATHER_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0),
            pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32))
        cs_dup_f = pl.cast(pl.cast(pl.mul(cs_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        cs_dup_idx = pl.cast(cs_dup_f, target_type=pl.INT32)
        cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))
        cs_sign = pl.neg(pl.sub(pl.mul(cs_lane, 2.0), 1.0))
        for cs_s0 in pl.range(0, T, ROPE_GATHER_T_TILE):
            cs_cos = pl.cast(freqs_cos[cs_s0 : cs_s0 + ROPE_GATHER_T_TILE, 0:HALF_ROPE], target_type=pl.FP32)
            cs_sin = pl.cast(freqs_sin[cs_s0 : cs_s0 + ROPE_GATHER_T_TILE, 0:HALF_ROPE], target_type=pl.FP32)
            rope_cos_il[cs_s0 : cs_s0 + ROPE_GATHER_T_TILE, 0:ROPE_DIM] = pl.gather(
                cs_cos, dim=-1, index=cs_dup_idx
            )
            rope_sin_signed[cs_s0 : cs_s0 + ROPE_GATHER_T_TILE, 0:ROPE_DIM] = pl.mul(
                pl.gather(cs_sin, dim=-1, index=cs_dup_idx), cs_sign
            )

    # Online-softmax merge across sparse-K tiles, sink-norm, then fused inverse RoPE.
    # One spmd block per (token, head-tile) -- T*(H//H_TILE) blocks -- so the merge
    # fans out over that many AIVs instead of T blocks each running a serial head-tile
    # loop. The inverse-RoPE rotation + rope-column pack is fused in (was a separate
    # "rope" spmd reading an attn_rope_stage GM round-trip): the head-tile's fp32 rope
    # segment is rotated in UB and packed straight into o_packed's rope columns.
    # with-form spmd so the dispatch TaskId (merge_tid) can be an explicit dep of
    # the manual-scope proj_a tasks below (which read merge_norm's o_packed cols).
    with pl.spmd(T * (H // H_TILE), name_hint="merge_norm", deps=[qk_tid, rope_tid]) as merge_tid:
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

        # Inverse RoPE gather in ROPE_GATHER_T_TILE-row subtiles.
        m_col = pl.col_expand_mul(
            pl.full([ROPE_GATHER_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0),
            pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32))
        m_dup_f = pl.cast(pl.cast(pl.mul(m_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        m_lane = pl.sub(m_col, pl.mul(m_dup_f, 2.0))
        m_swap_idx = pl.cast(pl.sub(pl.add(m_col, 1.0), pl.mul(m_lane, 2.0)), target_type=pl.INT32)
        m_cos_il = rope_cos_il[m_t : m_t + 1, 0 : ROPE_DIM]
        m_sin_signed = rope_sin_signed[m_t : m_t + 1, 0 : ROPE_DIM]
        n_rope_bf16 = pl.full([H_TILE, ROPE_DIM], dtype=pl.BF16, value=0.0)
        for m_sub in pl.unroll(H_TILE // ROPE_GATHER_T_TILE):
            m_s0 = m_sub * ROPE_GATHER_T_TILE
            m_rope_s = n_full[m_s0 : m_s0 + ROPE_GATHER_T_TILE, NOPE_DIM : HEAD_DIM]
            m_swapped_s = pl.gather(m_rope_s, dim=-1, index=m_swap_idx)
            m_rot_s = pl.add(
                pl.col_expand_mul(m_rope_s, m_cos_il),
                pl.col_expand_mul(m_swapped_s, m_sin_signed),
            )
            n_rope_bf16[m_s0 : m_s0 + ROPE_GATHER_T_TILE, 0 : ROPE_DIM] = pl.cast(
                m_rot_s, target_type=pl.BF16, mode="rint"
            )

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
    # Per-group pipeline: proj_a[g] (dyn MX o_packed @ wo_a) -> proj_b[g]
    # (dyn MX o_r @ wo_b) -> proj_b_act sums FP32 group partials -> BF16.
    # ========================================================================
    o_r_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.FP32)
    partials = pl.create_tensor([T_PAD, O_GROUPS * D], dtype=pl.FP32)
    proj_b_tids = pl.array.create(O_GROUPS, pl.TASK_ID)
    wo_a_kn = pl.reshape(wo_a, [O_GROUPS * O_GROUP_IN, O_LORA])
    wo_a_scale_flat = pl.reshape(
        wo_a_scale, [O_GROUPS * _WO_A_SCALE_ROWS_PER_G, PROJ_A_MM_N_TILE]
    )
    mx_scale_ws = pl.create_tensor(
        [_MX_WS_SLOTS * MM_T_TILE, A_K_TILE // MX_BLOCK_K], dtype=pl.FP8E8M0
    )

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
                # Peel K=0 with matmul_mx (init Acc); remaining via matmul_mx_acc.
                k0 = 0
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
                    wo_a_scale_flat,
                    [
                        g * _WO_A_SCALE_ROWS_PER_G
                        + (nf * _A_K_CHUNKS + k0 // A_K_TILE) * _A_KS,
                        0,
                    ],
                    [A_K_TILE // MX_BLOCK_K, PROJ_A_MM_N_TILE],
                    target_memory=pl.Mem.Mat,
                    mx_layout="mx_b_nn",
                )
                srow = (
                    g * PA_NFRAGS * _A_K_CHUNKS
                    + nf * _A_K_CHUNKS
                    + k0 // A_K_TILE
                ) * MM_T_TILE
                la = pl.move(
                    pl.move(pl.tile.reinterpret_view(xa_q, pl.FP8E4M3FN), target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.Left,
                )
                la = pl.set_validshape(la, T, A_K_TILE)
                pl.store(pl.tile.reinterpret_view(xa_s, pl.FP8E8M0), [srow, 0], mx_scale_ws)
                las = pl.move(
                    pl.load(
                        mx_scale_ws,
                        [srow, 0],
                        [MM_T_TILE, A_K_TILE // MX_BLOCK_K],
                        target_memory=pl.Mem.Mat,
                        mx_layout="mx_a_zz",
                    ),
                    target_memory=pl.Mem.LeftScale,
                )
                las = pl.tget_scale_addr(las, la)
                las = pl.set_validshape(las, T, A_K_TILE // MX_BLOCK_K)
                rb = pl.move(wa_tile, target_memory=pl.Mem.Right)
                rbs = pl.move(was_tile, target_memory=pl.Mem.RightScale)
                rbs = pl.tget_scale_addr(rbs, rb)
                acc_a = pl.matmul_mx(la, las, rb, rbs)
                for db in pl.unroll(15):  # _A_K_CHUNKS - 1
                    k0 = (db + 1) * A_K_TILE
                    xa_tile2 = pl.load(
                        o_packed,
                        [row_base_o, k0],
                        [MM_T_TILE, A_K_TILE],
                        valid_shapes=[T, A_K_TILE],
                        target_memory=pl.Mem.Vec,
                    )
                    xa_f2 = pl.cast(xa_tile2, target_type=pl.FP32, mode="none")
                    xa_q2, xa_s2 = pl.mx_quant(xa_f2, mode="mxfp8_e4m3")
                    wa_tile2 = pl.load(
                        wo_a_kn,
                        [wa_row0 + k0, n0],
                        [A_K_TILE, PROJ_A_MM_N_TILE],
                        target_memory=pl.Mem.Mat,
                    )
                    was_tile2 = pl.load(
                        wo_a_scale_flat,
                        [
                            g * _WO_A_SCALE_ROWS_PER_G
                            + (nf * _A_K_CHUNKS + (db + 1)) * _A_KS,
                            0,
                        ],
                        [A_K_TILE // MX_BLOCK_K, PROJ_A_MM_N_TILE],
                        target_memory=pl.Mem.Mat,
                        mx_layout="mx_b_nn",
                    )
                    srow2 = (
                        g * PA_NFRAGS * _A_K_CHUNKS
                        + nf * _A_K_CHUNKS
                        + (db + 1)
                    ) * MM_T_TILE
                    la2 = pl.move(
                        pl.move(pl.tile.reinterpret_view(xa_q2, pl.FP8E4M3FN), target_memory=pl.Mem.Mat),
                        target_memory=pl.Mem.Left,
                    )
                    la2 = pl.set_validshape(la2, T, A_K_TILE)
                    pl.store(pl.tile.reinterpret_view(xa_s2, pl.FP8E8M0), [srow2, 0], mx_scale_ws)
                    las2 = pl.move(
                        pl.load(
                            mx_scale_ws,
                            [srow2, 0],
                            [MM_T_TILE, A_K_TILE // MX_BLOCK_K],
                            target_memory=pl.Mem.Mat,
                            mx_layout="mx_a_zz",
                        ),
                        target_memory=pl.Mem.LeftScale,
                    )
                    las2 = pl.tget_scale_addr(las2, la2)
                    las2 = pl.set_validshape(las2, T, A_K_TILE // MX_BLOCK_K)
                    rb2 = pl.move(wa_tile2, target_memory=pl.Mem.Right)
                    rbs2 = pl.move(was_tile2, target_memory=pl.Mem.RightScale)
                    rbs2 = pl.tget_scale_addr(rbs2, rb2)
                    acc_a = pl.matmul_mx_acc(acc_a, la2, las2, rb2, rbs2)
                pl.store(acc_a, [0, out_col_g + n0], o_r_pad)

            with pl.spmd(
                PB_DCHUNKS,
                name_hint="proj_b_mm",
                deps=[pa_tid],
            ) as pb_tid:
                dc = pl.tile.get_block_idx()
                d0 = dc * PROJ_B_D_CHUNK
                for nf in pl.range(PROJ_B_D_CHUNK // PROJ_B_MM_N_TILE):
                    n0 = d0 + nf * PROJ_B_MM_N_TILE
                    nb = n0 // PROJ_B_MM_N_TILE
                    # Peel K=0 with matmul_mx (init Acc); remaining via matmul_mx_acc.
                    k0 = col_g
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
                        [
                            (nb * _WO_B_NUM_K + k0 // B_K_TILE) * _B_KS,
                            0,
                        ],
                        [B_K_TILE // MX_BLOCK_K, PROJ_B_MM_N_TILE],
                        target_memory=pl.Mem.Mat,
                        mx_layout="mx_b_nn",
                    )
                    srow = (
                        _A_SLOTS
                        + g * PB_DCHUNKS * _B_K_CHUNKS
                        + dc * _B_K_CHUNKS
                        + (k0 - col_g) // B_K_TILE
                    ) * MM_T_TILE
                    bl = pl.move(
                        pl.move(pl.tile.reinterpret_view(or_q, pl.FP8E4M3FN), target_memory=pl.Mem.Mat),
                        target_memory=pl.Mem.Left,
                    )
                    bl = pl.set_validshape(bl, T, B_K_TILE)
                    pl.store(pl.tile.reinterpret_view(or_s, pl.FP8E8M0), [srow, 0], mx_scale_ws)
                    bls = pl.move(
                        pl.load(
                            mx_scale_ws,
                            [srow, 0],
                            [MM_T_TILE, B_K_TILE // MX_BLOCK_K],
                            target_memory=pl.Mem.Mat,
                            mx_layout="mx_a_zz",
                        ),
                        target_memory=pl.Mem.LeftScale,
                    )
                    bls = pl.tget_scale_addr(bls, bl)
                    bls = pl.set_validshape(bls, T, B_K_TILE // MX_BLOCK_K)
                    br = pl.move(wb_tile, target_memory=pl.Mem.Right)
                    brs = pl.move(wbs_tile, target_memory=pl.Mem.RightScale)
                    brs = pl.tget_scale_addr(brs, br)
                    acc_b = pl.matmul_mx(bl, bls, br, brs)
                    for db in pl.unroll(3):  # _B_K_CHUNKS - 1
                        k0 = col_g + (db + 1) * B_K_TILE
                        or_tile2 = pl.load(
                            o_r_pad,
                            [0, k0],
                            [MM_T_TILE, B_K_TILE],
                            valid_shapes=[T, B_K_TILE],
                            target_memory=pl.Mem.Vec,
                        )
                        or_q2, or_s2 = pl.mx_quant(or_tile2, mode="mxfp8_e4m3")
                        wb_tile2 = pl.load(
                            wo_b,
                            [k0, n0],
                            [B_K_TILE, PROJ_B_MM_N_TILE],
                            target_memory=pl.Mem.Mat,
                        )
                        wbs_tile2 = pl.load(
                            wo_b_scale,
                            [
                                (nb * _WO_B_NUM_K + k0 // B_K_TILE) * _B_KS,
                                0,
                            ],
                            [B_K_TILE // MX_BLOCK_K, PROJ_B_MM_N_TILE],
                            target_memory=pl.Mem.Mat,
                            mx_layout="mx_b_nn",
                        )
                        srow2 = (
                            _A_SLOTS
                            + g * PB_DCHUNKS * _B_K_CHUNKS
                            + dc * _B_K_CHUNKS
                            + (db + 1)
                        ) * MM_T_TILE
                        bl2 = pl.move(
                            pl.move(pl.tile.reinterpret_view(or_q2, pl.FP8E4M3FN), target_memory=pl.Mem.Mat),
                            target_memory=pl.Mem.Left,
                        )
                        bl2 = pl.set_validshape(bl2, T, B_K_TILE)
                        pl.store(pl.tile.reinterpret_view(or_s2, pl.FP8E8M0), [srow2, 0], mx_scale_ws)
                        bls2 = pl.move(
                            pl.load(
                                mx_scale_ws,
                                [srow2, 0],
                                [MM_T_TILE, B_K_TILE // MX_BLOCK_K],
                                target_memory=pl.Mem.Mat,
                                mx_layout="mx_a_zz",
                            ),
                            target_memory=pl.Mem.LeftScale,
                        )
                        bls2 = pl.tget_scale_addr(bls2, bl2)
                        bls2 = pl.set_validshape(bls2, T, B_K_TILE // MX_BLOCK_K)
                        br2 = pl.move(wb_tile2, target_memory=pl.Mem.Right)
                        brs2 = pl.move(wbs_tile2, target_memory=pl.Mem.RightScale)
                        brs2 = pl.tget_scale_addr(brs2, br2)
                        acc_b = pl.matmul_mx_acc(acc_b, bl2, bls2, br2, brs2)
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
    cmp_sparse_indices: pl.Tensor[[T, CMP_TOPK], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[O_GROUPS, _WO_A_SCALE_ROWS_PER_G, PROJ_A_MM_N_TILE], pl.FP8E8M0],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[_WO_B_SCALE_ROWS, PROJ_B_MM_N_TILE], pl.FP8E8M0],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    sparse_attn_hca(
        q,
        ori_kv,
        window_swa_indices,
        cmp_kv,
        cmp_kv_scale,
        cmp_block_table,
        cmp_sparse_indices,
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


def _golden_o_proj_mx(o_model, wo_a, wo_a_scale, wo_b, wo_b_scale):
    """Grouped MLAEpilog o_a + o_b via dyn MX + mx_matmul_fp8 (AscendC Hybrid)."""
    import torch

    def _b_scale_a(s):
        return unpack_scale_b_nn_tiled(
            s,
            k_tile_rows=_A_KS,
            n_tile=PROJ_A_MM_N_TILE,
            logical_k=O_GROUP_IN // MX_BLOCK_K,
            logical_n=O_LORA,
        )

    def _b_scale_b(s):
        return unpack_scale_b_nn_tiled(
            s,
            k_tile_rows=_B_KS,
            n_tile=PROJ_B_MM_N_TILE,
            logical_k=(O_GROUPS * O_LORA) // MX_BLOCK_K,
            logical_n=D,
        )

    def mx_matmul_act_tiled(x_f, w, w_s, k_tile):
        acc = None
        for k0 in range(0, x_f.shape[-1], k_tile):
            xq, xs = dynamic_mx_quant_e4m3(x_f[..., k0 : k0 + k_tile])
            part = mx_matmul_fp8(
                xq, xs, w[k0 : k0 + k_tile], w_s[k0 // MX_BLOCK_K : (k0 + k_tile) // MX_BLOCK_K]
            )
            acc = part if acc is None else acc + part
        return acc

    t_total = o_model.shape[0]
    o_r = torch.zeros(t_total, O_GROUPS, O_LORA, dtype=torch.float32)
    for g in range(O_GROUPS):
        og = o_model[:, g, :]
        o_r[:, g, :] = mx_matmul_act_tiled(og, wo_a[g], _b_scale_a(wo_a_scale[g]), A_K_TILE)

    out = torch.zeros(t_total, D, dtype=torch.float32)
    wb_s = _b_scale_b(wo_b_scale)
    for g in range(O_GROUPS):
        col0 = g * O_LORA
        or_g = o_r[:, g, :]
        wo_b_g = wo_b[col0 : col0 + O_LORA, :]
        wo_b_s_g = wb_s[col0 // MX_BLOCK_K : (col0 + O_LORA) // MX_BLOCK_K, :]
        out += mx_matmul_act_tiled(or_g, wo_b_g, wo_b_s_g, B_K_TILE)
    return out


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
    cmp_sparse_indices = tensors["cmp_sparse_indices"]
    attn_sink = tensors["attn_sink"].float()
    cos = tensors["freqs_cos"].float()
    sin = tensors["freqs_sin"].float()
    wo_a = tensors["wo_a"]
    wo_a_scale = tensors["wo_a_scale"]
    wo_b = tensors["wo_b"]
    wo_b_scale = tensors["wo_b_scale"]

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
    o_model = o.float().view(B, seq_per_batch, O_GROUPS, O_GROUP_IN).reshape(T, O_GROUPS, O_GROUP_IN)
    out = _golden_o_proj_mx(o_model, wo_a, wo_a_scale, wo_b, wo_b_scale)

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

    cmp_valid = min(get_standalone_cmp_valid(compress_ratio), TOPK - WIN)

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
        """Build the sparse index list with a full window prefix and padded compressed tail.

        The compressed tail width follows the active specialization (TOPK - WIN):
        the pruned build narrows it to `cmp_valid` columns, the full-blocks
        baseline keeps the IDX_TOPK-wide padded tail.
        """
        indices = torch.full((T, CMP_TOPK), -1, dtype=torch.int32)
        if cmp_valid:
            indices[:, :cmp_valid] = torch.arange(cmp_valid, dtype=torch.int32)
        if short_window_fixture:
            indices[:, :] = -1
        if mixed_topk_fixture:
            indices[:, :] = -1
            mixed_cmp_valid = min(cmp_valid, IDX_TOPK)
            if mixed_cmp_valid:
                indices[:, :mixed_cmp_valid] = torch.arange(mixed_cmp_valid, dtype=torch.int32)
        if cache_window_replacement_fixture:
            indices[:, :] = -1
        if causal_regression_fixture:
            indices[0, :] = -1
        return indices

    def init_cos():
        """Build the split-half cosine table used by the inverse-RoPE reference."""
        angles = torch.arange(T * HALF_ROPE).reshape(T, HALF_ROPE) * 1e-3
        cos_half = torch.cos(angles)
        return torch.cat([cos_half, cos_half], dim=-1)

    def init_sin():
        """Build the split-half sine table used by the inverse-RoPE reference."""
        angles = torch.arange(T * HALF_ROPE).reshape(T, HALF_ROPE) * 1e-3
        sin_half = torch.sin(angles)
        return torch.cat([sin_half, sin_half], dim=-1)

    wo_a_list = []
    wo_a_scale_list = []
    for _g in range(O_GROUPS):
        wa, was = gen_mxfp8_weight_kn(
            (O_GROUP_IN, O_LORA), dequant_std=0.1, chan_cv=0.50,
            n_tile=PROJ_A_MM_N_TILE, k_tile=A_K_TILE,
        )
        wo_a_list.append(wa)
        wo_a_scale_list.append(was)
    wo_a_mx = torch.stack(wo_a_list, dim=0)
    wo_a_scale_mx = torch.stack(wo_a_scale_list, dim=0)
    wo_b_mx, wo_b_scale_mx = gen_mxfp8_weight_kn(
        (O_GROUPS * O_LORA, D), dequant_std=0.1, chan_cv=0.50,
        n_tile=PROJ_B_MM_N_TILE, k_tile=B_K_TILE,
    )

    def init_wo_a():
        """Initialize grouped o_a MXFP8 Right weights."""
        return wo_a_mx

    def init_wo_a_scale():
        """Initialize grouped o_a MXFP8 E8M0 scales (MX_B_NN packed)."""
        return wo_a_scale_mx

    def init_wo_b():
        """Initialize o_b MXFP8 Right weights."""
        return wo_b_mx

    def init_wo_b_scale():
        """Initialize o_b MXFP8 E8M0 scales (MX_B_NN packed)."""
        return wo_b_scale_mx

    return [
        TensorSpec("q", [T, H, HEAD_DIM], torch.bfloat16, init_value=init_q),
        TensorSpec("ori_kv", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_ori_kv),
        TensorSpec("window_swa_indices", [T, WIN], torch.int32, init_value=init_window_swa_indices),
        TensorSpec("cmp_kv", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.float8_e4m3fn, init_value=init_cmp_kv),
        TensorSpec("cmp_kv_scale", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, KV_SCALE_COLS], torch.float32, init_value=init_cmp_kv_scale),
        TensorSpec("cmp_block_table", [B, CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("cmp_sparse_indices", [T, CMP_TOPK], torch.int32, init_value=init_cmp_sparse_indices),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("freqs_cos", [T, ROPE_DIM], torch.bfloat16, init_value=init_cos),
        TensorSpec("freqs_sin", [T, ROPE_DIM], torch.bfloat16, init_value=init_sin),
        TensorSpec("wo_a", [O_GROUPS, O_GROUP_IN, O_LORA], torch.float8_e4m3fn, init_value=init_wo_a),
        TensorSpec(
            "wo_a_scale",
            [O_GROUPS, _WO_A_SCALE_ROWS_PER_G, PROJ_A_MM_N_TILE],
            torch.float8_e8m0fnu,
            init_value=init_wo_a_scale,
        ),
        TensorSpec("wo_b", [O_GROUPS * O_LORA, D], torch.float8_e4m3fn, init_value=init_wo_b),
        TensorSpec(
            "wo_b_scale",
            [_WO_B_SCALE_ROWS, PROJ_B_MM_N_TILE],
            torch.float8_e8m0fnu,
            init_value=init_wo_b_scale,
        ),
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
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--enable-dep-gen", action="store_true", default=False,
                        help="Capture PTO2 dependency edges (deps.json); the swimlane "
                             "converter draws fanout/fanin arrows from the sibling file.")
    parser.add_argument("--enable-pmu", nargs="?", const=2, default=0, type=int, choices=[0, 1, 2, 4])
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    compress_ratio = args.compress_ratio
    print(f"compress_ratio={compress_ratio} "
          f"-> TOPK={TOPK} SPARSE_BLOCKS={SPARSE_BLOCKS} PADDED_TOPK={PADDED_TOPK}", flush=True)

    oproj_tol = ATOL_RTOL.get("oproj_mxfp8", ATOL_RTOL["fia_mxfp8"])
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
        compile_only=args.compile_only,
        compare_fn={
            # HCA: window + FP8 compressed KV + 2-block merge + MX o_proj; A5 bf16/MX
            # noise concentrates on ~1 token and stays cosine≈1. Allow 5% outlier pts
            # (same band as qkv_mxfp8), tighter than a functional-bug fail.
            "attn_out": ratio_allclose(
                atol=oproj_tol["atol"],
                rtol=oproj_tol["rtol"],
                max_error_ratio=0.05,
            ),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
