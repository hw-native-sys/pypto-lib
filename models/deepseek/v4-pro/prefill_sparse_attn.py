# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 token-major prefill sparse attention with grouped MXFP8 output projection.

Sparse reads are split by cache source. ``swa_indices`` contains physical rows
in the original KV cache; ``cmp_indices`` contains compressed logical slots that
are lowered through ``cmp_block_table``. ``-1`` marks invalid entries.

FA / RoPE / merge stay BF16 (存8算16): ``cmp_kv`` is C8 (e4m3 + group-64 FP32
scale interim); window ``ori_kv`` remains BF16.
"""

import pypto.language as pl

from config import (
    BLOCK_SIZE,
    FLASH as M,
    MX_BLOCK_K,
    PREFILL_BATCH,
    PREFILL_CMP_BLOCK_NUM,
    PREFILL_CMP_MAX_BLOCKS,
    PREFILL_ORI_BLOCK_NUM,
    PREFILL_ORI_MAX_BLOCKS,
    PREFILL_SEQ,
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

# Prefill target shape. T is fixed at 128.
B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S

# Model config.
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_DIM // 2
ROPE_HALF = HALF_ROPE
NOPE_DIM = M.nope_head_dim
IDX_TOPK = M.index_topk
WIN = M.sliding_window
TOPK = WIN + IDX_TOPK
SOFTMAX_SCALE = M.softmax_scale
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
O_GROUP_IN_SCALE = O_GROUP_IN // MX_BLOCK_K
O_LORA_TOTAL = O_GROUPS * O_LORA
O_LORA_TOTAL_SCALE = O_LORA_TOTAL // MX_BLOCK_K

# Cache shapes.
SUPPORTED_COMPRESS_RATIOS = (0, 4, 128)
DEFAULT_COMPRESS_RATIO = 4
PREFILL_MAX_COMPRESSED = max(1, min(IDX_TOPK, WIN + WIN // 2))
PREFILL_SPARSE_TOPK = min(TOPK, min(WIN, S) + PREFILL_MAX_COMPRESSED)
ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS
ORI_BLOCK_NUM = PREFILL_ORI_BLOCK_NUM
CMP_MAX_BLOCKS = PREFILL_CMP_MAX_BLOCKS
CMP_BLOCK_NUM = PREFILL_CMP_BLOCK_NUM

# Kernel tiling (mirrors decode sparse-attn).
HEAD_TILE = 16                       # head-tile granularity for storage / merge
QK_M_TILE = 32                       # head rows cube-batched per QK/PV matmul
GATHER_TOKEN_TILE = 1                 # one token per gather task (A5 pl.write bias coords)
BIAS_TOKEN_TILE = 16
QUANT_TOKEN_TILE = 8
ROPE_GATHER_T_TILE = 4               # A5 tgather wide-box bug: gather <=4 rows
A_K_TILE = 256                       # proj_a cube K-frag: K*2B = 512B = one a2a3 L2 line (128 wastes half)
A_N_TILE = 128
A_T_TILE = 32                        # token tile for the proj_a/proj_b vec post-process
B_K_TILE = 256
B_N_TILE = 128
B_T_TILE = 32
QUANT_TILE = 512
QUANT_K_TILE = O_GROUPS * O_LORA // 2
# o_proj per-group decoupling (MXFP8): proj_a_mm (dyn MX @ wo_a) ->
# proj_b_mm (dyn MX @ wo_b, FP32 partials) -> proj_b_act (sum -> BF16).
PROJ_A_MM_N_TILE = 128                    # proj_a cube N frag (L0A/L0B wall); PA_NFRAGS per group
PROJ_B_MM_N_TILE = 256                    # proj_b_mm cube N frag (L0C wall)
PROJ_B_D_CHUNK = 1024                     # proj_b D-chunk per task (keeps proj_b at O_GROUPS*PB_DCHUNKS tasks)
PROJ_B_ACT_N_TILE = 512                   # proj_b_act vector N frag (UB wall)
PA_NFRAGS = O_LORA // PROJ_A_MM_N_TILE
PB_DCHUNKS = D // PROJ_B_D_CHUNK
NUM_QUANT_T_CHUNKS = 4                    # quant split per (group, token-chunk) over vector cores
QUANT_T_CHUNK = T // NUM_QUANT_T_CHUNKS
PROJ_B_ACT_T_TILE = 16                    # inner token tile for the O_GROUPS-way INT32->FP32 accumulate
PROJ_B_ACT_TBLK = 32                      # proj_b_act token block per task
PB_ACT_NREG = D // PROJ_B_ACT_N_TILE
PB_ACT_TBLKS = T // PROJ_B_ACT_TBLK
assert T % QUANT_TOKEN_TILE == 0
assert T % ROPE_GATHER_T_TILE == 0
assert HEAD_TILE % ROPE_GATHER_T_TILE == 0
assert O_GROUP_IN % A_K_TILE == 0    # proj_a_mm peels 0:A_K_TILE then covers O_GROUP_IN // A_K_TILE chunks
assert O_LORA % B_K_TILE == 0        # proj_b_mm peels 0:B_K_TILE then covers O_LORA // B_K_TILE chunks
assert D % PROJ_B_MM_N_TILE == 0 and D % PROJ_B_D_CHUNK == 0 and PROJ_B_D_CHUNK % PROJ_B_MM_N_TILE == 0
assert T % NUM_QUANT_T_CHUNKS == 0 and QUANT_T_CHUNK % QUANT_TOKEN_TILE == 0
assert T % PROJ_B_ACT_TBLK == 0 and PROJ_B_ACT_TBLK % PROJ_B_ACT_T_TILE == 0
assert D % PROJ_B_ACT_N_TILE == 0 and O_LORA % QUANT_TILE == 0
assert O_GROUP_IN % MX_BLOCK_K == 0
assert O_LORA % MX_BLOCK_K == 0
assert (O_GROUPS * O_LORA) % MX_BLOCK_K == 0

_A_K_CHUNKS = O_GROUP_IN // A_K_TILE
_B_K_CHUNKS = O_LORA // B_K_TILE
_A_KS = A_K_TILE // MX_BLOCK_K
_B_KS = B_K_TILE // MX_BLOCK_K
# Tiled MX_B_NN scale layouts (per convert_x2 tile; col offset always 0).
_WO_A_SCALE_ROWS_PER_G = PA_NFRAGS * _A_K_CHUNKS * _A_KS
_WO_B_NUM_N = D // PROJ_B_MM_N_TILE
_WO_B_NUM_K = O_LORA_TOTAL // B_K_TILE
_WO_B_SCALE_ROWS = _WO_B_NUM_N * _WO_B_NUM_K * _B_KS
assert _A_K_CHUNKS == 16  # pl.unroll literal in proj_a_mm
assert _B_K_CHUNKS == 4   # pl.unroll literal in proj_b_mm
assert _WO_A_SCALE_ROWS_PER_G == PA_NFRAGS * O_GROUP_IN_SCALE
assert _WO_B_SCALE_ROWS == _WO_B_NUM_N * O_LORA_TOTAL_SCALE
# Token tiles (ts0) are sequential; slots = groups × N-frags/D-chunks × K-chunks.
_A_SLOTS = O_GROUPS * PA_NFRAGS * _A_K_CHUNKS
_B_SLOTS = PB_DCHUNKS * O_GROUPS * _B_K_CHUNKS
_MX_WS_SLOTS = _A_SLOTS + _B_SLOTS

# Sparse K split into <=3 merge blocks of PREFILL_ATTN_TILE rows.
PREFILL_ATTN_TILE = 128
PREFILL_ATTN_BLOCKS = (PREFILL_SPARSE_TOPK + PREFILL_ATTN_TILE - 1) // PREFILL_ATTN_TILE
PREFILL_SPARSE_PAD = PREFILL_ATTN_BLOCKS * PREFILL_ATTN_TILE
# Softmax mask bias: match decode's -1e20 (not most-negative fp32).
NEG_INF = -1.0e20

@pl.jit.inline
def prefill_sparse_attn(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.FP8E4M3FN],
    cmp_kv_scale: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, KV_SCALE_COLS], pl.FP32],
    cmp_block_table: pl.Tensor[[CMP_MAX_BLOCKS], pl.INT32],
    cmp_indices: pl.Tensor[[T, IDX_TOPK], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[O_GROUPS, _WO_A_SCALE_ROWS_PER_G, PROJ_A_MM_N_TILE], pl.FP8E8M0],
    wo_b: pl.Tensor[[O_LORA_TOTAL, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[_WO_B_SCALE_ROWS, PROJ_B_MM_N_TILE], pl.FP8E8M0],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    """Gather cache-first SWA/compressed rows, then run sparse attention and o-proj."""
    # Additive softmax bias (0 valid / NEG_INF invalid). Window bias can be
    # vectorized from INT32 indices; compressed bias cannot — vector cast of
    # cmp_indices → bias wrongly marks t>0 slots NEG_INF on A5 (same bug as
    # decode). Init WIN:PREFILL_SPARSE_PAD to NEG_INF; gather_kv fills compressed
    # lanes via scalar writes.
    sparse_bias = pl.create_tensor([T, PREFILL_SPARSE_PAD], dtype=pl.FP32)
    with pl.spmd(T // BIAS_TOKEN_TILE, name_hint="build_bias") as bias_tid:
        bias_blk = pl.tile.get_block_idx()
        bias_t0 = bias_blk * BIAS_TOKEN_TILE
        bias_win_idx = pl.cast(swa_indices[bias_t0:bias_t0 + BIAS_TOKEN_TILE, 0:WIN], target_type=pl.FP32)
        bias_win_raw_flag = pl.minimum(pl.maximum(pl.add(bias_win_idx, 1.0), 0.0), 1.0)
        sparse_bias[bias_t0:bias_t0 + BIAS_TOKEN_TILE, 0:WIN] = pl.mul(
            pl.sub(bias_win_raw_flag, 1.0), -NEG_INF)
        # Compressed bias: do NOT vector-cast cmp_indices (A5 marks t>0 wrong). Init
        # NEG_INF; gather_kv fills 0.0 / NEG_INF via scalar pl.write.
        sparse_bias[bias_t0:bias_t0 + BIAS_TOKEN_TILE, WIN:PREFILL_SPARSE_PAD] = pl.full(
            [BIAS_TOKEN_TILE, PREFILL_SPARSE_PAD - WIN], dtype=pl.FP32, value=NEG_INF)

    # Gather KV per token into sparse_kv. Row-at-a-time stores (no [ATTN_TILE, HEAD_DIM]
    # UB stage) — staging the full 128x512 BF16 tile overflows A5 UB once scalar bias
    # writes / C8 dequant temps share the same vector scope. Invalid slots stay zero;
    # compressed C8 rows are dequantized to BF16 here (存8算16); ori_kv stays BF16.
    ori_kv_flat = pl.reshape(ori_kv, [ORI_MAX_BLOCKS * BLOCK_SIZE, HEAD_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    cmp_kv_scale_flat = pl.reshape(cmp_kv_scale, [CMP_BLOCK_NUM * BLOCK_SIZE, KV_SCALE_COLS])
    sparse_kv = pl.create_tensor([T * PREFILL_SPARSE_PAD, HEAD_DIM], dtype=pl.BF16)
    with pl.spmd(
        ((T + GATHER_TOKEN_TILE - 1) // GATHER_TOKEN_TILE) * PREFILL_ATTN_BLOCKS,
        name_hint="gather_kv",
        deps=[bias_tid],
    ) as gather_tid:
        gather_block = pl.tile.get_block_idx()
        gather_token_block = gather_block // PREFILL_ATTN_BLOCKS
        gather_sb = gather_block - gather_token_block * PREFILL_ATTN_BLOCKS
        gather_t0 = gather_token_block * GATHER_TOKEN_TILE
        gather_k0 = gather_sb * PREFILL_ATTN_TILE
        # Per-row C8→BF16: expand group scales [1, KV_SCALE_COLS]→[1, HEAD_DIM].
        g_scale_idx = pl.cast(
            pl.div(
                pl.cast(pl.arange(0, [1, HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32),
                64.0,
            ),
            target_type=pl.INT32,
            mode="trunc",
        )
        zero_row = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
        for gather_dt in pl.range(GATHER_TOKEN_TILE):
            gather_t = gather_t0 + gather_dt
            if gather_t < T:
                block_base = gather_t * PREFILL_SPARSE_PAD + gather_k0
                for gather_ki in pl.range(PREFILL_ATTN_TILE):
                    gather_k = gather_k0 + gather_ki
                    dst = block_base + gather_ki
                    if gather_t < num_tokens:
                        if gather_k < WIN:
                            gather_raw = pl.read(swa_indices, [gather_t, gather_k])
                            if gather_raw >= 0:
                                src = pl.cast(gather_raw, pl.INDEX)
                                sparse_kv[dst:dst + 1, :] = ori_kv_flat[src:src + 1, :]
                            else:
                                sparse_kv[dst:dst + 1, :] = zero_row
                        else:
                            gather_cmp_k = gather_k - WIN
                            if gather_cmp_k < IDX_TOPK:
                                gather_raw = pl.read(cmp_indices, [gather_t, gather_cmp_k])
                                if gather_raw >= 0:
                                    pl.write(sparse_bias, [gather_t, gather_k], 0.0)
                                    cmp_slot = gather_raw
                                    blk_slot = cmp_slot // BLOCK_SIZE
                                    blk = pl.cast(pl.read(cmp_block_table, [blk_slot]), pl.INDEX)
                                    src = blk * BLOCK_SIZE + (cmp_slot - blk_slot * BLOCK_SIZE)
                                    g_row_fp8 = cmp_kv_flat[src:src + 1, :]
                                    g_row_sc = cmp_kv_scale_flat[src:src + 1, :]
                                    g_sc_exp = pl.gather(g_row_sc, dim=-1, index=g_scale_idx)
                                    g_dq = pl.mul(pl.cast(g_row_fp8, target_type=pl.FP32), g_sc_exp)
                                    sparse_kv[dst:dst + 1, :] = pl.cast(
                                        g_dq, target_type=pl.BF16, mode="rint"
                                    )
                                else:
                                    pl.write(sparse_bias, [gather_t, gather_k], NEG_INF)
                                    sparse_kv[dst:dst + 1, :] = zero_row
                            else:
                                sparse_kv[dst:dst + 1, :] = zero_row
                    else:
                        sparse_kv[dst:dst + 1, :] = zero_row

    # Block OUTER (one KV/bias load per block), head-batch INNER (QK_M_TILE rows per matmul,
    # sliced to HEAD_TILE stores). qk_kv_k/qk_kv_v are two views so QK (b_trans) and PV don't
    # collide (#1532). Invalid blocks carry a -inf bias and die via beta == 0 in the merge.
    # sparse_blk_* are the per-(token, head-tile, block) softmax stats for that merge.
    blk_rows = T * (H // HEAD_TILE) * PREFILL_ATTN_BLOCKS * HEAD_TILE
    sparse_blk_mi = pl.create_tensor([blk_rows, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([blk_rows, 1], dtype=pl.FP32)
    sparse_blk_oi = pl.create_tensor([blk_rows, HEAD_DIM], dtype=pl.FP32)
    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    with pl.spmd(T, name_hint="qk_pv", deps=[gather_tid]) as qk_tid:
        qk_t = pl.tile.get_block_idx()
        if qk_t < num_tokens:
            qk_kv_base = qk_t * PREFILL_SPARSE_PAD
            qk_token_base = qk_t * (H // HEAD_TILE) * PREFILL_ATTN_BLOCKS * HEAD_TILE
            for qk_sb in pl.range(PREFILL_ATTN_BLOCKS):
                qk_s0 = qk_kv_base + qk_sb * PREFILL_ATTN_TILE
                qk_kv_k = sparse_kv[qk_s0:qk_s0 + PREFILL_ATTN_TILE, :]
                qk_kv_v = sparse_kv[qk_s0:qk_s0 + PREFILL_ATTN_TILE, :]
                qk_bias_row = sparse_bias[qk_t:qk_t + 1, qk_sb * PREFILL_ATTN_TILE:qk_sb * PREFILL_ATTN_TILE + PREFILL_ATTN_TILE]
                for qk_hb in pl.pipeline(H // QK_M_TILE, stage=2):
                    qk_head_row = qk_t * H + qk_hb * QK_M_TILE
                    qk_q_tile = q_flat[qk_head_row:qk_head_row + QK_M_TILE, :]
                    qk_raw = pl.matmul(qk_q_tile, qk_kv_k, b_trans=True, out_dtype=pl.FP32)
                    # Broadcast-add the per-block bias directly (col_expand_add) instead of
                    # col_expand into a dead pl.full(0) base + a separate add (mirrors decode).
                    qk_scaled = pl.mul(qk_raw, SOFTMAX_SCALE)
                    qk_scores = pl.col_expand_add(qk_scaled, qk_bias_row)
                    qk_mi = pl.maximum(pl.row_max(qk_scores), -1.0e20)
                    qk_exp = pl.exp(pl.row_expand_sub(qk_scores, qk_mi))
                    # li sums the FP32 exp; only the PV matmul uses the BF16 cast.
                    qk_li = pl.row_sum(qk_exp)
                    qk_exp_bf16 = pl.cast(qk_exp, target_type=pl.BF16, mode="rint")
                    qk_oi = pl.matmul(qk_exp_bf16, qk_kv_v, out_dtype=pl.FP32)
                    for qk_sub in pl.unroll(QK_M_TILE // HEAD_TILE):
                        qk_h_idx = qk_hb * (QK_M_TILE // HEAD_TILE) + qk_sub
                        qk_r0 = qk_sub * HEAD_TILE
                        qk_row = qk_token_base + qk_h_idx * PREFILL_ATTN_BLOCKS * HEAD_TILE + qk_sb * HEAD_TILE
                        sparse_blk_mi[qk_row:qk_row + HEAD_TILE, :] = qk_mi[qk_r0:qk_r0 + HEAD_TILE, :]
                        sparse_blk_li[qk_row:qk_row + HEAD_TILE, :] = qk_li[qk_r0:qk_r0 + HEAD_TILE, :]
                        sparse_blk_oi[qk_row:qk_row + HEAD_TILE, :] = qk_oi[qk_r0:qk_r0 + HEAD_TILE, :]

    # Precompute head-invariant interleaved cos / sign*sin in ROPE_GATHER_T_TILE-row
    # gathers (A5 tgather wide-box bug). Same HALF_ROPE→ROPE_DIM expand as decode.
    rope_cos_il = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_cs") as rope_tid:
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

    # Online-softmax merge across blocks, sink-norm, fused inverse RoPE (decode HCA style).
    # One spmd block per (token, head-tile) so inv-RoPE gathers stay at ROPE_GATHER_T_TILE
    # head-rows and UB stays under the A5 wall (unlike fusing into a T-way all-heads loop).
    o_packed = pl.create_tensor([O_GROUPS * T, O_GROUP_IN], dtype=pl.BF16)
    with pl.spmd(T * (H // HEAD_TILE), name_hint="merge_norm", deps=[qk_tid, rope_tid]) as merge_tid:
        m_idx = pl.tile.get_block_idx()
        m_t = m_idx // (H // HEAD_TILE)
        m_h_idx = m_idx - m_t * (H // HEAD_TILE)
        m_h0 = m_h_idx * HEAD_TILE
        m_token_base = m_t * (H // HEAD_TILE) * PREFILL_ATTN_BLOCKS * HEAD_TILE
        if m_t < num_tokens:
            m_blk_base = m_token_base + m_h_idx * PREFILL_ATTN_BLOCKS * HEAD_TILE
            m_mi = sparse_blk_mi[m_blk_base:m_blk_base + HEAD_TILE, :]
            m_li = sparse_blk_li[m_blk_base:m_blk_base + HEAD_TILE, :]
            m_oi = sparse_blk_oi[m_blk_base:m_blk_base + HEAD_TILE, :]
            if PREFILL_ATTN_BLOCKS > 1:
                for m_sb in pl.range(1, PREFILL_ATTN_BLOCKS):
                    m_row = m_blk_base + m_sb * HEAD_TILE
                    cur_mi = sparse_blk_mi[m_row:m_row + HEAD_TILE, :]
                    cur_li = sparse_blk_li[m_row:m_row + HEAD_TILE, :]
                    cur_oi = sparse_blk_oi[m_row:m_row + HEAD_TILE, :]
                    mi_new = pl.maximum(m_mi, cur_mi)
                    alpha = pl.exp(pl.sub(m_mi, mi_new))
                    beta = pl.exp(pl.sub(cur_mi, mi_new))
                    m_li = pl.add(pl.mul(alpha, m_li), pl.mul(beta, cur_li))
                    m_oi = pl.add(pl.row_expand_mul(m_oi, alpha), pl.row_expand_mul(cur_oi, beta))
                    m_mi = mi_new
            sink_bias = pl.reshape(attn_sink[m_h0:m_h0 + HEAD_TILE], [HEAD_TILE, 1])
            sink_tile = pl.add(pl.sub(m_mi, m_mi), sink_bias)
            denom = pl.add(m_li, pl.exp(pl.sub(sink_tile, m_mi)))
            n_full = pl.row_expand_div(m_oi, denom)[0:HEAD_TILE, :]
            n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode="rint")
        else:
            n_full = pl.full([HEAD_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
            n_bf16 = pl.full([HEAD_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)

        # Inverse RoPE gather in ROPE_GATHER_T_TILE-row subtiles (head rows, not tokens).
        m_col = pl.col_expand_mul(
            pl.full([ROPE_GATHER_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0),
            pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32))
        m_dup_f = pl.cast(pl.cast(pl.mul(m_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        m_lane = pl.sub(m_col, pl.mul(m_dup_f, 2.0))
        m_swap_idx = pl.cast(pl.sub(pl.add(m_col, 1.0), pl.mul(m_lane, 2.0)), target_type=pl.INT32)
        m_cos_il = rope_cos_il[m_t : m_t + 1, 0:ROPE_DIM]
        m_sin_signed = rope_sin_signed[m_t : m_t + 1, 0:ROPE_DIM]
        n_rope_bf16 = pl.full([HEAD_TILE, ROPE_DIM], dtype=pl.BF16, value=0.0)
        for m_sub in pl.unroll(HEAD_TILE // ROPE_GATHER_T_TILE):
            m_s0 = m_sub * ROPE_GATHER_T_TILE
            m_rope_s = n_full[m_s0 : m_s0 + ROPE_GATHER_T_TILE, NOPE_DIM:HEAD_DIM]
            m_swapped_s = pl.gather(m_rope_s, dim=-1, index=m_swap_idx)
            m_rot_s = pl.add(
                pl.col_expand_mul(m_rope_s, m_cos_il),
                pl.col_expand_mul(m_swapped_s, m_sin_signed),
            )
            n_rope_bf16[m_s0 : m_s0 + ROPE_GATHER_T_TILE, 0:ROPE_DIM] = pl.cast(
                m_rot_s, target_type=pl.BF16, mode="rint"
            )

        for n_hi in pl.range(HEAD_TILE):
            gh = m_h0 + n_hi
            g = gh // HEADS_PER_GROUP
            pack_row = g * T + m_t
            col = (gh - g * HEADS_PER_GROUP) * HEAD_DIM
            o_packed[pack_row:pack_row + 1, col:col + NOPE_DIM] = n_bf16[n_hi:n_hi + 1, 0:NOPE_DIM]
            o_packed[pack_row:pack_row + 1, col + NOPE_DIM:col + HEAD_DIM] = n_rope_bf16[
                n_hi:n_hi + 1, 0:ROPE_DIM
            ]

    # ====================================================================================
    # Grouped MXFP8 output projection, decoupled per-group:
    # proj_a_mm (dyn MX @ wo_a) -> proj_b_mm (dyn MX @ wo_b) -> proj_b_act (sum FP32).
    # Peel K=0 with matmul_mx (init Acc); remaining via matmul_mx_acc (A5 Acc init bug).
    # Right scales use tiled MX_B_NN (same as decode); flat [K/32, N] mis-scales on A5.
    # ====================================================================================
    o_r = pl.create_tensor([T, O_LORA_TOTAL], dtype=pl.FP32)
    partials = pl.create_tensor([T, O_GROUPS * D], dtype=pl.FP32)
    proj_a_tids = pl.array.create(O_GROUPS * PA_NFRAGS, pl.TASK_ID)
    proj_b_tids = pl.array.create(PB_DCHUNKS * O_GROUPS, pl.TASK_ID)
    wo_a_kn = pl.reshape(wo_a, [O_GROUPS * O_GROUP_IN, O_LORA])
    wo_a_scale_flat = pl.reshape(
        wo_a_scale, [O_GROUPS * _WO_A_SCALE_ROWS_PER_G, PROJ_A_MM_N_TILE]
    )
    mx_scale_ws = pl.create_tensor(
        [_MX_WS_SLOTS * A_T_TILE, A_K_TILE // MX_BLOCK_K], dtype=pl.FP8E8M0
    )

    with pl.manual_scope():
        for g in pl.parallel(O_GROUPS):
            row_base_o = g * T
            out_col_g = g * O_LORA
            col_g = g * O_LORA
            wa_row0 = g * O_GROUP_IN
            for nf in pl.range(PA_NFRAGS):
                n0 = nf * PROJ_A_MM_N_TILE
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_a_mm", deps=[merge_tid]) as pa_tid:
                    for ts0 in pl.range(0, T, A_T_TILE):
                        # Peel K=0 with matmul_mx (init Acc); remaining via matmul_mx_acc.
                        k0 = 0
                        xa_tile = pl.load(
                            o_packed,
                            [row_base_o + ts0, k0],
                            [A_T_TILE, A_K_TILE],
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
                        ) * A_T_TILE
                        la = pl.move(
                            pl.move(pl.tile.reinterpret_view(xa_q, pl.FP8E4M3FN), target_memory=pl.Mem.Mat),
                            target_memory=pl.Mem.Left,
                        )
                        la = pl.set_validshape(la, A_T_TILE, A_K_TILE)
                        pl.store(pl.tile.reinterpret_view(xa_s, pl.FP8E8M0), [srow, 0], mx_scale_ws)
                        las = pl.move(
                            pl.load(
                                mx_scale_ws,
                                [srow, 0],
                                [A_T_TILE, A_K_TILE // MX_BLOCK_K],
                                target_memory=pl.Mem.Mat,
                                mx_layout="mx_a_zz",
                            ),
                            target_memory=pl.Mem.LeftScale,
                        )
                        las = pl.tget_scale_addr(las, la)
                        las = pl.set_validshape(las, A_T_TILE, A_K_TILE // MX_BLOCK_K)
                        rb = pl.move(wa_tile, target_memory=pl.Mem.Right)
                        rbs = pl.move(was_tile, target_memory=pl.Mem.RightScale)
                        rbs = pl.tget_scale_addr(rbs, rb)
                        acc_a = pl.matmul_mx(la, las, rb, rbs)
                        for db in pl.unroll(15):  # _A_K_CHUNKS - 1
                            k0 = (db + 1) * A_K_TILE
                            xa_tile2 = pl.load(
                                o_packed,
                                [row_base_o + ts0, k0],
                                [A_T_TILE, A_K_TILE],
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
                            ) * A_T_TILE
                            la2 = pl.move(
                                pl.move(pl.tile.reinterpret_view(xa_q2, pl.FP8E4M3FN), target_memory=pl.Mem.Mat),
                                target_memory=pl.Mem.Left,
                            )
                            la2 = pl.set_validshape(la2, A_T_TILE, A_K_TILE)
                            pl.store(pl.tile.reinterpret_view(xa_s2, pl.FP8E8M0), [srow2, 0], mx_scale_ws)
                            las2 = pl.move(
                                pl.load(
                                    mx_scale_ws,
                                    [srow2, 0],
                                    [A_T_TILE, A_K_TILE // MX_BLOCK_K],
                                    target_memory=pl.Mem.Mat,
                                    mx_layout="mx_a_zz",
                                ),
                                target_memory=pl.Mem.LeftScale,
                            )
                            las2 = pl.tget_scale_addr(las2, la2)
                            las2 = pl.set_validshape(las2, A_T_TILE, A_K_TILE // MX_BLOCK_K)
                            rb2 = pl.move(wa_tile2, target_memory=pl.Mem.Right)
                            rbs2 = pl.move(was_tile2, target_memory=pl.Mem.RightScale)
                            rbs2 = pl.tget_scale_addr(rbs2, rb2)
                            acc_a = pl.matmul_mx_acc(acc_a, la2, las2, rb2, rbs2)
                        pl.store(acc_a, [ts0, out_col_g + n0], o_r)
                proj_a_tids[g * PA_NFRAGS + nf] = pa_tid

        for dc in pl.parallel(PB_DCHUNKS):
            d0 = dc * PROJ_B_D_CHUNK
            for g in pl.range(O_GROUPS):
                col_g = g * O_LORA
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="proj_b_mm",
                    deps=[proj_a_tids[g * PA_NFRAGS + j] for j in range(PA_NFRAGS)],
                ) as pb_tid:
                    for nf in pl.range(PROJ_B_D_CHUNK // PROJ_B_MM_N_TILE):
                        n0 = d0 + nf * PROJ_B_MM_N_TILE
                        nb = n0 // PROJ_B_MM_N_TILE
                        for ts0 in pl.range(0, T, B_T_TILE):
                            # Peel K=0 with matmul_mx (init Acc); remaining via matmul_mx_acc.
                            k0 = col_g
                            or_tile = pl.load(
                                o_r,
                                [ts0, k0],
                                [B_T_TILE, B_K_TILE],
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
                                + (dc * O_GROUPS + g) * _B_K_CHUNKS
                                + (k0 - col_g) // B_K_TILE
                            ) * B_T_TILE
                            bl = pl.move(
                                pl.move(pl.tile.reinterpret_view(or_q, pl.FP8E4M3FN), target_memory=pl.Mem.Mat),
                                target_memory=pl.Mem.Left,
                            )
                            bl = pl.set_validshape(bl, B_T_TILE, B_K_TILE)
                            pl.store(pl.tile.reinterpret_view(or_s, pl.FP8E8M0), [srow, 0], mx_scale_ws)
                            bls = pl.move(
                                pl.load(
                                    mx_scale_ws,
                                    [srow, 0],
                                    [B_T_TILE, B_K_TILE // MX_BLOCK_K],
                                    target_memory=pl.Mem.Mat,
                                    mx_layout="mx_a_zz",
                                ),
                                target_memory=pl.Mem.LeftScale,
                            )
                            bls = pl.tget_scale_addr(bls, bl)
                            bls = pl.set_validshape(bls, B_T_TILE, B_K_TILE // MX_BLOCK_K)
                            br = pl.move(wb_tile, target_memory=pl.Mem.Right)
                            brs = pl.move(wbs_tile, target_memory=pl.Mem.RightScale)
                            brs = pl.tget_scale_addr(brs, br)
                            acc_b = pl.matmul_mx(bl, bls, br, brs)
                            for db in pl.unroll(3):  # _B_K_CHUNKS - 1
                                k0 = col_g + (db + 1) * B_K_TILE
                                or_tile2 = pl.load(
                                    o_r,
                                    [ts0, k0],
                                    [B_T_TILE, B_K_TILE],
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
                                    + (dc * O_GROUPS + g) * _B_K_CHUNKS
                                    + (db + 1)
                                ) * B_T_TILE
                                bl2 = pl.move(
                                    pl.move(pl.tile.reinterpret_view(or_q2, pl.FP8E4M3FN), target_memory=pl.Mem.Mat),
                                    target_memory=pl.Mem.Left,
                                )
                                bl2 = pl.set_validshape(bl2, B_T_TILE, B_K_TILE)
                                pl.store(pl.tile.reinterpret_view(or_s2, pl.FP8E8M0), [srow2, 0], mx_scale_ws)
                                bls2 = pl.move(
                                    pl.load(
                                        mx_scale_ws,
                                        [srow2, 0],
                                        [B_T_TILE, B_K_TILE // MX_BLOCK_K],
                                        target_memory=pl.Mem.Mat,
                                        mx_layout="mx_a_zz",
                                    ),
                                    target_memory=pl.Mem.LeftScale,
                                )
                                bls2 = pl.tget_scale_addr(bls2, bl2)
                                bls2 = pl.set_validshape(bls2, B_T_TILE, B_K_TILE // MX_BLOCK_K)
                                br2 = pl.move(wb_tile2, target_memory=pl.Mem.Right)
                                brs2 = pl.move(wbs_tile2, target_memory=pl.Mem.RightScale)
                                brs2 = pl.tget_scale_addr(brs2, br2)
                                acc_b = pl.matmul_mx_acc(acc_b, bl2, bls2, br2, brs2)
                            pl.store(acc_b, [ts0, g * D + n0], partials)
                proj_b_tids[dc * O_GROUPS + g] = pb_tid

    with pl.spmd(PB_ACT_NREG * PB_ACT_TBLKS, name_hint="proj_b_act",
                 deps=[proj_b_tids[i] for i in range(PB_DCHUNKS * O_GROUPS)]) as act_tid:
        act_idx = pl.tile.get_block_idx()
        nreg = act_idx // PB_ACT_TBLKS
        tblk = act_idx - nreg * PB_ACT_TBLKS
        ob_n0 = nreg * PROJ_B_ACT_N_TILE
        t0 = tblk * PROJ_B_ACT_TBLK
        for b_tb in pl.range(t0, t0 + PROJ_B_ACT_TBLK, PROJ_B_ACT_T_TILE):
            acc = pl.full([PROJ_B_ACT_T_TILE, PROJ_B_ACT_N_TILE], dtype=pl.FP32, value=0.0)
            for g in pl.range(O_GROUPS):
                p_g = partials[b_tb:b_tb + PROJ_B_ACT_T_TILE, g * D + ob_n0:g * D + ob_n0 + PROJ_B_ACT_N_TILE]
                acc = pl.add(acc, p_g)
            attn_out[b_tb:b_tb + PROJ_B_ACT_T_TILE, ob_n0:ob_n0 + PROJ_B_ACT_N_TILE] = pl.cast(
                acc, target_type=pl.BF16, mode="rint"
            )

    return attn_out

@pl.jit
def prefill_sparse_attn_test(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.FP8E4M3FN],
    cmp_kv_scale: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, KV_SCALE_COLS], pl.FP32],
    cmp_block_table: pl.Tensor[[CMP_MAX_BLOCKS], pl.INT32],
    cmp_indices: pl.Tensor[[T, IDX_TOPK], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[O_GROUPS, _WO_A_SCALE_ROWS_PER_G, PROJ_A_MM_N_TILE], pl.FP8E8M0],
    wo_b: pl.Tensor[[O_LORA_TOTAL, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[_WO_B_SCALE_ROWS, PROJ_B_MM_N_TILE], pl.FP8E8M0],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    return prefill_sparse_attn(
        q,
        ori_kv,
        swa_indices,
        cmp_kv,
        cmp_kv_scale,
        cmp_block_table,
        cmp_indices,
        attn_sink,
        num_tokens,
        freqs_cos,
        freqs_sin,
        wo_a,
        wo_a_scale,
        wo_b,
        wo_b_scale,
        attn_out,
    )

def _golden_o_proj_mx(o_model, wo_a, wo_a_scale, wo_b, wo_b_scale):
    """Grouped MLAEpilog o_a + o_b via dyn MX + mx_matmul_fp8 (AscendC Hybrid)."""
    import torch

    def _b_scale_a(s):
        return unpack_scale_b_nn_tiled(
            s,
            k_tile_rows=_A_KS,
            n_tile=PROJ_A_MM_N_TILE,
            logical_k=O_GROUP_IN_SCALE,
            logical_n=O_LORA,
        )

    def _b_scale_b(s):
        return unpack_scale_b_nn_tiled(
            s,
            k_tile_rows=_B_KS,
            n_tile=PROJ_B_MM_N_TILE,
            logical_k=O_LORA_TOTAL_SCALE,
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
        o_r[:, g, :] = mx_matmul_act_tiled(
            o_model[:, g, :], wo_a[g], _b_scale_a(wo_a_scale[g]), A_K_TILE
        )

    out = torch.zeros(t_total, D, dtype=torch.float32)
    wb_s = _b_scale_b(wo_b_scale)
    for g in range(O_GROUPS):
        col0 = g * O_LORA
        out += mx_matmul_act_tiled(
            o_r[:, g, :],
            wo_b[col0 : col0 + O_LORA, :],
            wb_s[col0 // MX_BLOCK_K : (col0 + O_LORA) // MX_BLOCK_K, :],
            B_K_TILE,
        )
    return out

def golden_prefill_sparse_attn(tensors):
    """Self-contained torch reference for the cache-first sparse-attn entry."""
    import torch

    num_tokens = int(tensors["num_tokens"])
    q = tensors["q"].float()
    ori_kv = tensors["ori_kv"].float()
    cmp_kv_fp8 = tensors["cmp_kv"]
    cmp_kv_scale = tensors["cmp_kv_scale"]
    cmp_kv = dequant_kv_c8_fp32_scale(
        cmp_kv_fp8.view(-1, HEAD_DIM),
        cmp_kv_scale.view(-1, KV_SCALE_COLS),
    ).view(CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
    cmp_block_table = tensors["cmp_block_table"]
    swa_indices = tensors["swa_indices"]
    cmp_indices = tensors["cmp_indices"]
    attn_sink = tensors["attn_sink"].float()
    cos = tensors["freqs_cos"].float()
    sin = tensors["freqs_sin"].float()
    wo_a = tensors["wo_a"]
    wo_a_scale = tensors["wo_a_scale"]
    wo_b = tensors["wo_b"]
    wo_b_scale = tensors["wo_b_scale"]

    # Padded sparse-K layout matching the kernel (WIN | IDX_TOPK | pad), not a
    # dense pack of only-valid rows — tile boundaries and online-softmax merge
    # must match PREFILL_ATTN_TILE blocks or compressed slots disagree on A5.
    o = torch.zeros(T, H, HEAD_DIM)
    ori_flat = ori_kv.reshape(-1, HEAD_DIM)
    for t in range(num_tokens):
        kv_rows = []
        valid = []
        for row_i in swa_indices[t].tolist():
            row = int(row_i)
            if row >= 0:
                kv_rows.append(ori_flat[row])
                valid.append(True)
            else:
                kv_rows.append(torch.zeros(HEAD_DIM, dtype=ori_flat.dtype))
                valid.append(False)
        cmp_cols = PREFILL_SPARSE_PAD - WIN
        for cmp_k in range(cmp_cols):
            if cmp_k >= IDX_TOPK:
                kv_rows.append(torch.zeros(HEAD_DIM, dtype=ori_flat.dtype))
                valid.append(False)
                continue
            cmp_slot = int(cmp_indices[t, cmp_k].item())
            if cmp_slot < 0 or cmp_slot >= CMP_MAX_BLOCKS * BLOCK_SIZE:
                kv_rows.append(torch.zeros(HEAD_DIM, dtype=ori_flat.dtype))
                valid.append(False)
                continue
            block_id = int(cmp_block_table[cmp_slot // BLOCK_SIZE].item())
            intra = cmp_slot % BLOCK_SIZE
            if block_id >= 0:
                kv_rows.append(cmp_kv[block_id, intra, 0])
                valid.append(True)
            else:
                kv_rows.append(torch.zeros(HEAD_DIM, dtype=ori_flat.dtype))
                valid.append(False)

        if not any(valid):
            continue
        kv_b = torch.stack(kv_rows, dim=0)
        valid_b = torch.tensor(valid, dtype=torch.bool)

        block_mi, block_li, block_oi = [], [], []
        for tile_start in range(0, PREFILL_SPARSE_PAD, PREFILL_ATTN_TILE):
            kv_tile = kv_b[tile_start : tile_start + PREFILL_ATTN_TILE]
            valid_tile = valid_b[tile_start : tile_start + PREFILL_ATTN_TILE]
            scores = (q[t] @ kv_tile.T) * SOFTMAX_SCALE
            scores = scores.masked_fill(~valid_tile.unsqueeze(0), NEG_INF)
            cur_mi = scores.max(dim=-1, keepdim=True).values
            cur_mi = torch.maximum(cur_mi, torch.tensor(-1.0e20))
            exp_scores = torch.exp(scores - cur_mi).masked_fill(~valid_tile.unsqueeze(0), 0.0)
            cur_li = exp_scores.sum(dim=-1, keepdim=True)
            cur_oi = exp_scores.to(torch.bfloat16).float() @ kv_tile.to(torch.bfloat16).float()
            block_mi.append(cur_mi)
            block_li.append(cur_li)
            block_oi.append(cur_oi)

        mi, li, oi = block_mi[0], block_li[0], block_oi[0]
        for cur_mi, cur_li, cur_oi in zip(block_mi[1:], block_li[1:], block_oi[1:]):
            mi_new = torch.maximum(mi, cur_mi)
            alpha = torch.exp(mi - mi_new)
            beta = torch.exp(cur_mi - mi_new)
            li = alpha * li + beta * cur_li
            oi = oi * alpha + cur_oi * beta
            mi = mi_new
        denom = li + torch.exp(attn_sink.unsqueeze(-1) - mi)
        o[t] = oi / denom

    rope_pair = o[..., NOPE_DIM:].unflatten(-1, (-1, 2))
    rope_even = rope_pair[..., 0]
    rope_odd = rope_pair[..., 1]
    cos_half = cos[:, :ROPE_HALF].unsqueeze(1)
    sin_half = sin[:, :ROPE_HALF].unsqueeze(1)
    inv_even = (rope_even * cos_half + rope_odd * sin_half).to(torch.bfloat16).float()
    inv_odd = (rope_odd * cos_half - rope_even * sin_half).to(torch.bfloat16).float()
    o_rope = torch.stack([inv_even, inv_odd], dim=-1).flatten(-2)
    o = torch.cat([o[..., :NOPE_DIM], o_rope], dim=-1).to(torch.bfloat16)

    o_model = o.float().view(T, O_GROUPS, O_GROUP_IN)
    out = _golden_o_proj_mx(o_model, wo_a, wo_a_scale, wo_b, wo_b_scale)
    tensors["attn_out"][:] = out.to(torch.bfloat16)

def get_prefill_cmp_valid(compress_ratio: int) -> int:
    """Map standalone ratio modes to visible compressed-cache length."""
    if compress_ratio == 0:
        return 0
    if compress_ratio in (4, 128):
        return min(IDX_TOPK, S // compress_ratio, CMP_MAX_BLOCKS * BLOCK_SIZE)
    raise ValueError(f"Unsupported compress_ratio={compress_ratio}; expected one of {SUPPORTED_COMPRESS_RATIOS}")

def build_tensor_specs(compress_ratio: int = DEFAULT_COMPRESS_RATIO):
    import torch
    from golden import ScalarSpec, TensorSpec
    from rope_tables import build_deepseek_v4_rope_tables, materialize_token_rope_tables

    num_tokens = T
    cmp_valid = get_prefill_cmp_valid(compress_ratio)
    shared_freqs_cos, shared_freqs_sin = build_deepseek_v4_rope_tables(M, compress_ratio, dtype=torch.bfloat16)
    shared_rope_cos, shared_rope_sin = materialize_token_rope_tables(
        shared_freqs_cos,
        shared_freqs_sin,
        torch.arange(T, dtype=torch.int32),
    )

    def init_q():
        return ((torch.rand(T, H, HEAD_DIM) - 0.5) * 0.05).to(torch.bfloat16)
    def init_ori_kv():
        return ((torch.rand(ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM) - 0.5) * 0.05).to(torch.bfloat16)
    def init_cmp_kv():
        return cmp_kv_fp8
    def init_cmp_kv_scale():
        return cmp_kv_scale_fp32
    # Pre-quantize demo cmp_kv once so fp8 + scale stay paired.
    _cmp_bf16 = ((torch.rand(CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM) - 0.5) * 0.05).to(torch.bfloat16)
    cmp_kv_fp8 = torch.empty(CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.float8_e4m3fn)
    cmp_kv_scale_fp32 = torch.empty(CMP_BLOCK_NUM, BLOCK_SIZE, 1, KV_SCALE_COLS, dtype=torch.float32)
    for _b in range(CMP_BLOCK_NUM):
        for _r in range(BLOCK_SIZE):
            _q, _s = golden_kv_c8_quant_row_fp32_scale(_cmp_bf16[_b, _r, 0])
            cmp_kv_fp8[_b, _r, 0] = _q
            cmp_kv_scale_fp32[_b, _r, 0] = _s
    def init_cmp_block_table():
        table = torch.zeros(CMP_MAX_BLOCKS, dtype=torch.int32)
        for blk in range(CMP_MAX_BLOCKS):
            table[blk] = blk
        return table
    def init_swa_indices():
        idx = torch.full((T, WIN), -1, dtype=torch.int32)
        for t in range(num_tokens):
            window_start = max(0, t - WIN + 1)
            window = torch.arange(window_start, t + 1, dtype=torch.int32)
            idx[t, :window.numel()] = window
        return idx
    def init_cmp_indices():
        idx = torch.full((T, IDX_TOPK), -1, dtype=torch.int32)
        if compress_ratio:
            for t in range(num_tokens):
                comp_count = min(cmp_valid, (t + 1) // compress_ratio, IDX_TOPK)
                if comp_count > 0:
                    idx[t, :comp_count] = torch.arange(comp_count, dtype=torch.int32)
        return idx
    def init_attn_sink():
        return torch.zeros(H)
    def init_freqs_cos():
        return shared_rope_cos.clone()
    def init_freqs_sin():
        return shared_rope_sin.clone()
    wo_a_list = []
    wo_a_scale_list = []
    for _g in range(O_GROUPS):
        wa, was = gen_mxfp8_weight_kn(
            (O_GROUP_IN, O_LORA),
            dequant_std=0.1,
            chan_cv=0.50,
            n_tile=PROJ_A_MM_N_TILE,
            k_tile=A_K_TILE,
        )
        wo_a_list.append(wa)
        wo_a_scale_list.append(was)
    wo_a_mx = torch.stack(wo_a_list, dim=0)
    wo_a_scale_mx = torch.stack(wo_a_scale_list, dim=0)
    wo_b_mx, wo_b_scale_mx = gen_mxfp8_weight_kn(
        (O_LORA_TOTAL, D),
        dequant_std=0.1,
        chan_cv=0.50,
        n_tile=PROJ_B_MM_N_TILE,
        k_tile=B_K_TILE,
    )

    def init_wo_a():
        return wo_a_mx
    def init_wo_a_scale():
        return wo_a_scale_mx
    def init_wo_b():
        return wo_b_mx
    def init_wo_b_scale():
        return wo_b_scale_mx

    return [
        TensorSpec("q", [T, H, HEAD_DIM], torch.bfloat16, init_value=init_q),
        TensorSpec("ori_kv", [ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_ori_kv),
        TensorSpec("swa_indices", [T, WIN], torch.int32, init_value=init_swa_indices),
        TensorSpec("cmp_kv", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.float8_e4m3fn, init_value=init_cmp_kv),
        TensorSpec("cmp_kv_scale", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, KV_SCALE_COLS], torch.float32, init_value=init_cmp_kv_scale),
        TensorSpec("cmp_block_table", [CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("cmp_indices", [T, IDX_TOPK], torch.int32, init_value=init_cmp_indices),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
        TensorSpec("freqs_cos", [T, ROPE_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [T, ROPE_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("wo_a", [O_GROUPS, O_GROUP_IN, O_LORA], torch.float8_e4m3fn, init_value=init_wo_a),
        TensorSpec(
            "wo_a_scale",
            [O_GROUPS, _WO_A_SCALE_ROWS_PER_G, PROJ_A_MM_N_TILE],
            torch.float8_e8m0fnu,
            init_value=init_wo_a_scale,
        ),
        TensorSpec("wo_b", [O_LORA_TOTAL, D], torch.float8_e4m3fn, init_value=init_wo_b),
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
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--compress-ratio", type=int, default=DEFAULT_COMPRESS_RATIO,
                        choices=list(SUPPORTED_COMPRESS_RATIOS))
    parser.add_argument("--enable-l2-swimlane", nargs="?", const=4, default=0, type=int)
    parser.add_argument("--enable-pmu", nargs="?", const=2, default=0, type=int, choices=[0, 1, 2, 4])
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    oproj_tol = ATOL_RTOL.get("oproj_mxfp8", ATOL_RTOL["fia_mxfp8"])
    result = run_jit(
        fn=prefill_sparse_attn_test,
        specs=build_tensor_specs(args.compress_ratio),
        golden_fn=golden_prefill_sparse_attn,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
            enable_pmu=args.enable_pmu,
        ),
        rtol=1e-3,
        atol=1e-3,
        compile_only=args.compile_only,
        compare_fn={
            "attn_out": ratio_allclose(
                atol=oproj_tol["atol"],
                rtol=oproj_tol["rtol"],
                max_error_ratio=oproj_tol["pct"],
            ),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
