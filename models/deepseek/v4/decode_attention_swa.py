# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 SWA (Sliding Window Attention) decode orchestration — `compress_ratio == 0` path.
Active in layers 0/1/7 of the model (3 of the 8 layers in demo). No KV compression, so neither
compressor nor indexer is invoked; topk for sparse_attn is window_topk_idxs only and the KV cache
holds only the sliding window (no compressed portion). YaRN frequency scaling is also disabled
in this path (model.py:478-479 selects base rope_theta when compress_ratio==0).
Companion files: attention_csa_draft.py (ratio=4)
                 attention_hca_draft.py (ratio=128)."""


import pypto.language as pl

from config import (
    FLASH as M,
    DECODE_BATCH,
    DECODE_ORI_BLOCK_NUM,
    DECODE_SEQ,
    BLOCK_SIZE,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
    KV_CMP_MAX_BLOCKS,
    KV_ORI_MAX_BLOCKS,
    KV_ORI_TABLE_MAX_BLOCKS,
)
from hc_pre import hc_pre
from hc_post import hc_post
from decode_sparse_attn_swa import sparse_attn_swa


# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
EPS = M.rms_norm_eps
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
NOPE_HEAD_DIM = M.nope_head_dim
Q_LORA = M.q_lora_rank
WIN = M.sliding_window
SOFTMAX_SCALE = M.softmax_scale
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
MAX_SEQ_LEN = M.max_position_embeddings
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS

# kernel-local (SWA: ratio-0, no compressor/indexer)
ORI_MAX_BLOCKS = KV_ORI_MAX_BLOCKS
ORI_TABLE_MAX_BLOCKS = KV_ORI_TABLE_MAX_BLOCKS
ORI_BLOCK_NUM = DECODE_ORI_BLOCK_NUM
TOPK = WIN                          # SWA: sparse_attn topk = window only
SPARSE_IDX_TOPK = M.index_topk      # sparse_attn module's IDX_TOPK (static shape contract)
SPARSE_TOPK = WIN + SPARSE_IDX_TOPK
SPARSE_CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS

# tiling
SPARSE_ROPE_TILE = 16
SPARSE_ROPE_INTERLEAVE_TILE = 2 * SPARSE_ROPE_TILE
NEG_INF = -1.0e20

ROPE_DIM = ROPE_HEAD_DIM
ROPE_HALF = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim

# QKV projection and RMSNorm tiling
Q_PROJ_TILE = 128
QPROJ_MM_N_TILE = 1024
Q_LORA_TILE = 256
KV_TILE = 64
QUANT_TILE = 256
T_TILE = 8
MATMUL_T_TILE = 16
T_MAX = max(T, MATMUL_T_TILE)
QR_M_TILE = MATMUL_T_TILE
QR_N_TILE = 128
QR_K_TILE = 256
QR_OK = 2
QR_K_SLICE = D // QR_OK
KV_M_TILE = MATMUL_T_TILE
KV_N_TILE = 128
KV_K_TILE = 256
KV_OK = 4
KV_K_SLICE = D // KV_OK
QPROJ_M_TILE = MATMUL_T_TILE
KV_RMS_T_TILE = 8
Q_ROPE_T_TILE = 8
Q_ROPE_H_TILE = 4
D_TILE = 128
assert H % Q_ROPE_H_TILE == 0
assert T % T_TILE == 0
assert T <= MATMUL_T_TILE
assert Q_LORA % QR_N_TILE == 0 and D % QR_OK == 0 and QR_K_SLICE % QR_K_TILE == 0
assert HEAD_DIM % KV_N_TILE == 0 and D % KV_OK == 0 and KV_K_SLICE % KV_K_TILE == 0
assert (H * HEAD_DIM) % QPROJ_MM_N_TILE == 0 and ((H * HEAD_DIM) // QPROJ_MM_N_TILE) % 4 == 0
assert Q_LORA % Q_PROJ_TILE == 0 and QPROJ_MM_N_TILE * QPROJ_M_TILE * 4 <= 128 * 1024
assert T % KV_RMS_T_TILE == 0
assert T % Q_ROPE_T_TILE == 0
assert D % D_TILE == 0, "D must be divisible by D_TILE"
# attn_norm_xgamma is pure elementwise x*gamma (no reduction), so it parallelizes
# to one token per core. A wide D-tile keeps the per-token pipeline short.
XG_D_TILE = 1024
assert D % XG_D_TILE == 0


@pl.jit.inline
def attn_norm_xgamma(
    x: pl.Tensor[[T, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    xg: pl.Tensor[[T, D], pl.BF16],
):
    # Deferred attention RMSNorm for the qkv-only path (qwen3/gate-style): the
    # per-token inv_rms is a positive scalar, and qkv_proj_rope re-RMSNorms BOTH
    # the qr and kv projections downstream, so inv_rms cancels exactly. We store
    # only xg = x*gamma and skip the sq_sum reduction + rsqrt + normalize pass
    # entirely. Valid ONLY where x's sole consumer is qkv (SWA); the CSA/HCA
    # compressor softmax gate is scale-sensitive and must keep the full rms_norm.
    # One token per core (pl.spmd): pure elementwise x*gamma has no reduction, so
    # each core just writes its own [1, XG_D_TILE] rows -- no 32B reduction-tile floor.
    t_dim = pl.tensor.dim(x, 0)
    norm_w_2d = pl.reshape(norm_w, [1, D])
    with pl.spmd(t_dim, name_hint="attn_norm_xgamma", allow_early_resolve=True) as xg_tid:
        tok = pl.tile.get_block_idx()
        for xg_db in pl.pipeline(D // XG_D_TILE, stage=2):
            xg_d0 = xg_db * XG_D_TILE
            xg_x_chunk = pl.cast(x[tok : tok + 1, xg_d0 : xg_d0 + XG_D_TILE], target_type=pl.FP32)
            norm_w_chunk = pl.cast(norm_w_2d[0:1, xg_d0 : xg_d0 + XG_D_TILE], target_type=pl.FP32)
            xg[tok : tok + 1, xg_d0 : xg_d0 + XG_D_TILE] = pl.cast(
                pl.col_expand_mul(xg_x_chunk, norm_w_chunk),
                target_type=pl.BF16,
                mode="rint",
            )

    return xg_tid


@pl.jit.inline
def qkv_proj_rope_swa(
    x: pl.Tensor[[T, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    rope_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[T, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    t_dim = pl.tensor.dim(x, 0)
    x_view = pl.reshape(x, [t_dim, D])
    rope_cos_view = pl.reshape(rope_cos, [t_dim, ROPE_DIM])
    rope_sin_view = pl.reshape(rope_sin, [t_dim, ROPE_DIM])
    kv_view = pl.reshape(kv, [t_dim, HEAD_DIM])
    qr_view = pl.reshape(qr, [t_dim, Q_LORA])
    qr_scale_view = pl.reshape(qr_scale, [t_dim, 1])
    t_matmul = pl.max(t_dim, MATMUL_T_TILE)

    # RoPE indices and interleaved cos/signed-sin rows are head-invariant.
    # Prepare them once per token tile so the 16 Q head-group tasks do not each
    # rebuild the same arange/cast/gather chain on their critical AIV path.
    q_rope_cos_il = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    q_rope_sin_signed = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    q_rope_swap_idx = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.INT32)
    for qrp_idx in pl.spmd(t_dim // Q_ROPE_T_TILE, name_hint="q_rope_prepare", allow_early_resolve=True):
        qrp_t0 = qrp_idx * Q_ROPE_T_TILE
        qrp_ones = pl.full([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        qrp_idx_i32 = pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32)
        qrp_idx_fp32 = pl.cast(qrp_idx_i32, target_type=pl.FP32)
        qrp_col = pl.col_expand_mul(qrp_ones, qrp_idx_fp32)
        qrp_half = pl.mul(qrp_col, 0.5)
        qrp_dup_i32 = pl.cast(qrp_half, target_type=pl.INT32, mode="trunc")
        qrp_dup_f = pl.cast(qrp_dup_i32, target_type=pl.FP32)
        qrp_dup_idx = pl.cast(qrp_dup_f, target_type=pl.INT32)
        qrp_lane = pl.sub(qrp_col, pl.mul(qrp_dup_f, 2.0))
        qrp_next_col = pl.add(qrp_col, 1.0)
        qrp_lane_offset = pl.mul(qrp_lane, 2.0)
        qrp_swap_f = pl.sub(qrp_next_col, qrp_lane_offset)
        qrp_swap_idx = pl.cast(qrp_swap_f, target_type=pl.INT32)
        qrp_sign = pl.sub(pl.mul(qrp_lane, 2.0), 1.0)
        qrp_cos_rows = rope_cos_view[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :]
        qrp_sin_rows = rope_sin_view[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :]
        qrp_cos = pl.cast(qrp_cos_rows, target_type=pl.FP32)
        qrp_sin = pl.cast(qrp_sin_rows, target_type=pl.FP32)
        qrp_cos_il = pl.gather(qrp_cos, dim=-1, index=qrp_dup_idx)
        qrp_sin_il = pl.gather(qrp_sin, dim=-1, index=qrp_dup_idx)
        qrp_sin_signed = pl.mul(qrp_sin_il, qrp_sign)
        q_rope_cos_il[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :] = qrp_cos_il
        q_rope_sin_signed[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :] = qrp_sin_signed
        q_rope_swap_idx[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :] = qrp_swap_idx

    # Split-K qr_proj (M=t_dim, K=D=4096, N=Q_LORA=1024). QR_N_TILE=128 gives
    # eight N-groups; QR_OK=2 expands them to 16 cube blocks and atomic-adds the
    # K partials into a zero-seeded output. Auto-dep on qr_fp32 orders the seed
    # before every atomic RMW.
    qr_fp32 = pl.create_tensor([T_MAX, Q_LORA], dtype=pl.FP32)
    qr_i8_matmul = pl.create_tensor([T_MAX, Q_LORA], dtype=pl.INT8)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_proj_seed"):
        for tc in pl.range(t_matmul // QR_M_TILE):
            ts0 = tc * QR_M_TILE
            for nb in pl.range(Q_LORA // QR_N_TILE):
                nseed0 = nb * QR_N_TILE
                qr_fp32[ts0 : ts0 + QR_M_TILE, nseed0 : nseed0 + QR_N_TILE] = pl.full(
                    [QR_M_TILE, QR_N_TILE], dtype=pl.FP32, value=0.0
                )
    for qbg_idx in pl.spmd((Q_LORA // QR_N_TILE) * QR_OK, name_hint="qr_proj_matmul", allow_early_resolve=True):
        q_a_col0 = (qbg_idx // QR_OK) * QR_N_TILE
        qr_k_base = (qbg_idx % QR_OK) * QR_K_SLICE
        for tc in pl.range(t_matmul // QR_M_TILE):
            t0 = tc * QR_M_TILE
            q_acc = pl.create_tensor([QR_M_TILE, QR_N_TILE], dtype=pl.FP32)
            for db in pl.pipeline(QR_K_SLICE // QR_K_TILE, stage=2):
                qr_d0 = qr_k_base + db * QR_K_TILE
                qr_rows = pl.min(QR_M_TILE, t_dim - t0)
                q_x_chunk_bf16 = pl.slice(x_view, [QR_M_TILE, QR_K_TILE], [t0, qr_d0], valid_shape=[qr_rows, QR_K_TILE])
                w_chunk = wq_a[qr_d0 : qr_d0 + QR_K_TILE, q_a_col0 : q_a_col0 + QR_N_TILE]
                if db == 0:
                    q_acc = pl.matmul(q_x_chunk_bf16, w_chunk, out_dtype=pl.FP32)
                else:
                    q_acc = pl.matmul_acc(q_acc, q_x_chunk_bf16, w_chunk)
            qr_fp32 = pl.assemble(qr_fp32, q_acc, [t0, q_a_col0], atomic=pl.AtomicType.Add)

    # Two passes per block: pass 1 computes amax; pass 2 recomputes norm and quantizes.
    for tg_idx in pl.spmd(t_dim // T_TILE, name_hint="qr_rms_norm_quant", allow_early_resolve=True):
        tg = tg_idx * T_TILE
        qr_sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        qr_amax_g = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for qr_rms_qb in pl.pipeline(Q_LORA // Q_LORA_TILE, stage=2):
            qr_rms_col0 = qr_rms_qb * Q_LORA_TILE
            qr_rms_chunk = qr_fp32[tg : tg + T_TILE, qr_rms_col0 : qr_rms_col0 + Q_LORA_TILE]
            qr_sq_sum = pl.add(qr_sq_sum, pl.reshape(pl.row_sum(pl.mul(qr_rms_chunk, qr_rms_chunk)), [1, T_TILE]))
            gamma_rms_cast = pl.cast(gamma_cq[qr_rms_col0 : qr_rms_col0 + Q_LORA_TILE], target_type=pl.FP32)
            gamma_rms_chunk = pl.reshape(gamma_rms_cast, [1, Q_LORA_TILE])
            qr_g = pl.col_expand_mul(qr_rms_chunk, gamma_rms_chunk)
            qr_g_abs = pl.abs(qr_g)
            qr_amax_g = pl.maximum(qr_amax_g, pl.reshape(pl.row_max(qr_g_abs), [1, T_TILE]))
        qr_inv_rms = pl.rsqrt(pl.add(pl.mul(qr_sq_sum, 1.0 / Q_LORA), EPS), high_precision=True)
        qr_inv_rms_t = pl.reshape(qr_inv_rms, [T_TILE, 1])
        qr_amax_floor = pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        qr_amax_normed = pl.mul(qr_inv_rms, qr_amax_g)
        qr_tile_amax = pl.maximum(qr_amax_floor, qr_amax_normed)

        qr_scale_quant_row = pl.div(pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), qr_tile_amax)
        qr_scale_quant_t = pl.reshape(qr_scale_quant_row, [T_TILE, 1])
        qr_tile_scale_dq = pl.reshape(pl.recip(qr_scale_quant_row), [T_TILE, 1])
        qr_scale_view[tg : tg + T_TILE, :] = qr_tile_scale_dq

        for qa in pl.pipeline(0, Q_LORA, QUANT_TILE, stage=2):
            qr_chunk = qr_fp32[tg : tg + T_TILE, qa : qa + QUANT_TILE]
            gamma_q_cast = pl.cast(gamma_cq[qa : qa + QUANT_TILE], target_type=pl.FP32)
            gamma_q_chunk = pl.reshape(gamma_q_cast, [1, QUANT_TILE])
            qr_q_normed = pl.col_expand_mul(pl.row_expand_mul(qr_chunk, qr_inv_rms_t), gamma_q_chunk)
            qr_q_scaled = pl.row_expand_mul(qr_q_normed, qr_scale_quant_t)
            qr_q_i32 = pl.cast(qr_q_scaled, target_type=pl.INT32, mode="rint")
            qr_q_half = pl.cast(qr_q_i32, target_type=pl.FP16, mode="round")
            qr_q_i8 = pl.cast(qr_q_half, target_type=pl.INT8, mode="trunc")
            qr_view[tg : tg + T_TILE, qa : qa + QUANT_TILE] = qr_q_i8
            qr_i8_matmul[tg : tg + T_TILE, qa : qa + QUANT_TILE] = qr_q_i8

    # UN-MIXED qproj: keep the pure-matmul scope (cube, INT32 -> GM) separate from
    # downstream vector work. This lets the scheduler defer q dequant until AIV is free
    # instead of pinning it next to qproj and competing with the critical qr_proj AIV work.
    q_proj_i32 = pl.create_tensor([T_MAX, H * HEAD_DIM], dtype=pl.INT32)
    for hg_idx in pl.spmd(((H * HEAD_DIM) // QPROJ_MM_N_TILE) // 2, name_hint="qproj_matmul", allow_early_resolve=True):
        hg = hg_idx * 2
        for h_inner in pl.range(2):
            w_col0 = (hg + h_inner) * QPROJ_MM_N_TILE
            for tc in pl.range(t_matmul // QPROJ_M_TILE):
                t0 = tc * QPROJ_M_TILE
                col_acc = pl.create_tensor([QPROJ_M_TILE, QPROJ_MM_N_TILE], dtype=pl.INT32)
                for qb in pl.pipeline(0, Q_LORA // Q_PROJ_TILE, stage=2):
                    qr_proj_col0 = qb * Q_PROJ_TILE
                    qr_i8_chunk = qr_i8_matmul[t0 : t0 + QPROJ_M_TILE, qr_proj_col0 : qr_proj_col0 + Q_PROJ_TILE]
                    wq_chunk = wq_b[qr_proj_col0 : qr_proj_col0 + Q_PROJ_TILE, w_col0 : w_col0 + QPROJ_MM_N_TILE]
                    if qr_proj_col0 == 0:
                        col_acc = pl.matmul(qr_i8_chunk, wq_chunk, out_dtype=pl.INT32)
                    else:
                        col_acc = pl.matmul_acc(col_acc, qr_i8_chunk, wq_chunk)
                q_proj_i32[t0 : t0 + QPROJ_M_TILE, w_col0 : w_col0 + QPROJ_MM_N_TILE] = col_acc

    # Fuse qproj dequant, per-head RMSNorm, NOPE writeback, and interleaved RoPE.
    # A full [token, head] tile fits in Vec UB, so dequantize each head once and
    # retain it across the RMS reduction instead of rereading/recomputing NOPE.
    # RoPE: out[j] = inv_rms * (x[j] * cos[j] + x[j^1] * sign[j] * sin[j]).
    q_flat = pl.reshape(q, [t_dim, H * HEAD_DIM])
    for hg_idx in pl.spmd(H // Q_ROPE_H_TILE, name_hint="qproj_dequant_rms_nope_rope", allow_early_resolve=True):
        hg = hg_idx * Q_ROPE_H_TILE
        for tg_idx in pl.range(t_dim // Q_ROPE_T_TILE):
            tg = tg_idx * Q_ROPE_T_TILE
            qr_scale_dq_t = qr_scale_view[tg : tg + Q_ROPE_T_TILE, :]
            q_cos_il = q_rope_cos_il[tg : tg + Q_ROPE_T_TILE, :]
            q_sin_signed = q_rope_sin_signed[tg : tg + Q_ROPE_T_TILE, :]
            q_swap_idx = q_rope_swap_idx[tg : tg + Q_ROPE_T_TILE, :]
            # Pipeline adjacent heads so the next head's GM reads overlap the
            # current head's vector RMS/rotation work, as in Qwen's decode loop.
            for h_inner in pl.pipeline(Q_ROPE_H_TILE, stage=2):
                h = hg + h_inner
                h0 = h * HEAD_DIM
                q_head_acc = q_proj_i32[tg : tg + Q_ROPE_T_TILE, h0 : h0 + HEAD_DIM]
                q_head_scale = pl.reshape(wq_b_scale[h0 : h0 + HEAD_DIM], [1, HEAD_DIM])
                q_head_acc_fp32 = pl.cast(q_head_acc, target_type=pl.FP32, mode="none")
                q_head_row_scaled = pl.row_expand_mul(q_head_acc_fp32, qr_scale_dq_t)
                q_head_dq = pl.col_expand_mul(q_head_row_scaled, q_head_scale)
                q_head_sq = pl.mul(q_head_dq, q_head_dq)
                q_head_sq_row = pl.row_sum(q_head_sq)
                q_head_sq_sum = pl.reshape(q_head_sq_row, [1, Q_ROPE_T_TILE])
                q_head_sq_mean = pl.mul(q_head_sq_sum, 1.0 / HEAD_DIM)
                q_head_var = pl.add(q_head_sq_mean, EPS)
                q_head_inv_rms = pl.rsqrt(q_head_var, high_precision=True)
                q_head_inv_rms_t = pl.reshape(q_head_inv_rms, [Q_ROPE_T_TILE, 1])

                q_nope_normed = pl.row_expand_mul(q_head_dq[:, 0:NOPE_DIM], q_head_inv_rms_t)
                q_nope_bf16 = pl.cast(q_nope_normed, target_type=pl.BF16, mode="rint")
                q_flat[tg : tg + Q_ROPE_T_TILE, h0 : h0 + NOPE_DIM] = q_nope_bf16

                # RoPE writeback on columns [h0+NOPE_DIM:h0+HEAD_DIM), inv_rms folded after.
                q_rope_chunk = q_head_dq[:, NOPE_DIM:HEAD_DIM]
                q_rope_swapped = pl.gather(q_rope_chunk, dim=-1, index=q_swap_idx)
                q_rope_base = pl.mul(q_rope_chunk, q_cos_il)
                q_rope_delta = pl.mul(q_rope_swapped, q_sin_signed)
                q_rope_rot = pl.add(q_rope_base, q_rope_delta)
                q_rope_normed = pl.row_expand_mul(q_rope_rot, q_head_inv_rms_t)
                q_rope_bf16 = pl.cast(q_rope_normed, target_type=pl.BF16, mode="rint")
                q_flat[tg : tg + Q_ROPE_T_TILE, h0 + NOPE_DIM : h0 + NOPE_DIM + ROPE_DIM] = q_rope_bf16

    # Split-K kv_proj uses four 128-column N-groups and KV_OK=4, again producing
    # 16 cube blocks. KV is off the critical path, so more K splits only add atomic
    # contention without shortening decode.
    kv_fp32 = pl.create_tensor([T_MAX, HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_proj_seed"):
        for tc in pl.range(t_matmul // KV_M_TILE):
            kts0 = tc * KV_M_TILE
            for nb in pl.range(HEAD_DIM // KV_N_TILE):
                kvseed0 = nb * KV_N_TILE
                kv_fp32[kts0 : kts0 + KV_M_TILE, kvseed0 : kvseed0 + KV_N_TILE] = pl.full(
                    [KV_M_TILE, KV_N_TILE], dtype=pl.FP32, value=0.0
                )
    # `late_dep` is a dummy barrier hung off the rms_norm TaskId: kv_proj is off the
    # critical path, so it resolves one hop after rms_norm and lets qr_proj_matmul
    # take the cores first.
    with pl.spmd((HEAD_DIM // KV_N_TILE) * KV_OK, name_hint="kv_proj_matmul", deps=[late_dep]) as _kv_tid:
        kbg = pl.tile.get_block_idx()
        kv_col0 = (kbg // KV_OK) * KV_N_TILE
        kv_k_base = (kbg % KV_OK) * KV_K_SLICE
        for tc in pl.range(t_matmul // KV_M_TILE):
            t0 = tc * KV_M_TILE
            kv_acc = pl.create_tensor([KV_M_TILE, KV_N_TILE], dtype=pl.FP32)
            for db in pl.pipeline(KV_K_SLICE // KV_K_TILE, stage=2):
                d0 = kv_k_base + db * KV_K_TILE
                kv_rows = pl.min(KV_M_TILE, t_dim - t0)
                kv_x_chunk_bf16 = pl.slice(x_view, [KV_M_TILE, KV_K_TILE], [t0, d0], valid_shape=[kv_rows, KV_K_TILE])
                wkv_chunk = wkv[d0 : d0 + KV_K_TILE, kv_col0 : kv_col0 + KV_N_TILE]
                if db == 0:
                    kv_acc = pl.matmul(kv_x_chunk_bf16, wkv_chunk, out_dtype=pl.FP32)
                else:
                    kv_acc = pl.matmul_acc(kv_acc, kv_x_chunk_bf16, wkv_chunk)
            kv_fp32 = pl.assemble(kv_fp32, kv_acc, [t0, kv_col0], atomic=pl.AtomicType.Add)

    # Fused KV RMSNorm + interleaved (CANN A3) RoPE. One spmd task per [KV_RMS_T_TILE, HEAD_DIM]
    # row block computes the per-row inv_rms once (pass 1) and consumes it locally for
    # BOTH the NOPE writeback and the rope rotation -- so inv_rms no longer round-trips
    # through GM (the old kv_inv_rms_tensor) and the two passes collapse into a single
    # dispatch. NOPE columns [0:NOPE_DIM) and rope columns [NOPE_DIM:HEAD_DIM) are
    # disjoint, so each task writes a clean, conflict-free row block of kv. Vec UB stays
    # well under the 192 KB cap (chunks are at most [KV_RMS_T_TILE, KV_TILE] fp32).
    for tg_idx in pl.spmd(t_dim // KV_RMS_T_TILE, name_hint="kv_rms_norm_rope"):
        tg = tg_idx * KV_RMS_T_TILE
        # Pass 1: per-row sum of squares over the full HEAD_DIM -> inv_rms.
        kv_sq_sum = pl.full([1, KV_RMS_T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HEAD_DIM // KV_TILE, stage=2):
            kv_sq_col0 = kb * KV_TILE
            kv_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, kv_sq_col0 : kv_sq_col0 + KV_TILE]
            kv_sq_sum = pl.add(kv_sq_sum, pl.reshape(pl.row_sum(pl.mul(kv_chunk, kv_chunk)), [1, KV_RMS_T_TILE]))
        kv_inv_rms = pl.rsqrt(pl.add(pl.mul(kv_sq_sum, 1.0 / HEAD_DIM), EPS), high_precision=True)
        kv_inv_rms_t = pl.reshape(kv_inv_rms, [KV_RMS_T_TILE, 1])

        # NOPE writeback: rms-normalize columns [0:NOPE_DIM) with per-column gamma.
        for nb in pl.pipeline(NOPE_DIM // KV_TILE, stage=2):
            n0 = nb * KV_TILE
            kv_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, n0 : n0 + KV_TILE]
            gamma_kv_cast = pl.cast(gamma_ckv[n0 : n0 + KV_TILE], target_type=pl.FP32)
            gamma_kv_chunk = pl.reshape(gamma_kv_cast, [1, KV_TILE])
            kv_normed = pl.col_expand_mul(pl.row_expand_mul(kv_chunk, kv_inv_rms_t), gamma_kv_chunk)
            kv_view[tg : tg + KV_RMS_T_TILE, n0 : n0 + KV_TILE] = pl.cast(kv_normed, target_type=pl.BF16, mode="rint")

        # RoPE writeback on columns [NOPE_DIM:HEAD_DIM), interleaved (CANN A3) swap-gather
        # (same form as qproj_dequant_rms_nope_rope), built in-kernel. inv_rms (per-row, the same
        # factor used for NOPE above) and gamma (per-column, full ROPE_DIM) are folded into
        # kv_rope_norm_chunk BEFORE the swap so the swapped lane n[j^1] carries gamma[j^1]
        # (gamma does NOT commute with the rotation; inv_rms does).
        #   out[j] = n[j]*cos_il[j] + n[j^1]*sign[j]*sin_il[j]
        gamma_rope_cast = pl.cast(gamma_ckv[NOPE_DIM : NOPE_DIM + ROPE_DIM], target_type=pl.FP32)
        gamma_rope = pl.reshape(gamma_rope_cast, [1, ROPE_DIM])
        kv_rope_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, NOPE_DIM : NOPE_DIM + ROPE_DIM]
        kv_rope_norm_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_rope_chunk, kv_inv_rms_t), gamma_rope)
        kv_ones = pl.full([KV_RMS_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        kv_col = pl.col_expand_mul(kv_ones, pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32))
        kv_dup_f = pl.cast(pl.cast(pl.mul(kv_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        kv_dup_idx = pl.cast(kv_dup_f, target_type=pl.INT32)                                       # j>>1
        kv_lane = pl.sub(kv_col, pl.mul(kv_dup_f, 2.0))                                            # j%2
        kv_swap_idx = pl.cast(pl.sub(pl.add(kv_col, 1.0), pl.mul(kv_lane, 2.0)), target_type=pl.INT32)  # j^1
        kv_sign = pl.sub(pl.mul(kv_lane, 2.0), 1.0)                                                # [-1,+1,...]
        kv_cos_il = pl.gather(pl.cast(rope_cos_view[tg : tg + KV_RMS_T_TILE, :], target_type=pl.FP32), dim=-1, index=kv_dup_idx)
        kv_sin_il = pl.gather(pl.cast(rope_sin_view[tg : tg + KV_RMS_T_TILE, :], target_type=pl.FP32), dim=-1, index=kv_dup_idx)
        kv_swapped = pl.gather(kv_rope_norm_chunk, dim=-1, index=kv_swap_idx)
        kv_rope_rot = pl.add(pl.mul(kv_rope_norm_chunk, kv_cos_il), pl.mul(pl.mul(kv_swapped, kv_sign), kv_sin_il))
        kv_rope_i16 = pl.cast(kv_rope_rot, target_type=pl.BF16, mode="rint")
        kv_view[tg : tg + KV_RMS_T_TILE, NOPE_DIM : NOPE_DIM + ROPE_DIM] = kv_rope_i16

    return q


@pl.jit.inline
def attention_swa(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    # hc_pre weights
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    # qkv_proj_rope weights
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    # KV cache (sliding-window only: [0, WIN) ori; no cmp portion)
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_slot_mapping: pl.Tensor[[T], pl.INT64],
    swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    swa_lens: pl.Tensor[[T], pl.INT32],
    position_ids: pl.Tensor[[T], pl.INT32],
    # sparse_attn
    attn_sink: pl.Tensor[[H], pl.FP32],
    # o_proj
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Tensor[[T, HC_MULT, D], pl.FP32],
):
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post_t = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb_t = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post_t, comb_t)

    rope_cos_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_rope_step"):
        for b in pl.range(B):
            for s_idx in pl.range(S):
                t = b * S + s_idx
                pos_b = pl.cast(pl.read(position_ids, [t]), pl.INDEX)
                cos_row = pl.cast(freqs_cos[pos_b : pos_b + 1, 0 : ROPE_HEAD_DIM], target_type=pl.FP32)
                sin_row = pl.cast(freqs_sin[pos_b : pos_b + 1, 0 : ROPE_HEAD_DIM], target_type=pl.FP32)
                rope_cos_t[t : t + 1, 0 : ROPE_HEAD_DIM] = pl.cast(cos_row, target_type=pl.BF16, mode="rint")
                rope_sin_t[t : t + 1, 0 : ROPE_HEAD_DIM] = pl.cast(sin_row, target_type=pl.BF16, mode="rint")

    x_normed_t = pl.create_tensor([T, D], dtype=pl.BF16)
    # Deferred attention norm: qkv re-RMSNorms both qr and kv, so the per-token
    # inv_rms cancels. Store only x*gamma; the sq_sum reduction + rsqrt are dropped.
    rms_tid = attn_norm_xgamma(x_mixed, attn_norm_w, x_normed_t)
    # Defers kv_proj_matmul one hop behind the norm so qr_proj_matmul dispatches first.
    late_dep = pl.system.task_dummy(deps=[rms_tid])
    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    qkv_proj_rope_swa(
        x_normed_t, wq_a, wq_b, wq_b_scale, wkv,
        rope_cos_t, rope_sin_t, gamma_cq, gamma_ckv,
        q, kv, qr, qr_scale, late_dep,
    )

    # Commit current decode KV and build its additive padding mask in one task.
    # The SWA attention kernel reads every visible row through metadata-expanded
    # physical cache indices, so all cache writes must complete before it starts.
    kv_cache_flat = pl.reshape(kv_cache, [ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    sparse_bias = pl.create_tensor([T, WIN], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_cache_insert_valid_bias"):
        for write_t in pl.range(T):
            write_row_i64 = pl.read(swa_slot_mapping, [write_t])
            if write_row_i64 >= 0:
                write_row = pl.cast(write_row_i64, pl.INDEX)
                kv_cache_flat[write_row : write_row + 1, 0 : HEAD_DIM] = kv[write_t : write_t + 1, 0 : HEAD_DIM]
        v_col = pl.cast(pl.arange(0, [1, WIN], dtype=pl.INT32), target_type=pl.FP32)
        v_col_m = pl.col_expand(pl.full([T, WIN], dtype=pl.FP32, value=0.0), v_col)
        v_lens = pl.cast(pl.reshape(swa_lens[0:T], [T, 1]), target_type=pl.FP32)
        v_valid = pl.minimum(
            pl.maximum(pl.neg(pl.row_expand_sub(v_col_m, v_lens)), 0.0),
            1.0,
        )
        sparse_bias[0:T, 0:WIN] = pl.mul(pl.sub(v_valid, 1.0), -NEG_INF)
    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    sparse_attn_swa(
        q, kv_cache_flat, swa_indices, sparse_bias,
        attn_sink, rope_cos_t, rope_sin_t,
        wo_a, wo_b, wo_b_scale, attn_out,
    )

    hc_post(attn_out, x_hc, post_t, comb_t, x_out)
    return x_out


@pl.jit
def attention_swa_test(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    # hc_pre weights
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    # qkv_proj_rope weights
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    # KV cache (sliding-window only: [0, WIN) ori; no cmp portion)
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    swa_slot_mapping: pl.Tensor[[T], pl.INT64],
    swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    swa_lens: pl.Tensor[[T], pl.INT32],
    position_ids: pl.Tensor[[T], pl.INT32],
    # sparse_attn
    attn_sink: pl.Tensor[[H], pl.FP32],
    # o_proj
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
):
    attention_swa(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv,
        gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin,
        kv_cache, swa_slot_mapping, swa_indices, swa_lens, position_ids,
        attn_sink,
        wo_a, wo_b, wo_b_scale,
        x_out,
    )
    return x_out


def golden_rms_norm(x, norm_w):
    import torch

    x = x.float()
    norm_w = norm_w.float()
    inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    return (x * inv * norm_w).to(torch.bfloat16)


def golden_qkv_proj_rope(tensors):
    """Torch reference: Q/KV LoRA + RoPE for an already attention-normalized input."""
    import torch

    x = tensors["x"].float()
    wq_a = tensors["wq_a"].float()
    wq_b = tensors["wq_b"]
    wq_b_scale = tensors["wq_b_scale"].float().view(-1)
    wkv = tensors["wkv"].float()
    rope_cos = tensors["rope_cos"].float()
    rope_sin = tensors["rope_sin"].float()
    gamma_cq = tensors["gamma_cq"].float()
    gamma_ckv = tensors["gamma_ckv"].float()

    def int8_quant_per_row(x):
        rows = x.reshape(-1, x.shape[-1]).float()
        amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = rows * scale_quant
        out_i32 = torch.round(scaled).to(torch.int32)
        out_half = out_i32.to(torch.float16)
        out_i8 = out_half.to(torch.int8)
        return out_i8.reshape_as(x), (1.0 / scale_quant).reshape(*x.shape[:-1], 1)

    def rms_norm(x, gamma, eps=EPS):
        inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        return x * inv * gamma

    def matmul_bf16_input_fp32(a, b):
        a_fp32 = a.to(torch.bfloat16).float()
        b_fp32 = b.to(torch.bfloat16).float()
        return torch.matmul(a_fp32, b_fp32).float()

    def apply_rope(x_rope, cos, sin):
        # x_rope: [T, ..., ROPE_DIM] with interleaved even/odd rotary pairs.
        x_pair = x_rope.unflatten(-1, (-1, 2))
        x_even, x_odd = x_pair[..., 0], x_pair[..., 1]
        cos_v = cos[..., :ROPE_HALF]
        sin_v = sin[..., :ROPE_HALF]
        while cos_v.ndim < x_even.ndim:
            cos_v = cos_v.unsqueeze(-2)
            sin_v = sin_v.unsqueeze(-2)
        y_even = (x_even * cos_v - x_odd * sin_v).to(torch.bfloat16)
        y_odd = (x_even * sin_v + x_odd * cos_v).to(torch.bfloat16)
        return torch.stack([y_even, y_odd], dim=-1).flatten(-2)

    t_dim = x.shape[0]
    token_x = x.view(t_dim, D)

    # Q path
    qr_out = rms_norm(matmul_bf16_input_fp32(token_x, wq_a), gamma_cq)   # [T, Q_LORA]
    # W8A8C16: wq_b W8 per-output-channel int8; qr_out A8 per-token int8.
    # flash: also quantizes wq_a/wkv to fp8 (default Linear dtype).
    qr_i8, qr_scale = int8_quant_per_row(qr_out.float())
    q_i32 = torch.matmul(qr_i8.to(torch.int32), wq_b.to(torch.int32))
    q_full = (q_i32.float() * qr_scale * wq_b_scale.view(1, -1)).view(t_dim, H, HEAD_DIM)
    inv = torch.rsqrt(q_full.square().mean(-1, keepdim=True) + EPS)
    q_full = q_full * inv                                            # per-head RMSNorm (no gamma)
    q_nope = q_full[..., :NOPE_DIM]
    q_rope = apply_rope(q_full[..., NOPE_DIM:], rope_cos, rope_sin)
    q_out = torch.cat([q_nope, q_rope], dim=-1)

    # KV path
    kv_full = rms_norm(matmul_bf16_input_fp32(token_x, wkv), gamma_ckv)  # [T, HEAD_DIM]
    kv_nope = kv_full[..., :NOPE_DIM]
    kv_rope_in = kv_full[..., NOPE_DIM:].unsqueeze(1)               # add a pseudo head dim
    kv_rope = apply_rope(kv_rope_in, rope_cos, rope_sin).squeeze(1)
    kv_out = torch.cat([kv_nope, kv_rope], dim=-1)

    tensors["q"][:]  = q_out.to(torch.bfloat16)
    tensors["kv"][:] = kv_out.to(torch.bfloat16)
    tensors["qr"][:] = qr_i8
    tensors["qr_scale"][:] = qr_scale


def golden_attention_swa(tensors):
    """End-to-end orchestration for the ratio=0 (SWA) layers.
    Mirrors Block.hc_pre + Attention.forward (decode branch, ratio==0 path: no compressor,
    no indexer, no cmp_kv) + Block.hc_post."""
    import torch

    from hc_pre import golden_hc_pre
    from decode_sparse_attn_swa import golden_sparse_attn
    from hc_post import golden_hc_post

    # ---- Block.hc_pre (model.py:691) ----
    x_mixed = torch.zeros(T, D, dtype=torch.bfloat16)
    post_t = torch.zeros(T, HC_MULT)
    comb_t = torch.zeros(T, HC_MULT * HC_MULT)
    golden_hc_pre({
        "x": tensors["x_hc"],
        "hc_fn": tensors["hc_attn_fn"],
        "hc_scale": tensors["hc_attn_scale"],
        "hc_base": tensors["hc_attn_base"],
        "x_mixed": x_mixed,
        "post": post_t,
        "comb": comb_t,
    })

    # ===== Attention.forward (model.py:484-543), ratio==0 branch =====
    position_ids = tensors["position_ids"].to(torch.int64)
    bsz, seqlen = B, S
    win = WIN
    rd = ROPE_HEAD_DIM

    freqs_cos = tensors["freqs_cos"]
    freqs_sin = tensors["freqs_sin"]
    rope_cos_T = torch.empty(T, rd, dtype=freqs_cos.dtype)
    rope_sin_T = torch.empty(T, rd, dtype=freqs_sin.dtype)
    for t in range(T):
        pos = int(position_ids[t].item())
        rope_cos_T[t] = freqs_cos[pos]
        rope_sin_T[t] = freqs_sin[pos]

    # q + win kv (model.py:495-504)
    q = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(T, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(T, 1, dtype=torch.float32)
    x_normed = golden_rms_norm(x_mixed, tensors["attn_norm_w"])
    golden_qkv_proj_rope({
        "x": x_normed,
        "wq_a": tensors["wq_a"],
        "wq_b": tensors["wq_b"],
        "wq_b_scale": tensors["wq_b_scale"],
        "wkv": tensors["wkv"],
        "rope_cos": rope_cos_T,
        "rope_sin": rope_sin_T,
        "gamma_cq": tensors["gamma_cq"],
        "gamma_ckv": tensors["gamma_ckv"],
        "q": q,
        "kv": kv,
        "qr": qr,                                                              # qr unused on SWA path
        "qr_scale": qr_scale,
    })

    kv_cache = tensors["kv_cache"]
    attn_out = torch.zeros(T, D, dtype=torch.bfloat16)

    # Current decode KV is visible to SWA through the same physical cache slots
    # that metadata points at.
    swa_slot_mapping = tensors["swa_slot_mapping"].to(torch.int64)
    for t in range(T):
        write_row = int(swa_slot_mapping[t].item())
        if write_row >= 0:
            write_blk = write_row // BLOCK_SIZE
            write_intra = write_row % BLOCK_SIZE
            kv_cache[write_blk, write_intra, 0] = kv[t]

    golden_sparse_attn({
        "q": q,
        "ori_kv": kv_cache,
        "swa_indices": tensors["swa_indices"],
        "swa_lens": tensors["swa_lens"],
        "attn_sink": tensors["attn_sink"],
        "freqs_cos": rope_cos_T,
        "freqs_sin": rope_sin_T,
        "wo_a": tensors["wo_a"],
        "wo_b": tensors["wo_b"],
        "wo_b_scale": tensors["wo_b_scale"],
        "attn_out": attn_out,
    })

    # ===== Block.hc_post (model.py:694) =====
    y = torch.zeros(T, HC_MULT, D, dtype=torch.float32)
    golden_hc_post({
        "x": attn_out,
        "residual": tensors["x_hc"],
        "post": post_t,
        "comb": comb_t,
        "y": y,
    })

    tensors["x_out"][:] = y


def build_tensor_specs(start_pos=None):
    import torch  # type: ignore[import]
    from decode_metadata import (
        block_table,
        paged_slot_mapping,
        position_ids_from_starts,
        resolve_start_positions,
        swa_indices_and_lens,
        swa_decode_start_set,
    )
    from golden import TensorSpec
    from rope_tables import build_deepseek_v4_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_deepseek_v4_rope_tables(M, 0, dtype=torch.bfloat16)

    def quant_w_per_output_channel(w):
        amax = w.float().abs().amax(dim=0).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = w.float() * scale_quant.view(1, H * HEAD_DIM)
        w_i32 = torch.round(scaled).to(torch.int32)
        w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
        w_i8 = w_i32.to(torch.float16).to(torch.int8)
        return w_i8, (1.0 / scale_quant).float()

    def quant_w_per_row(w):
        amax = w.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = w.float() * scale_quant.unsqueeze(-1)
        w_i32 = torch.round(scaled).to(torch.int32)
        w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
        w_i8 = w_i32.to(torch.float16).to(torch.int8)
        return w_i8, (1.0 / scale_quant).float()

    def init_x_hc():
        return torch.empty(T, HC_MULT, D).uniform_(-1, 1)
    # Real layer-0 (SWA) hc_attn scale/base (fn synthetic at real magnitude). A synthetic
    # scale=0.5/base=0 leaves hc_pre post~=1 + near-uniform comb, cancelling attn_out and the
    # hc residual to near-zero in x_out where quant noise blows up the relative tail.
    def init_hc_attn_fn():
        return torch.randn(MIX_HC, HC_DIM) * 0.039
    def init_hc_attn_scale():
        return torch.tensor([2.076026, 0.018729, 0.245936])
    def init_hc_attn_base():
        return torch.tensor([
            3.9083, -2.0399, -2.2033, -2.017,
            -2.4443, -10.3158, -8.9943, -6.3581,
            9.8577, -9.5177, -24.8724, -22.8929,
            -21.545, 0.7791, -3.386, 1.1948,
            -20.9605, -0.7702, 1.4218, -4.8994,
            1.5177, -29.7663, -30.1413, -1.2413,
        ])
    def init_attn_norm_w():
        return torch.ones(D)
    def init_wq_a():
        return torch.randn(D, Q_LORA) / D ** 0.5
    def init_wq_b():
        return torch.randn(Q_LORA, H * HEAD_DIM) / Q_LORA ** 0.5
    def init_wkv():
        return torch.randn(D, HEAD_DIM) / D ** 0.5
    def init_gamma_cq():
        return torch.ones(Q_LORA)
    def init_gamma_ckv():
        return torch.ones(HEAD_DIM)
    def init_freqs_cos():
        return shared_freqs_cos.clone()
    def init_freqs_sin():
        return shared_freqs_sin.clone()
    def init_normalized_cache(shape):
        cache = torch.randn(*shape)
        denom = cache.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(EPS)
        return (cache / denom).to(torch.bfloat16)

    def init_kv_cache():
        return init_normalized_cache((ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM))

    def init_block_table():
        return block_table(batch=B, table_blocks=ORI_TABLE_MAX_BLOCKS, physical_blocks=ORI_MAX_BLOCKS)

    def init_attn_sink():
        return torch.zeros(H)
    def init_default_start_pos():
        # Canonical SWA start-position set (sliding-window regimes + 8k long-context).
        return swa_decode_start_set(batch=B, window=WIN)
    def init_start_pos():
        return resolve_start_positions(
            start_pos,
            batch=B,
            seq=S,
            max_seq_len=MAX_SEQ_LEN,
            default_fn=init_default_start_pos,
        )
    def init_position_ids():
        return position_ids_from_starts(init_start_pos(), seq=S).reshape(-1).contiguous()
    def init_swa_slot_mapping():
        return paged_slot_mapping(
            position_ids_from_starts(init_start_pos(), seq=S),
            init_block_table(),
            block_size=BLOCK_SIZE,
        ).reshape(-1).contiguous()
    def init_swa_metadata():
        return swa_indices_and_lens(
            position_ids_from_starts(init_start_pos(), seq=S),
            init_block_table(),
            block_size=BLOCK_SIZE,
            window=WIN,
        )
    def init_swa_indices():
        return init_swa_metadata()[0].contiguous()
    def init_swa_lens():
        return init_swa_metadata()[1].contiguous()
    def init_wo_a():
        return torch.randn(O_GROUPS, O_LORA, O_GROUP_IN) / O_GROUP_IN ** 0.5
    def init_wo_b():
        return torch.randn(D, O_GROUPS * O_LORA) / (O_GROUPS * O_LORA) ** 0.5

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = quant_w_per_output_channel(wq_b_bf16)
    wo_b_bf16 = init_wo_b().to(torch.bfloat16)
    wo_b_i8, wo_b_scale = quant_w_per_row(wo_b_bf16)

    return [
        TensorSpec("x_hc", [T, HC_MULT, D], torch.float32, init_value=init_x_hc),
        TensorSpec("hc_attn_fn", [MIX_HC, HC_DIM], torch.float32, init_value=init_hc_attn_fn),
        TensorSpec("hc_attn_scale", [3], torch.float32, init_value=init_hc_attn_scale),
        TensorSpec("hc_attn_base", [MIX_HC], torch.float32, init_value=init_hc_attn_base),
        TensorSpec("attn_norm_w", [D], torch.bfloat16, init_value=init_attn_norm_w),
        TensorSpec("wq_a", [D, Q_LORA], torch.bfloat16, init_value=init_wq_a),
        TensorSpec("wq_b", [Q_LORA, H * HEAD_DIM], torch.int8, init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [H * HEAD_DIM], torch.float32, init_value=lambda: wq_b_scale),
        TensorSpec("wkv", [D, HEAD_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("gamma_cq", [Q_LORA], torch.bfloat16, init_value=init_gamma_cq),
        TensorSpec("gamma_ckv", [HEAD_DIM], torch.bfloat16, init_value=init_gamma_ckv),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("kv_cache", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_kv_cache, is_output=True),
        TensorSpec("swa_slot_mapping", [T], torch.int64, init_value=init_swa_slot_mapping),
        TensorSpec("swa_indices", [T, WIN], torch.int32, init_value=init_swa_indices),
        TensorSpec("swa_lens", [T], torch.int32, init_value=init_swa_lens),
        TensorSpec("position_ids", [T], torch.int32, init_value=init_position_ids),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("x_out", [T, HC_MULT, D], torch.float32, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, ratio_reldiff, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--start-pos", type=int, default=None,
                        help="Uniform fixture-only start_pos override for all batches; "
                             "default (unset) uses the canonical per-batch SWA set that includes the 8k point.")
    parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2, 4))
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=attention_swa_test,
        specs=build_tensor_specs(args.start_pos),
        golden_fn=golden_attention_swa,
        runtime_dir=args.runtime_dir,
        golden_data=args.golden_data,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-2,
        atol=1e-2,
        compare_fn={
            # Tightened from CANN's 1e-2 bar: realistic hc_attn gates keep x_out
            # well-conditioned (0% over 3e-3 across seeds; worst rdiff ~0.16).
            "x_out": ratio_reldiff(diff_thd=3e-3, pct_thd=0.008, max_diff_hd=1),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
