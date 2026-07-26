# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Indexer (decode) — Hybrid LI A8C8 FP8+FP32 + MXFP8 q_b.

Mirrors model.py Indexer decode branch (prefill ``start_pos == 0`` omitted).
Precision (AscendC Hybrid MXFP8-MXFP4 Step 5):
  - ``qr`` INT8 + ``qr_scale`` FP32 from QKV Step 4 (unchanged API)
  - ``wq_b`` MXFP8 W8A8 (e4m3 + e8m0 block=32) via ``matmul_mx``
  - ``indexer_q`` after Hadamard: dynamic per-token-head FP8 e4m3 + FP32 scale (max=448)
  - Indexer Cache C8: dynamic per-position FP8 e4m3 + FP32 scale (max=448; not main KV group64)
  - ``weights_proj`` / hadamard / rope / compressor wkv/wgate: BF16 (not quantized)
  - LI score ``batch_matmul``: FP8 activations → FP32 acc, scales applied in reduce
The inner Compressor is invoked via ``indexer_compressor``."""


import pypto.language as pl

from config import (
    FLASH as M,
    DECODE_BATCH,
    DECODE_SEQ,
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    DECODE_IDX_BLOCK_NUM,
    IDX_CACHE_MAX_BLOCKS,
    FP32_NEG_INF,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
    MX_BLOCK_K,
)
from decode_indexer_compressor import indexer_compressor
from mx_quant_common import (
    ATOL_RTOL,
    FP8_E4M3_MAX,
    dynamic_mx_quant_e4m3,
    gen_mxfp8_weight_kn,
    mx_matmul_fp8,
    unpack_scale_b_nn_tiled,
)

# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
Q_LORA = M.q_lora_rank
Q_LORA_SCALE = Q_LORA // MX_BLOCK_K
ROPE_HEAD_DIM = M.qk_rope_head_dim
IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
IDX_NOPE_HEAD_DIM = M.index_nope_head_dim
WEIGHTS_SCALE = M.index_weights_scale
MAX_SEQ_LEN = M.max_position_embeddings
OFFSET = M.sliding_window

# kernel-local
COMPRESS_RATIO = 4   # the indexer only runs on ratio-4 layers
IDX_TOPK = M.index_topk
INNER_OVERLAP = COMPRESS_RATIO == 4
INNER_COFF = 1 + int(INNER_OVERLAP)
INNER_HEAD_DIM = IDX_HEAD_DIM
INNER_OUT_DIM = INNER_COFF * INNER_HEAD_DIM
INNER_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
INNER_STATE_PHYSICAL_BLOCKS = 65
INNER_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + INNER_STATE_BLOCK_SIZE - 1) // INNER_STATE_BLOCK_SIZE
INNER_STATE_BLOCK_NUM = INNER_STATE_PHYSICAL_BLOCKS
INNER_STATE_DIM = 2 * INNER_OUT_DIM

IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
IDX_CACHE_BLOCK_NUM = DECODE_IDX_BLOCK_NUM
SCORE_LEN = IDX_KV_LEN

# tiling
CACHE_TILE = 64
assert BLOCK_SIZE % CACHE_TILE == 0, "CACHE_TILE must not cross a paged idx_kv_cache block"
# matmul/reduce tile over contiguous GM scratch, not the paged KV cache
MAT_TILE = 512
# Keep REDUCE_TILE == BLOCK_SIZE so each reduce iter is one paged block (cb == page).
REDUCE_TILE = 128
assert BLOCK_SIZE % REDUCE_TILE == 0, "REDUCE_TILE must tile the paged idx_kv_cache block"
# score_kv_quant / score_reduce fan the cache-tile loop across NSPLIT extra lanes: T * NSPLIT.
QUANT_NSPLIT = 4
REDUCE_NSPLIT = 4
Q_TILE = 256
assert Q_LORA % MX_BLOCK_K == 0 and Q_TILE % MX_BLOCK_K == 0
# Q_OUT_TILE is the per-task N granularity (sets idx_qr_proj task count); MM_N_TILE
# is the Mat-safe cube N-tile. Q_OUT_TILE fans Q_OUT_TILE // MM_N_TILE cube ops per
# task so task count halves without growing the [Q_TILE, MM_N_TILE] L1 wq load.
Q_OUT_TILE = 1024
MM_N_TILE = 256   # 256×256 FP8 Right = 64KB; was 512 (128KB overflow on A5 MX)
MM_ROW_TILE = 16
T_PAD = ((T + MM_ROW_TILE - 1) // MM_ROW_TILE) * MM_ROW_TILE
# weights_proj is one 16-row boxed matmul per task; decode T fits in one row tile.
# Fail loudly if a config makes T exceed it (would drop rows).
assert T_PAD == MM_ROW_TILE, "weights_proj single-row-tile scope assumes decode T <= MM_ROW_TILE"
HEAD_DIM_TILE = 32
D_TILE = 512
# weights_proj splits K, not N: a [D_TILE, IDX_N_HEADS] row block reads contiguous GM,
# while an N slice would take 32B out of every 128B row. Each task writes its own
# partial row block, summed by a separate reduce scope -- a zero-seed + atomic-add
# assemble races here, since T_PAD == MM_ROW_TILE makes the seed a full-extent write.
# WEIGHTS_K_SLICE // D_TILE == 2, so the inner loop is a pl.range: a degenerate
# 2-iteration pl.pipeline(stage=2) miscompiles over matmul.
WEIGHTS_OK = 4
WEIGHTS_K_SLICE = D // WEIGHTS_OK
assert WEIGHTS_K_SLICE % D_TILE == 0
QH_QUANT_TILE = 64
# cube tile for q @ hadamard; L0C caps it at QH_MM_TILE * IDX_HEAD_DIM * 4B <= 64KiB.
QH_MM_TILE = 64
QH_HEAD_DIM_TILE = 64
ROPE_ROW_BLOCK = S * IDX_N_HEADS
# A5 FP32 tgather corrupts wide boxes (same as sparse-attn / compressor); gather ≤4 rows.
ROPE_ROW_TILE = 4
assert (T * IDX_N_HEADS) % ROPE_ROW_TILE == 0
assert ROPE_ROW_BLOCK % ROPE_ROW_TILE == 0
TOPK_HALF_LEN = SCORE_LEN // 2
TOPK_HALF_PAIR_OFFSET = 2 * TOPK_HALF_LEN
TOPK_PAIR_WIDTH = 2 * IDX_TOPK
assert SCORE_LEN == 2 * TOPK_HALF_LEN, "decode indexer topk expects an even score length"
assert TOPK_HALF_LEN == 2048, "decode indexer 4096-value topk uses two 2048-value halves"
assert IDX_TOPK <= TOPK_HALF_LEN, "per-half candidate list must cover the final topk width"

_QR_SPMD = IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE
_QR_NS = Q_OUT_TILE // MM_N_TILE
_QR_K_CHUNKS = Q_LORA // Q_TILE
_QR_KS = Q_TILE // MX_BLOCK_K
_QR_NUM_N = IDX_N_HEADS * IDX_HEAD_DIM // MM_N_TILE  # == _QR_SPMD * _QR_NS
# Tiled MX_B_NN: each (N-tile, K-chunk) independently convert_x2'd; col offset 0.
_WQ_B_SCALE_ROWS = _QR_NUM_N * _QR_K_CHUNKS * _QR_KS  # == _QR_NUM_N * Q_LORA_SCALE
# ns is sequential (pl.range); slots = SPMD × K-chunks
_MX_WS_SLOTS = _QR_SPMD * _QR_K_CHUNKS
assert _WQ_B_SCALE_ROWS == _QR_NUM_N * Q_LORA_SCALE
assert _QR_K_CHUNKS == 4  # pl.unroll literal in idx_qr_proj


@pl.jit.inline
def indexer(
    x: pl.Tensor[[B, S, D], pl.BF16],
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[_WQ_B_SCALE_ROWS, MM_N_TILE], pl.FP8E8M0],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],  # shared by q rotation and inner Compressor
    inner_kv: pl.Tensor[[B, S, INNER_HEAD_DIM], pl.FP32],
    inner_compress_state: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32],
    inner_compress_state_block_table: pl.Tensor[[B, INNER_STATE_MAX_BLOCKS], pl.INT32],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.BF16],
    # C8 indexer cache: FP8 e4m3 KV (quant-on-write) + per-position FP32 dequant scale.
    idx_kv_cache: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.FP8E4M3FN]],
    idx_kv_scale: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[B, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    score: pl.Tensor[[B, S, SCORE_LEN], pl.FP32],
    topk_idxs: pl.Tensor[[B, S, SCORE_LEN], pl.INT32],
    position_ids: pl.Tensor[[B, S], pl.INT32],
    idx_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
    offset: pl.Scalar[pl.INT32],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    # idx_qr_proj: dequant INT8 qr → dyn MX → matmul_mx wq_b → FP32 (no per-channel scale).
    qr_proj = pl.create_tensor([T, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.FP32)
    mx_scale_ws = pl.create_tensor(
        [_MX_WS_SLOTS * T_PAD, Q_TILE // MX_BLOCK_K], dtype=pl.FP8E8M0
    )
    for ot in pl.spmd(IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE, name_hint="idx_qr_proj_matmul"):
        o_base = ot * Q_OUT_TILE
        for ns in pl.range(0, Q_OUT_TILE, MM_N_TILE):
            w_col0 = o_base + ns
            nb = ot * _QR_NS + ns // MM_N_TILE
            # Peel K=0 with matmul_mx (init Acc); remaining via matmul_mx_acc.
            q0 = 0
            kb = 0
            qr_tile = pl.load(
                qr,
                [0, q0],
                [T_PAD, Q_TILE],
                valid_shapes=[T, Q_TILE],
                target_memory=pl.Mem.Vec,
            )
            # CCEC has no INT8→FP32 castData; go via FP16.
            qr_f = pl.cast(
                pl.cast(qr_tile, target_type=pl.FP16, mode="none"),
                target_type=pl.FP32,
                mode="none",
            )
            qr_scale_v = pl.load(
                qr_scale,
                [0, 0],
                [T_PAD, 1],
                valid_shapes=[T, 1],
                target_memory=pl.Mem.Vec,
            )
            qr_dq = pl.row_expand_mul(qr_f, qr_scale_v)
            qr_q, qr_s = pl.mx_quant(qr_dq, mode="mxfp8_e4m3")
            wq_tile = pl.load(
                wq_b,
                [q0, w_col0],
                [Q_TILE, MM_N_TILE],
                target_memory=pl.Mem.Mat,
            )
            ws_tile = pl.load(
                wq_b_scale,
                [(nb * _QR_K_CHUNKS + kb) * _QR_KS, 0],
                [Q_TILE // MX_BLOCK_K, MM_N_TILE],
                target_memory=pl.Mem.Mat,
                mx_layout="mx_b_nn",
            )
            srow = (ot * _QR_K_CHUNKS + kb) * T_PAD
            qr_la = pl.move(
                pl.move(pl.tile.reinterpret_view(qr_q, pl.FP8E4M3FN), target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.Left,
            )
            qr_la = pl.set_validshape(qr_la, T, Q_TILE)
            pl.store(pl.tile.reinterpret_view(qr_s, pl.FP8E8M0), [srow, 0], mx_scale_ws)
            qr_las = pl.move(
                pl.load(
                    mx_scale_ws,
                    [srow, 0],
                    [T_PAD, Q_TILE // MX_BLOCK_K],
                    target_memory=pl.Mem.Mat,
                    mx_layout="mx_a_zz",
                ),
                target_memory=pl.Mem.LeftScale,
            )
            qr_las = pl.tget_scale_addr(qr_las, qr_la)
            qr_las = pl.set_validshape(qr_las, T, Q_TILE // MX_BLOCK_K)
            wq_rb = pl.move(wq_tile, target_memory=pl.Mem.Right)
            wq_rbs = pl.move(ws_tile, target_memory=pl.Mem.RightScale)
            wq_rbs = pl.tget_scale_addr(wq_rbs, wq_rb)
            qr_acc = pl.matmul_mx(qr_la, qr_las, wq_rb, wq_rbs)
            for db in pl.unroll(3):  # _QR_K_CHUNKS - 1
                q0 = (db + 1) * Q_TILE
                qr_tile2 = pl.load(
                    qr,
                    [0, q0],
                    [T_PAD, Q_TILE],
                    valid_shapes=[T, Q_TILE],
                    target_memory=pl.Mem.Vec,
                )
                qr_f2 = pl.cast(
                    pl.cast(qr_tile2, target_type=pl.FP16, mode="none"),
                    target_type=pl.FP32,
                    mode="none",
                )
                qr_scale_v2 = pl.load(
                    qr_scale,
                    [0, 0],
                    [T_PAD, 1],
                    valid_shapes=[T, 1],
                    target_memory=pl.Mem.Vec,
                )
                qr_dq2 = pl.row_expand_mul(qr_f2, qr_scale_v2)
                qr_q2, qr_s2 = pl.mx_quant(qr_dq2, mode="mxfp8_e4m3")
                wq_tile2 = pl.load(
                    wq_b,
                    [q0, w_col0],
                    [Q_TILE, MM_N_TILE],
                    target_memory=pl.Mem.Mat,
                )
                ws_tile2 = pl.load(
                    wq_b_scale,
                    [(nb * _QR_K_CHUNKS + (db + 1)) * _QR_KS, 0],
                    [Q_TILE // MX_BLOCK_K, MM_N_TILE],
                    target_memory=pl.Mem.Mat,
                    mx_layout="mx_b_nn",
                )
                srow2 = (ot * _QR_K_CHUNKS + (db + 1)) * T_PAD
                qr_la2 = pl.move(
                    pl.move(pl.tile.reinterpret_view(qr_q2, pl.FP8E4M3FN), target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.Left,
                )
                qr_la2 = pl.set_validshape(qr_la2, T, Q_TILE)
                pl.store(pl.tile.reinterpret_view(qr_s2, pl.FP8E8M0), [srow2, 0], mx_scale_ws)
                qr_las2 = pl.move(
                    pl.load(
                        mx_scale_ws,
                        [srow2, 0],
                        [T_PAD, Q_TILE // MX_BLOCK_K],
                        target_memory=pl.Mem.Mat,
                        mx_layout="mx_a_zz",
                    ),
                    target_memory=pl.Mem.LeftScale,
                )
                qr_las2 = pl.tget_scale_addr(qr_las2, qr_la2)
                qr_las2 = pl.set_validshape(qr_las2, T, Q_TILE // MX_BLOCK_K)
                wq_rb2 = pl.move(wq_tile2, target_memory=pl.Mem.Right)
                wq_rbs2 = pl.move(ws_tile2, target_memory=pl.Mem.RightScale)
                wq_rbs2 = pl.tget_scale_addr(wq_rbs2, wq_rb2)
                qr_acc = pl.matmul_mx_acc(qr_acc, qr_la2, qr_las2, wq_rb2, wq_rbs2)
            pl.store(qr_acc, [0, w_col0], qr_proj)

    qr_proj_flat = pl.reshape(qr_proj, [T * IDX_N_HEADS, IDX_HEAD_DIM])
    # BF16 q for the Hadamard matmul: nope half rounded from the FP32 dequant, rope
    # half rotated then rounded.
    qr_bf16 = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.BF16)
    # spmd over ROPE_ROW_TILE-row blocks; batch_idx = block base // ROPE_ROW_BLOCK
    # picks the per-batch cos/sin row. Rotation indices/sign and cos_il/sin_il are
    # built once per block.
    #   out[j] = x[j]*cos_il[j] + x[j^1]*sign[j]*sin_il[j]  (sign folded into sin_il_signed)
    for idx in pl.spmd(T * IDX_N_HEADS // ROPE_ROW_TILE, name_hint="qr_rope"):
        o0 = idx * ROPE_ROW_TILE
        batch_idx = o0 // ROPE_ROW_BLOCK
        cos_b = cos[batch_idx : batch_idx + 1, 0 : ROPE_HEAD_DIM // 2]
        sin_b = sin[batch_idx : batch_idx + 1, 0 : ROPE_HEAD_DIM // 2]
        rope_ones = pl.full([ROPE_ROW_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        rope_col = pl.col_expand_mul(rope_ones, pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32))
        rope_dup_f = pl.cast(pl.cast(pl.mul(rope_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        rope_dup_idx = pl.cast(rope_dup_f, target_type=pl.INT32)                                       # j>>1
        rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))                                          # j%2
        rope_swap_idx = pl.cast(pl.sub(pl.add(rope_col, 1.0), pl.mul(rope_lane, 2.0)), target_type=pl.INT32)  # j^1
        rope_sign = pl.sub(pl.mul(rope_lane, 2.0), 1.0)                                                # [-1,+1,...]
        cos_b32 = pl.col_expand_mul(pl.full([ROPE_ROW_TILE, ROPE_HEAD_DIM // 2], dtype=pl.FP32, value=1.0), cos_b)
        sin_b32 = pl.col_expand_mul(pl.full([ROPE_ROW_TILE, ROPE_HEAD_DIM // 2], dtype=pl.FP32, value=1.0), sin_b)
        cos_il = pl.gather(cos_b32, dim=-1, index=rope_dup_idx)
        # fold sign into sin_il
        sin_il_signed = pl.mul(pl.gather(sin_b32, dim=-1, index=rope_dup_idx), rope_sign)
        qr_nope_slice = qr_proj_flat[o0 : o0 + ROPE_ROW_TILE, 0 : IDX_NOPE_HEAD_DIM]
        qr_bf16[o0 : o0 + ROPE_ROW_TILE, 0 : IDX_NOPE_HEAD_DIM] = pl.cast(qr_nope_slice, target_type=pl.BF16, mode="rint")
        qr_rope_slice = qr_proj_flat[o0 : o0 + ROPE_ROW_TILE, IDX_NOPE_HEAD_DIM : IDX_HEAD_DIM]
        qr_swapped = pl.gather(qr_rope_slice, dim=-1, index=rope_swap_idx)
        rope_rot = pl.add(pl.mul(qr_rope_slice, cos_il), pl.mul(qr_swapped, sin_il_signed))
        qr_bf16[o0 : o0 + ROPE_ROW_TILE, IDX_NOPE_HEAD_DIM : IDX_HEAD_DIM] = pl.cast(rope_rot, target_type=pl.BF16, mode="rint")

    # cube-only scope: q @ hadamard lands in GM, keeping the vector amax/quant below
    # in its own scope so the two run as separate cube and vector tasks.
    qh_acc_gm = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.FP32)
    for idx in pl.spmd(T * IDX_N_HEADS // QH_MM_TILE, name_hint="qr_hadamard_matmul"):
        o0 = idx * QH_MM_TILE
        qh_acc = pl.matmul(qr_bf16[o0 : o0 + QH_MM_TILE, :], hadamard, out_dtype=pl.FP32)
        qh_acc_gm[o0 : o0 + QH_MM_TILE, :] = qh_acc

    qr_hadamard_fp8 = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.FP8E4M3FN)
    qr_hadamard_scale_dq = pl.create_tensor([T * IDX_N_HEADS, 1], dtype=pl.FP32)
    for idx in pl.spmd(T * IDX_N_HEADS // QH_QUANT_TILE, name_hint="qr_hadamard_quant"):
        o0 = idx * QH_QUANT_TILE
        qh_amax = pl.full([1, QH_QUANT_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for h0 in pl.range(0, IDX_HEAD_DIM, QH_HEAD_DIM_TILE):
            qh_a_f32 = qh_acc_gm[o0 : o0 + QH_QUANT_TILE, h0 : h0 + QH_HEAD_DIM_TILE]
            qh_a_abs = pl.maximum(qh_a_f32, pl.neg(qh_a_f32))
            qh_a_max = pl.reshape(pl.row_max(qh_a_abs), [1, QH_QUANT_TILE])
            qh_amax = pl.maximum(qh_amax, qh_a_max)
        qh_scale_quant_row = pl.div(pl.full([1, QH_QUANT_TILE], dtype=pl.FP32, value=FP8_E4M3_MAX), qh_amax)
        qh_scale_dq = pl.reshape(pl.recip(qh_scale_quant_row), [QH_QUANT_TILE, 1])
        qr_hadamard_scale_dq[o0 : o0 + QH_QUANT_TILE, :] = qh_scale_dq
        qh_scale_quant = pl.reshape(qh_scale_quant_row, [QH_QUANT_TILE, 1])
        for h1 in pl.range(0, IDX_HEAD_DIM, QH_HEAD_DIM_TILE):
            qh_q_f32 = qh_acc_gm[o0 : o0 + QH_QUANT_TILE, h1 : h1 + QH_HEAD_DIM_TILE]
            qh_q_scaled = pl.row_expand_mul(qh_q_f32, qh_scale_quant)
            qh_fp8 = pl.cast(qh_q_scaled, target_type=pl.FP8E4M3FN, mode="rint")
            qr_hadamard_fp8[o0 : o0 + QH_QUANT_TILE, h1 : h1 + QH_HEAD_DIM_TILE] = qh_fp8

    x_flat = pl.reshape(x, [T, D])
    weights = pl.create_tensor([T_PAD, IDX_N_HEADS], dtype=pl.FP32)
    weights_partial = pl.create_tensor([WEIGHTS_OK * MM_ROW_TILE, IDX_N_HEADS], dtype=pl.FP32)
    # Deferred behind the caller's rms_norm dummy barrier: qkv's qr_proj_matmul is the
    # critical path and must win the cores when rms_norm retires.
    with pl.spmd(WEIGHTS_OK, name_hint="weights_proj", deps=[late_dep]) as _weights_tid:
        kb = pl.tile.get_block_idx()
        k_base = kb * WEIGHTS_K_SLICE
        weights_acc = pl.create_tensor([MM_ROW_TILE, IDX_N_HEADS], dtype=pl.FP32)
        for db in pl.range(WEIGHTS_K_SLICE // D_TILE):
            d0 = k_base + db * D_TILE
            x_tile = pl.slice(x_flat, [MM_ROW_TILE, D_TILE], [0, d0], valid_shape=[pl.min(MM_ROW_TILE, T), D_TILE])
            weights_proj_tile = weights_proj[d0 : d0 + D_TILE, :]
            if db == 0:
                weights_acc = pl.matmul(x_tile, weights_proj_tile, out_dtype=pl.FP32)
            else:
                weights_acc = pl.matmul_acc(weights_acc, x_tile, weights_proj_tile)
        weights_partial[kb * MM_ROW_TILE : kb * MM_ROW_TILE + MM_ROW_TILE, :] = weights_acc

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="weights_proj_reduce"):
        w_sum = weights_partial[0:MM_ROW_TILE, :]
        for kb in pl.unroll(1, WEIGHTS_OK):
            w_sum = pl.add(w_sum, weights_partial[kb * MM_ROW_TILE : kb * MM_ROW_TILE + MM_ROW_TILE, :])
        weights[0:MM_ROW_TILE, :] = pl.mul(w_sum, WEIGHTS_SCALE)

    indexer_compressor(
        x, inner_kv,
        inner_compress_state, inner_compress_state_block_table,
        inner_wkv, inner_wgate, inner_ape, inner_norm_w,
        cos, sin, hadamard, idx_kv_cache, idx_kv_scale,
        position_ids, idx_slot_mapping, inner_state_slot_mapping,
        late_dep,
    )

    kv_cache_fp8_flat = pl.reshape(idx_kv_cache, [IDX_CACHE_BLOCK_NUM * BLOCK_SIZE, IDX_HEAD_DIM])
    kv_scale_flat = pl.reshape(idx_kv_scale, [IDX_CACHE_BLOCK_NUM * BLOCK_SIZE, 1])
    idx_block_table_flat = pl.reshape(idx_block_table, [B * IDX_CACHE_MAX_BLOCKS])
    score_flat = pl.reshape(score, [T, SCORE_LEN])

    # Two GM-handoff stages: FP8 matmul (cube, reads paged C8 directly) -> reduce (vec).
    score_acc_gm = pl.create_tensor([T * IDX_KV_LEN, IDX_N_HEADS], dtype=pl.FP32)

    # read paged C8 KV one page per tile, matmul with the per-step-quantized FP8 query
    for tg in pl.spmd(T, name_hint="score_mat"):
        b = tg // S
        s = tg - b * S
        clen_b = pl.read(kv_seq_lens, [b]) // COMPRESS_RATIO
        cblk_b = (clen_b + BLOCK_SIZE - 1) // BLOCK_SIZE
        qb = b * S * IDX_N_HEADS
        qr_full = qr_hadamard_fp8[qb + s * IDX_N_HEADS : qb + (s + 1) * IDX_N_HEADS, 0 : IDX_HEAD_DIM]
        for cb in pl.pipeline(0, cblk_b, stage=2):
            cache0 = cb * BLOCK_SIZE
            idx_blk_id = pl.cast(
                pl.read(idx_block_table_flat, [b * IDX_CACHE_MAX_BLOCKS + cb]),
                pl.INDEX,
            )
            kv0 = idx_blk_id * BLOCK_SIZE
            base = tg * IDX_KV_LEN + cache0
            kv_fp8_mat = kv_cache_fp8_flat[kv0 : kv0 + BLOCK_SIZE, :]
            score_acc_mat = pl.matmul(kv_fp8_mat, qr_full, out_dtype=pl.FP32, b_trans=True)
            score_acc_gm[base : base + BLOCK_SIZE, :] = score_acc_mat

    for unit in pl.spmd(T * REDUCE_NSPLIT, name_hint="score_reduce"):
        tg = unit // REDUCE_NSPLIT
        split = unit - tg * REDUCE_NSPLIT
        b = tg // S
        s = tg - b * S
        clen_b = pl.read(kv_seq_lens, [b]) // COMPRESS_RATIO
        pos_t = pl.read(position_ids, [b, s])
        visible_len_t = pl.min(pl.min(clen_b, (pos_t + 1) // COMPRESS_RATIO), SCORE_LEN)
        cblk_t = (visible_len_t + REDUCE_TILE - 1) // REDUCE_TILE
        tb = b * S
        qb = b * S * IDX_N_HEADS
        qh_scale_s = pl.reshape(qr_hadamard_scale_dq[qb + s * IDX_N_HEADS : qb + (s + 1) * IDX_N_HEADS, :], [1, IDX_N_HEADS])
        weights_row_s = pl.reshape(weights[tb + s : tb + s + 1, :], [1, IDX_N_HEADS])
        lane_iters = (cblk_t - split + REDUCE_NSPLIT - 1) // REDUCE_NSPLIT
        for cb_local in pl.pipeline(0, lane_iters, stage=2):
            cb = split + cb_local * REDUCE_NSPLIT
            cache0 = cb * REDUCE_TILE
            valid_len = pl.min(REDUCE_TILE, visible_len_t - cache0)
            base = tg * IDX_KV_LEN + cache0
            idx_blk_id = pl.cast(
                pl.read(idx_block_table_flat, [b * IDX_CACHE_MAX_BLOCKS + cb]),
                pl.INDEX,
            )
            kv0 = idx_blk_id * BLOCK_SIZE
            score_acc_red = score_acc_gm[base : base + REDUCE_TILE, :]
            kv_dq_red = kv_scale_flat[kv0 : kv0 + REDUCE_TILE, :]  # paged per-position dequant scale
            # Apply q-scale then ReLU; kv dequant after head-sum (same as golden).
            # Scalar 0.0 (gate.py style) — full(0) / mul(x,0) ReLU forms were unreliable on A5.
            score_tile_red = pl.col_expand_mul(score_acc_red, qh_scale_s)
            relu_score_red = pl.maximum(score_tile_red, 0.0)
            weighted_score_red = pl.col_expand_mul(relu_score_red, weights_row_s)
            weighted_score_row = pl.mul(pl.row_sum(weighted_score_red), kv_dq_red)
            weighted_score_s = pl.reshape(weighted_score_row, [1, REDUCE_TILE])
            weighted_score_valid_s = pl.fillpad(pl.set_validshape(weighted_score_s, 1, valid_len), pad_value=pl.PadValue.min)
            weighted_score_valid_s = pl.maximum(
                weighted_score_valid_s,
                pl.full([1, REDUCE_TILE], dtype=pl.FP32, value=FP32_NEG_INF),
            )
            score_flat[tb + s : tb + s + 1, cache0 : cache0 + REDUCE_TILE] = weighted_score_valid_s

    topk_idxs_flat = pl.reshape(topk_idxs, [T, SCORE_LEN])
    for t in pl.spmd(T, name_hint="topk"):
        invalid_idxs = pl.full([1, SCORE_LEN], dtype=pl.INT32, value=-1)
        topk_idxs_flat[t : t + 1, :] = invalid_idxs
        batch_idx = t // S
        token_s = t - batch_idx * S
        cache_len_b = pl.read(kv_seq_lens, [batch_idx]) // COMPRESS_RATIO
        pos_t = pl.read(position_ids, [batch_idx, token_s])
        visible_len_t = pl.min(pl.min(cache_len_b, (pos_t + 1) // COMPRESS_RATIO), SCORE_LEN)
        if visible_len_t > 0:
            offset_i32 = pl.cast(offset, target_type=pl.INT32)
            score_full_raw = score_flat[t : t + 1, 0:SCORE_LEN]
            score_full = pl.fillpad(pl.set_validshape(score_full_raw, 1, visible_len_t), pad_value=pl.PadValue.min)
            score_full = pl.maximum(score_full, pl.full([1, SCORE_LEN], dtype=pl.FP32, value=FP32_NEG_INF))
            idx_init = pl.arange(0, [1, SCORE_LEN], dtype=pl.UINT32)
            sorted_full = pl.sort32(score_full, idx_init)
            sorted_full = pl.mrgsort(sorted_full, block_len=64)
            sorted_full = pl.mrgsort(sorted_full, block_len=256)
            sorted_full = pl.mrgsort(sorted_full, block_len=1024)

            # After the 1024 merge, the 4096-score row is two sorted 2048-score
            # runs. sort32/mrgsort keeps score/index pairs interleaved, so the
            # second 2048-score run starts at pair-lane offset 2 * 2048.
            half0_candidates = sorted_full[:, 0:TOPK_PAIR_WIDTH]
            half1_candidates = sorted_full[:, TOPK_HALF_PAIR_OFFSET : TOPK_HALF_PAIR_OFFSET + TOPK_PAIR_WIDTH]
            merged_candidates = pl.mrgsort(half0_candidates, half1_candidates)
            topk_pairs = merged_candidates[:, 0:TOPK_PAIR_WIDTH]
            topk_idxs_tile = pl.gather(topk_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            valid_topk = pl.min(IDX_TOPK, visible_len_t)
            topk_idxs_valid = pl.set_validshape(topk_idxs_tile, 1, valid_topk)
            topk_idxs_flat[t : t + 1, 0:IDX_TOPK] = pl.add(topk_idxs_valid, offset_i32)

    return score, topk_idxs


@pl.jit
def indexer_test(
    x: pl.Tensor[[B, S, D], pl.BF16],
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[_WQ_B_SCALE_ROWS, MM_N_TILE], pl.FP8E8M0],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_kv: pl.Tensor[[B, S, INNER_HEAD_DIM], pl.FP32],
    inner_compress_state: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32],
    inner_compress_state_block_table: pl.Tensor[[B, INNER_STATE_MAX_BLOCKS], pl.INT32],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.FP8E4M3FN]],
    idx_kv_scale: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[B, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    score: pl.Out[pl.Tensor[[B, S, SCORE_LEN], pl.FP32]],
    topk_idxs: pl.Out[pl.Tensor[[B, S, SCORE_LEN], pl.INT32]],
    position_ids: pl.Tensor[[B, S], pl.INT32],
    idx_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
    offset: pl.Scalar[pl.INT32],
):
    # Standalone: no rms_norm producer, so the barrier fences nothing (ready on submit).
    late_dep = pl.system.task_dummy(deps=[])
    indexer(
        x,
        qr,
        qr_scale,
        wq_b,
        wq_b_scale,
        weights_proj,
        cos,
        sin,
        hadamard,
        inner_kv,
        inner_compress_state,
        inner_compress_state_block_table,
        inner_wkv,
        inner_wgate,
        inner_ape,
        inner_norm_w,
        idx_kv_cache,
        idx_kv_scale,
        idx_block_table,
        score,
        topk_idxs,
        position_ids,
        idx_slot_mapping,
        inner_state_slot_mapping,
        kv_seq_lens,
        offset,
        late_dep,
    )
    return score, idx_kv_cache, idx_kv_scale, topk_idxs


def _int8_quant_per_row(x):
    """Per-row INT8 symmetric quant matching the QKV Step 4 ``qr`` output."""
    import torch

    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)


def _fp8_quant_per_row(x):
    """Per-row FP8 e4m3 symmetric quant with FP32 dequant scale (max=448)."""
    import torch

    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = FP8_E4M3_MAX / amax
    scaled = rows * scale_quant
    out_fp8 = scaled.to(torch.float8_e4m3fn)
    scale_dequant = 1.0 / scale_quant
    return out_fp8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)


def golden_indexer(tensors):
    """Torch reference for Indexer.forward decode branch; prefill `start_pos == 0` path is omitted."""
    import torch
    from decode_indexer_compressor import golden_compressor

    x = tensors["x"].float()
    qr = tensors["qr"]
    qr_scale = tensors["qr_scale"].float()
    wq_b = tensors["wq_b"]
    wq_b_scale = tensors["wq_b_scale"]  # tiled MX_B_NN packed e8m0
    weights_proj = tensors["weights_proj"].float()
    cos = tensors["cos"]
    sin = tensors["sin"]
    hadamard = tensors["hadamard"].float()

    kv_seq_lens = tensors["kv_seq_lens"].to(torch.int64)
    offset = int(tensors["offset"])

    bsz, seqlen, _ = x.shape
    ratio, rd = COMPRESS_RATIO, ROPE_HEAD_DIM

    def _b_scale(s):
        return unpack_scale_b_nn_tiled(
            s,
            k_tile_rows=_QR_KS,
            n_tile=MM_N_TILE,
            logical_k=Q_LORA_SCALE,
            logical_n=IDX_N_HEADS * IDX_HEAD_DIM,
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

    qr_f = qr.float() * qr_scale
    q_proj = mx_matmul_act_tiled(qr_f, wq_b, _b_scale(wq_b_scale), Q_TILE)
    q = q_proj.view(B, S, IDX_N_HEADS, IDX_HEAD_DIM)

    x_pair = q[..., -rd:].unflatten(-1, (-1, 2))
    x0, x1 = x_pair[..., 0], x_pair[..., 1]
    cos_v = cos.view(B, 1, 1, -1)
    sin_v = sin.view(B, 1, 1, -1)
    y0 = (x0 * cos_v - x1 * sin_v).to(torch.bfloat16)
    y1 = (x0 * sin_v + x1 * cos_v).to(torch.bfloat16)

    q = torch.cat([q[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1)

    q = q.to(torch.bfloat16).float() @ hadamard
    # Hybrid LI A8C8: q and Indexer Cache are FP8 e4m3 per row/position + FP32 dequant scale.

    inner_tensors = {
        "x": tensors["x"],
        "kv": tensors["inner_kv"],
        "wkv": tensors["inner_wkv"],
        "wgate": tensors["inner_wgate"],
        "ape": tensors["inner_ape"],
        "norm_w": tensors["inner_norm_w"],
        "cos": tensors["cos"],
        "sin": tensors["sin"],
        "hadamard": tensors["hadamard"],
        "compress_state": tensors["inner_compress_state"],
        "compress_state_block_table": tensors["inner_compress_state_block_table"],
        "idx_kv_cache": tensors["idx_kv_cache"],
        "idx_kv_scale": tensors["idx_kv_scale"],
        "position_ids": tensors["position_ids"],
        "idx_slot_mapping": tensors["idx_slot_mapping"],
        "inner_state_slot_mapping": tensors["inner_state_slot_mapping"],
    }
    golden_compressor(inner_tensors)

    weights = (x @ weights_proj) * WEIGHTS_SCALE

    # C8 cache: pre-quantized FP8 KV + per-position dequant scale (no score-time re-quant)
    idx_kv_cache_fp8 = tensors["idx_kv_cache"]
    idx_kv_scale = tensors["idx_kv_scale"].float()
    idx_block_table = tensors["idx_block_table"]
    score_full = torch.full((bsz, seqlen, SCORE_LEN), FP32_NEG_INF, dtype=torch.float32)
    topk_idxs = torch.full((bsz, seqlen, SCORE_LEN), -1, dtype=torch.int32)
    q_fp8, q_scale = _fp8_quant_per_row(q.reshape(B * S * IDX_N_HEADS, IDX_HEAD_DIM))
    q_fp8 = q_fp8.view(B, S, IDX_N_HEADS, IDX_HEAD_DIM)
    q_scale = q_scale.view(B, S, IDX_N_HEADS, 1)

    for b in range(bsz):
        cache_len = int(kv_seq_lens[b].item()) // ratio
        if cache_len <= 0:
            continue

        kv_fp8_rows = []
        kv_scale_rows = []
        for slot in range(cache_len):
            blk_id = int(idx_block_table[b, slot // BLOCK_SIZE].item())
            kv_fp8_rows.append(idx_kv_cache_fp8[blk_id, slot % BLOCK_SIZE, 0])
            kv_scale_rows.append(idx_kv_scale[blk_id, slot % BLOCK_SIZE, 0, 0])
        kv_fp8 = torch.stack(kv_fp8_rows, dim=0).view(cache_len, IDX_HEAD_DIM)
        kv_scale = torch.stack(kv_scale_rows, dim=0).view(cache_len, 1)
        score_raw = torch.einsum("shd,td->sht", q_fp8[b].float(), kv_fp8.float())
        score = score_raw * q_scale[b]
        score = (torch.relu(score) * weights[b].unsqueeze(-1)).sum(dim=1)
        score = score * kv_scale.view(1, cache_len)
        for s in range(seqlen):
            visible_len = min(cache_len, int(tensors["position_ids"][b, s].item() + 1) // ratio, SCORE_LEN)
            if visible_len <= 0:
                continue
            score_full[b, s, :visible_len] = score[s, :visible_len].to(torch.float32)
            k = min(IDX_TOPK, visible_len)
            _, idx = score[s, :visible_len].topk(k, dim=-1)
            topk_idxs[b, s, :k] = idx.to(torch.int32)
            topk_idxs[b, s, :k] += offset

    tensors["score"][:] = score_full

    tensors["topk_idxs"][:] = topk_idxs.view(B, S, SCORE_LEN)


def build_tensor_specs(start_pos=None):
    import torch  # type: ignore[import]
    from decode_metadata import (
        block_table,
        compressed_slot_mapping,
        csa_decode_start_set,
        kv_seq_lens_from_starts,
        position_ids_from_starts,
        resolve_start_positions,
        state_slot_mapping,
    )
    from golden import ScalarSpec, TensorSpec
    from rope_tables import build_deepseek_v4_rope_tables, materialize_half_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_deepseek_v4_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)

    def init_x():
        return torch.rand(B, S, D)
    def init_qr():
        return torch.rand(T, Q_LORA)
    # weights_proj / inner compressor calibrated to the real DeepSeek-V4-Flash CSA indexer
    # (mean l8/l32 of extract_weights_flash): zero-mean Gaussian at the measured std, gamma
    # near the measured mean. idx wq_b uses the MXFP8 grid below (not a benign randn INT8).
    def init_weights_proj():
        return torch.randn(D, IDX_N_HEADS) * 0.2313
    def init_rope_positions():
        return init_position_ids().to(torch.int64)[:, 0]
    def init_cos():
        return materialize_half_rope_tables(shared_freqs_cos, shared_freqs_sin, init_rope_positions())[0]
    def init_sin():
        return materialize_half_rope_tables(shared_freqs_cos, shared_freqs_sin, init_rope_positions())[1]
    def init_hadamard():
        return torch.rand(IDX_HEAD_DIM, IDX_HEAD_DIM) * (IDX_HEAD_DIM ** -0.5)
    def init_inner_compress_state():
        state = torch.zeros(INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM)
        state[:, :, INNER_OUT_DIM:] = FP32_NEG_INF
        return state
    def init_inner_compress_state_block_table():
        return block_table(
            batch=B,
            table_blocks=INNER_STATE_MAX_BLOCKS,
            physical_blocks=INNER_STATE_PHYSICAL_BLOCKS,
        )
    def init_inner_wkv():
        return torch.randn(INNER_OUT_DIM, D) * 0.0293
    def init_inner_wgate():
        return torch.randn(INNER_OUT_DIM, D) * 0.0512
    def init_inner_ape():
        return torch.randn(COMPRESS_RATIO, INNER_OUT_DIM) * 0.1528
    def init_inner_norm_w():
        return 0.6850 + 0.2610 * torch.randn(INNER_HEAD_DIM)
    def init_idx_block_table():
        return block_table(
            batch=B,
            table_blocks=IDX_CACHE_MAX_BLOCKS,
            physical_blocks=IDX_CACHE_MAX_BLOCKS,
        )
    def init_default_start_pos():
        # Canonical CSA start-position set (ratio-4 compressor + indexer + sliding-window + 8k).
        return csa_decode_start_set(
            batch=B, seq=S, compress_ratio=COMPRESS_RATIO,
            state_block_size=INNER_STATE_BLOCK_SIZE, cache_tile=CACHE_TILE)
    def init_start_pos():
        return resolve_start_positions(
            start_pos,
            batch=B,
            seq=S,
            max_seq_len=MAX_SEQ_LEN,
            default_fn=init_default_start_pos,
        )
    def init_position_ids():
        return position_ids_from_starts(init_start_pos(), seq=S)
    def init_kv_seq_lens():
        return kv_seq_lens_from_starts(init_start_pos(), seq=S)
    def init_inner_state_slot_mapping():
        return state_slot_mapping(
            init_position_ids(),
            init_inner_compress_state_block_table(),
            state_block_size=INNER_STATE_BLOCK_SIZE,
        )
    def init_idx_slot_mapping():
        positions = init_position_ids()
        return compressed_slot_mapping(
            positions,
            init_idx_block_table(),
            compress_ratio=COMPRESS_RATIO,
            block_size=BLOCK_SIZE,
        )

    # idx wq_b: MXFP8 Right [Q_LORA, N] + tiled MX_B_NN scale.
    wq_b, wq_b_scale = gen_mxfp8_weight_kn(
        (Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM),
        dequant_std=0.108,
        chan_cv=0.56,
        n_tile=MM_N_TILE,
        k_tile=Q_TILE,
    )
    qr_i8, qr_scale = _int8_quant_per_row(init_qr())

    # C8 indexer cache fixture: FP8 e4m3 + FP32 scale from one bf16-rounded random draw
    idx_kv_cache_bf16 = torch.rand(IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM).to(torch.bfloat16)
    idx_kv_fp8, idx_kv_sc = _fp8_quant_per_row(
        idx_kv_cache_bf16.float().reshape(IDX_CACHE_BLOCK_NUM * BLOCK_SIZE, IDX_HEAD_DIM))
    idx_kv_fp8 = idx_kv_fp8.view(IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM)
    idx_kv_sc = idx_kv_sc.view(IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1)

    return [
        TensorSpec("x", [B, S, D], torch.bfloat16, init_value=init_x),
        TensorSpec("qr", [T, Q_LORA], torch.int8, init_value=lambda: qr_i8),
        TensorSpec("qr_scale", [T, 1], torch.float32, init_value=lambda: qr_scale),
        TensorSpec("wq_b", [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], torch.float8_e4m3fn, init_value=lambda: wq_b),
        TensorSpec(
            "wq_b_scale", [_WQ_B_SCALE_ROWS, MM_N_TILE], torch.float8_e8m0fnu,
            init_value=lambda: wq_b_scale,
        ),
        TensorSpec("weights_proj", [D, IDX_N_HEADS], torch.bfloat16, init_value=init_weights_proj),
        TensorSpec("cos", [B, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_cos),
        TensorSpec("sin", [B, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_sin),
        TensorSpec("hadamard", [IDX_HEAD_DIM, IDX_HEAD_DIM], torch.bfloat16, init_value=init_hadamard),
        TensorSpec("inner_kv", [B, S, INNER_HEAD_DIM], torch.float32),
        TensorSpec("inner_compress_state", [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], torch.float32, init_value=init_inner_compress_state),
        TensorSpec("inner_compress_state_block_table", [B, INNER_STATE_MAX_BLOCKS], torch.int32, init_value=init_inner_compress_state_block_table),
        TensorSpec("inner_wkv", [INNER_OUT_DIM, D], torch.bfloat16, init_value=init_inner_wkv),
        TensorSpec("inner_wgate", [INNER_OUT_DIM, D], torch.bfloat16, init_value=init_inner_wgate),
        TensorSpec("inner_ape", [COMPRESS_RATIO, INNER_OUT_DIM], torch.float32, init_value=init_inner_ape),
        TensorSpec("inner_norm_w", [INNER_HEAD_DIM], torch.bfloat16, init_value=init_inner_norm_w),
        TensorSpec("idx_kv_cache", [IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], torch.float8_e4m3fn, init_value=lambda: idx_kv_fp8, is_output=True),
        TensorSpec("idx_kv_scale", [IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1], torch.float32, init_value=lambda: idx_kv_sc, is_output=True),
        TensorSpec("idx_block_table", [B, IDX_CACHE_MAX_BLOCKS], torch.int32, init_value=init_idx_block_table),
        # Outputs are fixed to SCORE_LEN; positions past cache_len are -inf for score and -1 for topk_idxs.
        TensorSpec("score", [B, S, SCORE_LEN], torch.float32, is_output=True),
        TensorSpec("topk_idxs", [B, S, SCORE_LEN], torch.int32, is_output=True),
        TensorSpec("position_ids", [B, S], torch.int32, init_value=init_position_ids),
        TensorSpec("idx_slot_mapping", [B, S], torch.int64, init_value=init_idx_slot_mapping),
        TensorSpec("inner_state_slot_mapping", [B, S], torch.int64, init_value=init_inner_state_slot_mapping),
        TensorSpec("kv_seq_lens", [B], torch.int32, init_value=init_kv_seq_lens),
        ScalarSpec("offset", torch.int32, OFFSET),
    ]


if __name__ == "__main__":
    import argparse
    import torch
    from golden import ratio_allclose, run_jit, topk_pair_compare

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", type=int, default=0, choices=[0, 1, 2],
                        help="L2 swimlane level: 0=off, 1=AICore timing, 2=+AICPU timing.")
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--start-pos", type=int, default=None,
                        help="Uniform fixture-only start_pos override for all batches; "
                             "default (unset) uses the canonical per-batch CSA set that includes the 8k point.")
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument(
        "--compile-only",
        action="store_true",
        default=False,
        help="Compile/codegen only (implicit on *sim platforms used by CI).",
    )
    args = parser.parse_args()

    # topk_pair_compare expects a tensor whose [..., i] entry is the score paired
    # with idx[..., i] (sorted along the top-k axis). Here `score` is per-key
    # (input-space) so it isn't pre-sorted; recover the paired scores on the fly
    # by gathering `score[topk_idxs - OFFSET]` over the valid first IDX_TOPK
    # slots, then delegate.
    def topk_idxs_compare(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        score = actual_outputs["score"]
        a_top = actual[..., :IDX_TOPK]
        e_top = expected[..., :IDX_TOPK]
        a_orig = (a_top.long() - OFFSET).clamp(min=0, max=score.shape[-1] - 1)
        paired = torch.gather(score, dim=-1, index=a_orig)
        synth_actual = {**actual_outputs, "_topk_paired_scores": paired}
        return topk_pair_compare("_topk_paired_scores")(
            a_top, e_top,
            actual_outputs=synth_actual,
            expected_outputs=expected_outputs,
            inputs=inputs,
            rtol=rtol, atol=atol,
        )
    topk_idxs_compare.__name__ = "topk_pair_compare"

    # Compare `score` only over the valid region (golden pads the tail with FP32_NEG_INF).
    # Host FP8 q vs device MX-q can leave a near-uniform global scale; align by LS
    # then allow 5% outliers (same band as HCA). Require cos≥0.99 so shape bugs fail.
    def score_valid_compare(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        expected_f = expected.cpu().to(torch.float32)
        valid = expected_f != FP32_NEG_INF
        a = actual.cpu().to(torch.float32)[valid]
        e = expected_f[valid]
        if a.numel() > 0:
            an = torch.nn.functional.normalize(a.flatten(), dim=0)
            en = torch.nn.functional.normalize(e.flatten(), dim=0)
            cos = float((an * en).sum())
            scale = (a * e).sum() / e.mul(e).sum().clamp_min(1e-12)
            e = e * scale
            if cos < 0.99:
                return False, f"score cosine {cos:.4f} < 0.99 (ls_scale={float(scale):.4f})"
        return ratio_allclose(atol=1e-4, rtol=0.25, max_error_ratio=0.05)(
            a,
            e,
            actual_outputs=actual_outputs,
            expected_outputs=expected_outputs,
            inputs=inputs,
            rtol=rtol, atol=atol,
        )
    score_valid_compare.__name__ = "score_valid_region_compare"

    indexer_tol = ATOL_RTOL["indexer_fp8"]

    result = run_jit(
        fn=indexer_test,
        specs=build_tensor_specs(args.start_pos),
        golden_fn=golden_indexer,
        runtime_dir=args.runtime_dir,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        compile_only=args.compile_only,
        rtol=indexer_tol["rtol"],
        atol=indexer_tol["atol"],
        compare_fn={
            "score":        score_valid_compare,
            "topk_idxs":    topk_idxs_compare,
            # C8 cache: history is exact; only the <=B boundary rows the compressor rewrote may
            # differ from the bf16 round of a fresh position.
            "idx_kv_cache": ratio_allclose(
                atol=indexer_tol["atol"], rtol=indexer_tol["rtol"],
                max_error_ratio=indexer_tol["pct"],
            ),
            "idx_kv_scale": ratio_allclose(
                atol=indexer_tol["atol"], rtol=indexer_tol["rtol"],
                max_error_ratio=indexer_tol["pct"],
            ),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
