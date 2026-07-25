# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 packed prefill indexer — Hybrid LI A8C8 FP8+FP32 + MXFP8 q_b.

Builds the compressed index KV cache and per-token compressed top-k indices
consumed by packed CSA prefill sparse attention.

Precision (AscendC Hybrid MXFP8-MXFP4 Step 5):
  - ``qr`` INT8 + ``qr_scale`` FP32 from QKV Step 4 (unchanged API)
  - ``wq_b`` MXFP8 W8A8 (e4m3 + e8m0 block=32) via ``matmul_mx``
  - ``indexer_q`` after Hadamard: dynamic per-token-head FP8 e4m3 + FP32 scale (max=448)
  - Indexer Cache C8: dynamic per-position FP8 e4m3 + FP32 scale (max=448; not main KV group64)
  - ``weights_proj`` / hadamard / rope / compressor wkv/wgate: BF16 (not quantized)
  - LI score ``batch_matmul``: FP8 activations → FP32 acc, scales applied in reduce
"""

import pypto.language as pl

from config import (
    FLASH as M,
    BLOCK_SIZE,
    CSA_INNER_STATE_PHYSICAL_BLOCKS,
    FP32_NEG_INF,
    IDX_CACHE_MAX_BLOCKS,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
    MX_BLOCK_K,
    PREFILL_IDX_BLOCK_NUM,
)
from mx_quant_common import (
    ATOL_RTOL,
    FP8_E4M3_MAX,
    dynamic_mx_quant_e4m3,
    gen_mxfp8_weight_kn,
    mx_matmul_fp8,
    unpack_scale_b_nn_tiled,
)
from prefill_indexer_compressor import (
    INNER_STATE_BLOCK_NUM,
    INNER_STATE_BLOCK_SIZE,
    INNER_STATE_MAX_BLOCKS,
    STATE_LEN as INNER_STATE_LEN,
    golden_prefill_indexer_compressor,
    prefill_indexer_compressor,
)

# model config (mirrors decode_indexer)
D = M.hidden_size
ROPE_HEAD_DIM = M.qk_rope_head_dim
IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
IDX_NOPE_HEAD_DIM = M.index_nope_head_dim
Q_LORA = M.q_lora_rank
Q_LORA_SCALE = Q_LORA // MX_BLOCK_K
WEIGHTS_SCALE = M.index_weights_scale
MAX_SEQ_LEN = M.max_position_embeddings
WIN = M.sliding_window

# kernel-local
COMPRESS_RATIO = 4   # the indexer only runs on ratio-4 layers
IDX_TOPK = M.index_topk
INNER_OVERLAP = COMPRESS_RATIO == 4
INNER_COFF = 1 + int(INNER_OVERLAP)
INNER_HEAD_DIM = IDX_HEAD_DIM
INNER_OUT_DIM = INNER_COFF * INNER_HEAD_DIM
INNER_COMPRESS_STATE_DIM = 2 * INNER_OUT_DIM
CACHE_TILE = 32

# Index cache table width mirrors decode. The physical idx_kv_cache pool is
# sized separately by PREFILL_IDX_BLOCK_NUM; keep the current score output cap
# at 256 rows because prefill_idx_score_out materializes [T, INDEXER_SCORE_CAP]
# in one Vec scope.
SPARSE_CMP_MAX_BLOCKS = 8
INDEXER_SCORE_MAX_BLOCKS = 2

B = 1
S = 128
T = B * S
START_POS = 0
TOPK_TILE = 16
assert T % TOPK_TILE == 0
INDEXER_SCORE_CAP = INDEXER_SCORE_MAX_BLOCKS * BLOCK_SIZE
assert INDEXER_SCORE_CAP == 256, "INDEXER_SCORE_CAP must stay at 256 rows"
INDEXER_SCORE_BLOCKS = max(1, (INDEXER_SCORE_CAP + CACHE_TILE - 1) // CACHE_TILE)
INDEXER_TOPK_CAP = min(IDX_TOPK, INDEXER_SCORE_CAP)
MAX_CMP_WRITES = max(1, T // COMPRESS_RATIO)

# Q-projection / score tiling (mirrors decode_indexer)
Q_TILE = 128
assert Q_LORA % MX_BLOCK_K == 0 and Q_TILE % MX_BLOCK_K == 0
Q_OUT_TILE = 256
HEAD_DIM_TILE = 32
D_TILE = 32
WEIGHTS_ROW_TILE = 32
QH_QUANT_BLOCK = 256
QH_QUANT_ROW_TILE = 64
ROPE_ROW_BLOCK = IDX_N_HEADS          # one token owns IDX_N_HEADS contiguous q rows + one cos/sin
ROPE_ROW_TILE = 32
# Per-token sort-tile width. The sort32/mrgsort/gather path requires a wide tile: a narrow (256)
# sort faults on device (507018) even with a proper prefix. 2048 matches the indexer KV length and
# is the confirmed fault-free width. The real score occupies only the first INDEXER_SCORE_CAP
# columns; the rest stays -inf.
SORT_LEN = 2048
MRG_TOPK_RUN = 1024   # final mrgsort run length (>= IDX_TOPK so the top-IDX_TOPK land sorted in run 0)
# topk_pairs (= 2*PREFILL_TOPK_CAP) must be a power of two aligned to the final mrgsort run: a
# misaligned prefix (e.g. 2*192) faults like a narrow sort. valid_topk then clamps to the budget.
PREFILL_TOPK_CAP = IDX_TOPK
assert PREFILL_TOPK_CAP < SORT_LEN and SORT_LEN >= INDEXER_SCORE_CAP
SCORE_INIT_TILE = 16                   # rows per -inf init write (keep [tile, SORT_LEN] under the Vec-buffer limit)
assert T % SCORE_INIT_TILE == 0
assert (IDX_N_HEADS * IDX_HEAD_DIM) % Q_OUT_TILE == 0
assert (T * IDX_N_HEADS) % QH_QUANT_BLOCK == 0
assert ROPE_ROW_BLOCK % ROPE_ROW_TILE == 0

_QR_SPMD = IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE
_QR_K_CHUNKS = Q_LORA // Q_TILE
_QR_KS = Q_TILE // MX_BLOCK_K
# Tiled MX_B_NN: each (N-tile=Q_OUT_TILE, K-chunk) independently convert_x2'd.
_WQ_B_SCALE_ROWS = _QR_SPMD * _QR_K_CHUNKS * _QR_KS  # == _QR_SPMD * Q_LORA_SCALE
_MX_WS_SLOTS = _QR_SPMD * _QR_K_CHUNKS
assert _WQ_B_SCALE_ROWS == _QR_SPMD * Q_LORA_SCALE
assert _QR_K_CHUNKS == 8  # pl.unroll literal in prefill_idx_qr_proj


@pl.jit.inline
def prefill_indexer(
    x: pl.Tensor[[T, D], pl.BF16],
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[_WQ_B_SCALE_ROWS, Q_OUT_TILE], pl.FP8E8M0],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[T, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[T, ROPE_HEAD_DIM // 2], pl.FP32],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.Tensor[
        [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM], pl.FP32
    ],
    inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.BF16],
    # C8 indexer cache: FP8 e4m3 KV (quant-on-write) + per-position FP32 dequant scale.
    idx_kv_cache: pl.Out[pl.Tensor[[PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.FP8E4M3FN]],
    idx_kv_scale: pl.Out[pl.Tensor[[PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    score: pl.Out[pl.Tensor[[T, INDEXER_SCORE_CAP], pl.FP32]],
    cmp_topk_indices: pl.Out[pl.Tensor[[T, IDX_TOPK], pl.INT32]],
    position_ids: pl.Tensor[[T], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    idx_slot_mapping: pl.Tensor[[T], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
):
    # === Q projection: dequant INT8 qr → dyn MX → matmul_mx wq_b → FP32 (mirrors decode) ===
    qr_proj = pl.create_tensor([T, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.FP32)
    mx_scale_ws = pl.create_tensor(
        [_MX_WS_SLOTS * T, Q_TILE // MX_BLOCK_K], dtype=pl.FP8E8M0
    )
    for idx in pl.spmd(IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE, name_hint="prefill_idx_qr_proj"):
        o0 = idx * Q_OUT_TILE
        # Peel K=0 with matmul_mx (init Acc); remaining via matmul_mx_acc.
        q0 = 0
        kb = 0
        qr_tile = pl.load(
            qr,
            [0, q0],
            [T, Q_TILE],
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
            [T, 1],
            target_memory=pl.Mem.Vec,
        )
        qr_dq = pl.row_expand_mul(qr_f, qr_scale_v)
        qr_q, qr_s = pl.mx_quant(qr_dq, mode="mxfp8_e4m3")
        wq_tile = pl.load(
            wq_b,
            [q0, o0],
            [Q_TILE, Q_OUT_TILE],
            target_memory=pl.Mem.Mat,
        )
        ws_tile = pl.load(
            wq_b_scale,
            [(idx * _QR_K_CHUNKS + kb) * _QR_KS, 0],
            [Q_TILE // MX_BLOCK_K, Q_OUT_TILE],
            target_memory=pl.Mem.Mat,
            mx_layout="mx_b_nn",
        )
        srow = (idx * _QR_K_CHUNKS + kb) * T
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
                [T, Q_TILE // MX_BLOCK_K],
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
        for db in pl.unroll(7):  # _QR_K_CHUNKS - 1
            q0 = (db + 1) * Q_TILE
            qr_tile2 = pl.load(
                qr,
                [0, q0],
                [T, Q_TILE],
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
                [T, 1],
                target_memory=pl.Mem.Vec,
            )
            qr_dq2 = pl.row_expand_mul(qr_f2, qr_scale_v2)
            qr_q2, qr_s2 = pl.mx_quant(qr_dq2, mode="mxfp8_e4m3")
            wq_tile2 = pl.load(
                wq_b,
                [q0, o0],
                [Q_TILE, Q_OUT_TILE],
                target_memory=pl.Mem.Mat,
            )
            ws_tile2 = pl.load(
                wq_b_scale,
                [(idx * _QR_K_CHUNKS + (db + 1)) * _QR_KS, 0],
                [Q_TILE // MX_BLOCK_K, Q_OUT_TILE],
                target_memory=pl.Mem.Mat,
                mx_layout="mx_b_nn",
            )
            srow2 = (idx * _QR_K_CHUNKS + (db + 1)) * T
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
                    [T, Q_TILE // MX_BLOCK_K],
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
        pl.store(qr_acc, [0, o0], qr_proj)

    # === Q RoPE (A3 interleaved swap-gather), one task per token (its IDX_N_HEADS rows + cos/sin) ===
    qr_proj_flat = pl.reshape(qr_proj, [T * IDX_N_HEADS, IDX_HEAD_DIM])
    qr_rope_out = pl.create_tensor([T * IDX_N_HEADS, ROPE_HEAD_DIM], dtype=pl.BF16)
    for idx in pl.spmd(T * IDX_N_HEADS // ROPE_ROW_BLOCK, name_hint="prefill_idx_qr_rope"):
        o0 = idx * ROPE_ROW_BLOCK
        token_idx = idx  # ROPE_ROW_BLOCK == IDX_N_HEADS, so one task == one token
        cos_b = cos[token_idx : token_idx + 1, 0 : ROPE_HEAD_DIM // 2]
        sin_b = sin[token_idx : token_idx + 1, 0 : ROPE_HEAD_DIM // 2]
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
        sin_il = pl.gather(sin_b32, dim=-1, index=rope_dup_idx)
        for ro in pl.range(0, ROPE_ROW_BLOCK, ROPE_ROW_TILE):
            r0 = o0 + ro
            qr_rope_slice = qr_proj_flat[r0 : r0 + ROPE_ROW_TILE, IDX_NOPE_HEAD_DIM : IDX_HEAD_DIM]
            qr_swapped = pl.gather(qr_rope_slice, dim=-1, index=rope_swap_idx)
            rope_rot = pl.add(pl.mul(qr_rope_slice, cos_il), pl.mul(pl.mul(qr_swapped, rope_sign), sin_il))
            qr_rope_out[r0 : r0 + ROPE_ROW_TILE, :] = pl.cast(rope_rot, target_type=pl.BF16, mode="rint")

    # === Q Hadamard rotation + per-row FP8 e4m3 quant (mirrors decode_indexer qr_hadamard_quant) ===
    qr_hadamard_fp8 = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.FP8E4M3FN)
    qr_hadamard_scale_dq = pl.create_tensor([T * IDX_N_HEADS, 1], dtype=pl.FP32)
    for idx in pl.spmd(T * IDX_N_HEADS // QH_QUANT_BLOCK, name_hint="prefill_idx_qr_hadamard_quant"):
        o0 = idx * QH_QUANT_BLOCK
        for ro in pl.range(0, QH_QUANT_BLOCK, QH_QUANT_ROW_TILE):
            qh_nope = pl.cast(
                qr_proj_flat[o0 + ro : o0 + ro + QH_QUANT_ROW_TILE, 0 : IDX_NOPE_HEAD_DIM],
                target_type=pl.BF16, mode="rint",
            )
            qh_rope = qr_rope_out[o0 + ro : o0 + ro + QH_QUANT_ROW_TILE, :]
            qh_acc = pl.matmul(qh_nope, hadamard[0 : IDX_NOPE_HEAD_DIM, :], out_dtype=pl.FP32)
            qh_acc = pl.matmul_acc(qh_acc, qh_rope, hadamard[IDX_NOPE_HEAD_DIM : IDX_HEAD_DIM, :])
            qh_amax = pl.full([1, QH_QUANT_ROW_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for h0 in pl.range(0, IDX_HEAD_DIM, HEAD_DIM_TILE):
                qh_a_f32 = qh_acc[0 : QH_QUANT_ROW_TILE, h0 : h0 + HEAD_DIM_TILE]
                qh_a_abs = pl.maximum(qh_a_f32, pl.neg(qh_a_f32))
                qh_a_max = pl.reshape(pl.row_max(qh_a_abs), [1, QH_QUANT_ROW_TILE])
                qh_amax = pl.maximum(qh_amax, qh_a_max)
            qh_scale_quant_row = pl.div(pl.full([1, QH_QUANT_ROW_TILE], dtype=pl.FP32, value=FP8_E4M3_MAX), qh_amax)
            qh_scale_dq = pl.reshape(pl.recip(qh_scale_quant_row), [QH_QUANT_ROW_TILE, 1])
            qr_hadamard_scale_dq[o0 + ro : o0 + ro + QH_QUANT_ROW_TILE, :] = qh_scale_dq
            qh_scale_quant = pl.reshape(qh_scale_quant_row, [QH_QUANT_ROW_TILE, 1])
            for h1 in pl.range(0, IDX_HEAD_DIM, HEAD_DIM_TILE):
                qh_q_f32 = qh_acc[0 : QH_QUANT_ROW_TILE, h1 : h1 + HEAD_DIM_TILE]
                qh_q_scaled = pl.row_expand_mul(qh_q_f32, qh_scale_quant)
                qh_fp8 = pl.cast(qh_q_scaled, target_type=pl.FP8E4M3FN, mode="rint")
                qr_hadamard_fp8[o0 + ro : o0 + ro + QH_QUANT_ROW_TILE, h1 : h1 + HEAD_DIM_TILE] = qh_fp8

    # === weights projection: (x @ weights_proj) * WEIGHTS_SCALE ===
    weights = pl.create_tensor([T, IDX_N_HEADS], dtype=pl.FP32)
    for idx in pl.spmd(T // WEIGHTS_ROW_TILE, name_hint="prefill_idx_weights_proj"):
        wrow0 = idx * WEIGHTS_ROW_TILE
        weights_acc = pl.create_tensor([WEIGHTS_ROW_TILE, IDX_N_HEADS], dtype=pl.FP32)
        for db in pl.pipeline(0, D // D_TILE, stage=2):
            d0 = db * D_TILE
            x_tile = x[wrow0 : wrow0 + WEIGHTS_ROW_TILE, d0 : d0 + D_TILE]
            wp_tile = weights_proj[d0 : d0 + D_TILE, :]
            if d0 == 0:
                weights_acc = pl.matmul(x_tile, wp_tile, out_dtype=pl.FP32)
            else:
                weights_acc = pl.matmul_acc(weights_acc, x_tile, wp_tile)
        weights[wrow0 : wrow0 + WEIGHTS_ROW_TILE, :] = pl.mul(weights_acc, WEIGHTS_SCALE)

    # === inner compressor: build the paged compressed index KV cache ===
    prefill_indexer_compressor(
        x,
        inner_compress_state, inner_compress_state_block_table,
        inner_wkv, inner_wgate, inner_ape, inner_norm_w,
        freqs_cos, freqs_sin, hadamard,
        idx_kv_cache, idx_kv_scale, idx_block_table,
        position_ids, num_tokens,
        idx_slot_mapping, inner_state_slot_mapping,
    )

    # === score: Hybrid LI A8C8 over the packed paged cache. The compressor already
    # stored each compressed row as FP8 e4m3 + a per-position FP32 dequant scale (C8),
    # so the score reads the paged FP8 block and its scale directly, multiplies by the
    # FP8 Hadamard Q tile with FP32 accumulation, then dequantizes and reduces in FP32.
    kv_cache_fp8_flat = pl.reshape(idx_kv_cache, [PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE, IDX_HEAD_DIM])
    kv_scale_flat = pl.reshape(idx_kv_scale, [PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE, 1])
    score_wide = pl.create_tensor([T, SORT_LEN], dtype=pl.FP32)                                  # wide sort scratch

    for si in pl.parallel(0, T, SCORE_INIT_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_idx_score_init"):
            score_wide[si : si + SCORE_INIT_TILE, :] = pl.full([SCORE_INIT_TILE, SORT_LEN], dtype=pl.FP32, value=FP32_NEG_INF)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_idx_score"):
        last_pos = pl.read(position_ids, [num_tokens - 1])
        max_visible = pl.min((last_pos + 1) // COMPRESS_RATIO, INDEXER_SCORE_CAP)
        for cb in pl.range(INDEXER_SCORE_BLOCKS):
            cache0 = cb * CACHE_TILE
            if max_visible > cache0:
                idx_blk_id = pl.cast(pl.read(idx_block_table, [cache0 // BLOCK_SIZE]), pl.INDEX)
                kv_row0 = idx_blk_id * BLOCK_SIZE + (cache0 % BLOCK_SIZE)
                # C8: the compressor stored this block as FP8 e4m3 + a per-position dequant scale.
                kv_q_fp8_full = kv_cache_fp8_flat[kv_row0 : kv_row0 + CACHE_TILE, 0 : IDX_HEAD_DIM]
                kv_cache_scale_dq = kv_scale_flat[kv_row0 : kv_row0 + CACHE_TILE, :]
                for t in pl.range(T):
                    if t < num_tokens:
                        q_s0 = t * IDX_N_HEADS
                        qr_hadamard_tile = qr_hadamard_fp8[q_s0 : q_s0 + IDX_N_HEADS, 0:IDX_HEAD_DIM]
                        score_acc_s = pl.matmul(kv_q_fp8_full, qr_hadamard_tile, out_dtype=pl.FP32, b_trans=True)
                        qh_scale_s = pl.reshape(qr_hadamard_scale_dq[q_s0 : q_s0 + IDX_N_HEADS, :], [1, IDX_N_HEADS])
                        score_tile_s = pl.col_expand_mul(pl.row_expand_mul(score_acc_s, kv_cache_scale_dq), qh_scale_s)
                        relu_score_s = pl.maximum(score_tile_s, pl.mul(score_tile_s, 0.0))
                        weighted_score_s = pl.reshape(pl.row_sum(pl.col_expand_mul(relu_score_s, weights[t : t + 1, :])), [1, CACHE_TILE])
                        pos = pl.read(position_ids, [t])
                        visible_t = pl.min((pos + 1) // COMPRESS_RATIO, INDEXER_SCORE_CAP)
                        if visible_t > cache0:
                            valid_len_t = pl.min(CACHE_TILE, visible_t - cache0)
                        else:
                            valid_len_t = 0
                        weighted_valid_t = pl.fillpad(pl.set_validshape(weighted_score_s, 1, valid_len_t), pad_value=pl.PadValue.min)
                        weighted_valid_t = pl.maximum(weighted_valid_t, pl.full([1, CACHE_TILE], dtype=pl.FP32, value=FP32_NEG_INF))
                        score_wide[t : t + 1, cache0 : cache0 + CACHE_TILE] = weighted_valid_t

    # Expose the real per-key scores (first INDEXER_SCORE_CAP cols of the wide sort scratch).
    score_out_flat = pl.reshape(score, [T, INDEXER_SCORE_CAP])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_idx_score_out"):
        score_out_flat[0:T, :] = score_wide[0:T, 0:INDEXER_SCORE_CAP]

    # === top-k per token over the visible (causally reachable) compressed positions ===
    for topk_idx in pl.spmd(T // TOPK_TILE, name_hint="prefill_idx_topk"):
        t0 = topk_idx * TOPK_TILE
        for ti in pl.range(TOPK_TILE):
            t = t0 + ti
            cmp_topk_indices[t : t + 1, 0:IDX_TOPK] = pl.full([1, IDX_TOPK], dtype=pl.INT32, value=-1)
            if t < num_tokens:
                pos = pl.read(position_ids, [t])
                visible_t = pl.min((pos + 1) // COMPRESS_RATIO, INDEXER_SCORE_CAP)
                if visible_t > 0:
                    # Sort the wide score row and gather the top-k indices (#505^'s exact wide+aligned
                    # sort: 2048 width, mrgsort 64/256/1024, topk_pairs = 2*IDX_TOPK proper prefix).
                    score_row = score_wide[t : t + 1, :]
                    idx_init = pl.arange(0, [1, SORT_LEN], dtype=pl.UINT32)
                    sorted_tile = pl.sort32(score_row, idx_init)
                    sorted_tile = pl.mrgsort(sorted_tile, block_len=64)
                    sorted_tile = pl.mrgsort(sorted_tile, block_len=256)
                    sorted_tile = pl.mrgsort(sorted_tile, block_len=MRG_TOPK_RUN)
                    topk_pairs = sorted_tile[:, 0 : 2 * PREFILL_TOPK_CAP]
                    topk_idxs_tile = pl.gather(topk_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                    valid_topk = pl.min(PREFILL_TOPK_CAP, visible_t)
                    cmp_topk_indices[t : t + 1, 0:PREFILL_TOPK_CAP] = pl.set_validshape(
                        topk_idxs_tile, 1, valid_topk)

    return idx_kv_cache, idx_kv_scale, score, cmp_topk_indices


def _int8_quant_per_row(x):
    """Per-row INT8 symmetric quant matching the QKV Step 4 ``qr`` output."""
    import torch

    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    out_i8 = torch.round(rows * scale_quant).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)


def _fp8_quant_per_row(x):
    """Per-row FP8 e4m3 symmetric quant with FP32 dequant scale (max=448)."""
    import torch

    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = FP8_E4M3_MAX / amax
    out_fp8 = (rows * scale_quant).to(torch.float8_e4m3fn)
    scale_dequant = 1.0 / scale_quant
    return out_fp8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)


def golden_prefill_indexer_core(tensors):
    import torch

    compressor_tensors = {
        "x": tensors["x"],
        "kv": torch.zeros(MAX_CMP_WRITES, IDX_HEAD_DIM, dtype=torch.bfloat16),
        "compress_state": tensors["inner_compress_state"],
        "inner_compress_state_block_table": tensors["inner_compress_state_block_table"],
        "wkv": tensors["inner_wkv"],
        "wgate": tensors["inner_wgate"],
        "ape": tensors["inner_ape"],
        "norm_w": tensors["inner_norm_w"],
        "freqs_cos": tensors["freqs_cos"],
        "freqs_sin": tensors["freqs_sin"],
        "hadamard": tensors["hadamard"],
        "idx_kv_cache": tensors["idx_kv_cache"],
        "idx_kv_scale": tensors["idx_kv_scale"],
        "idx_block_table": tensors["idx_block_table"],
        "position_ids": tensors["position_ids"],
        "num_tokens": tensors["num_tokens"],
        "idx_slot_mapping": tensors["idx_slot_mapping"],
        "inner_state_slot_mapping": tensors["inner_state_slot_mapping"],
    }
    golden_prefill_indexer_compressor(compressor_tensors)
    tensors["idx_kv_cache"][:] = compressor_tensors["idx_kv_cache"]
    tensors["idx_kv_scale"][:] = compressor_tensors["idx_kv_scale"]

    # --- Real lightning-indexer score + per-token causal-masked top-k ---
    # Hybrid LI A8C8: score each token's query against the compressed index KV through the
    # FP8 e4m3 + FP32 scale path, causal-mask each token to ((pos+1)//ratio), then top-k.
    num_tokens = int(tensors["num_tokens"])
    position_ids = tensors["position_ids"].long()
    rd = ROPE_HEAD_DIM
    cmp_topk_indices = torch.full((T, IDX_TOPK), -1, dtype=torch.int32)
    score_full = torch.full((T, INDEXER_SCORE_CAP), FP32_NEG_INF, dtype=torch.float32)
    visible = ((position_ids + 1) // COMPRESS_RATIO).clamp(max=INDEXER_SCORE_CAP)
    max_visible = int(visible[:num_tokens].max().item()) if num_tokens > 0 else 0
    if max_visible == 0:
        return cmp_topk_indices, score_full

    # Q: dequant INT8 qr → dyn MX → matmul_mx wq_b → per-token interleaved RoPE → Hadamard.
    qr = tensors["qr"]
    qr_scale = tensors["qr_scale"].float()
    wq_b = tensors["wq_b"]
    wq_b_scale = tensors["wq_b_scale"]
    hadamard = tensors["hadamard"].float()
    cos = tensors["cos"].float().view(T, 1, -1)
    sin = tensors["sin"].float().view(T, 1, -1)

    def _b_scale(s):
        return unpack_scale_b_nn_tiled(
            s,
            k_tile_rows=_QR_KS,
            n_tile=Q_OUT_TILE,
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
    q = mx_matmul_act_tiled(qr_f, wq_b, _b_scale(wq_b_scale), Q_TILE).view(T, IDX_N_HEADS, IDX_HEAD_DIM)
    q_pair = q[..., -rd:].unflatten(-1, (-1, 2))
    q0, q1 = q_pair[..., 0], q_pair[..., 1]
    y0 = (q0 * cos - q1 * sin).to(torch.bfloat16)
    y1 = (q0 * sin + q1 * cos).to(torch.bfloat16)
    q = torch.cat([q[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1)
    q = q.to(torch.bfloat16).float() @ hadamard

    weights = (tensors["x"].float() @ tensors["weights_proj"].float()) * WEIGHTS_SCALE  # [T, heads]

    # C8: the compressor already stored FP8 KV + a per-position dequant scale.
    cache_flat_fp8 = tensors["idx_kv_cache"].reshape(PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE, IDX_HEAD_DIM)
    scale_flat = tensors["idx_kv_scale"].float().reshape(PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE, 1)
    idx_block_table = tensors["idx_block_table"]
    rows = [
        int(idx_block_table[c // BLOCK_SIZE].item()) * BLOCK_SIZE + (c % BLOCK_SIZE)
        for c in range(max_visible)
    ]
    kv_fp8 = torch.stack([cache_flat_fp8[r] for r in rows], dim=0)
    kv_sc = torch.stack([scale_flat[r] for r in rows], dim=0)  # [max_visible, 1]

    q_fp8, q_sc = _fp8_quant_per_row(q.reshape(T * IDX_N_HEADS, IDX_HEAD_DIM))
    q_fp8 = q_fp8.view(T, IDX_N_HEADS, IDX_HEAD_DIM)
    q_sc = q_sc.view(T, IDX_N_HEADS, 1)
    q_dq = q_fp8.float() * q_sc
    kv_dq = kv_fp8.float() * kv_sc  # [max_visible, IDX_HEAD_DIM]
    score = torch.einsum("thd,cd->thc", q_dq, kv_dq)
    score = (torch.relu(score) * weights.unsqueeze(-1)).sum(dim=1)  # [T, max_visible]

    # Per-token causal mask, then top-k over the visible compressed positions.
    col = torch.arange(max_visible).unsqueeze(0)
    score = score.masked_fill(col >= visible.unsqueeze(1), FP32_NEG_INF)
    score_full[:, :max_visible] = score
    for t in range(num_tokens):
        k = int(min(INDEXER_TOPK_CAP, int(visible[t].item())))
        if k > 0:
            sel = score[t].topk(k, dim=-1)[1]
            cmp_topk_indices[t, :k] = sel.to(torch.int32)
    return cmp_topk_indices, score_full


def golden_prefill_indexer(tensors):
    import torch

    cmp_topk_indices, score_full = golden_prefill_indexer_core(tensors)
    topk_idxs = torch.full((T, INDEXER_SCORE_CAP), -1, dtype=torch.int32)
    compare_cols = min(IDX_TOPK, INDEXER_SCORE_CAP)
    topk_idxs[:, 0:compare_cols] = cmp_topk_indices[:, 0:compare_cols]
    tensors["score"][:] = score_full
    tensors["topk_idxs"][:] = topk_idxs


@pl.jit
def prefill_indexer_test(
    x: pl.Tensor[[T, D], pl.BF16],
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[_WQ_B_SCALE_ROWS, Q_OUT_TILE], pl.FP8E8M0],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[T, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[T, ROPE_HEAD_DIM // 2], pl.FP32],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.Tensor[
        [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM], pl.FP32
    ],
    inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.InOut[pl.Tensor[[PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.FP8E4M3FN]],
    idx_kv_scale: pl.InOut[pl.Tensor[[PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    score: pl.Out[pl.Tensor[[T, INDEXER_SCORE_CAP], pl.FP32]],
    topk_idxs: pl.Out[pl.Tensor[[T, INDEXER_SCORE_CAP], pl.INT32]],
    position_ids: pl.Tensor[[T], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    idx_slot_mapping: pl.Tensor[[T], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
):
    cmp_topk_indices = pl.create_tensor([T, IDX_TOPK], dtype=pl.INT32)
    prefill_indexer(
        x, qr, qr_scale, wq_b, wq_b_scale, weights_proj,
        cos, sin, freqs_cos, freqs_sin, hadamard,
        inner_compress_state, inner_compress_state_block_table,
        inner_wkv, inner_wgate, inner_ape, inner_norm_w,
        idx_kv_cache, idx_kv_scale, idx_block_table,
        score, cmp_topk_indices,
        position_ids, num_tokens,
        idx_slot_mapping, inner_state_slot_mapping,
    )
    # Expose the kernel's topk (first INDEXER_SCORE_CAP cols of cmp_topk_indices) as topk_idxs.
    for tb in pl.spmd(T // TOPK_TILE, name_hint="prefill_idx_topk_copy"):
        t0 = tb * TOPK_TILE
        for ti in pl.range(TOPK_TILE):
            t = t0 + ti
            topk_idxs[t : t + 1, 0:INDEXER_SCORE_CAP] = cmp_topk_indices[t : t + 1, 0:INDEXER_SCORE_CAP]
    return score, idx_kv_cache, idx_kv_scale, topk_idxs


def build_tensor_specs(start_pos: int = START_POS):
    import torch
    from golden import ScalarSpec, TensorSpec
    from rope_tables import build_deepseek_v4_rope_tables, materialize_half_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_deepseek_v4_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)

    num_tokens = T
    if start_pos < 0 or start_pos + T > MAX_SEQ_LEN:
        raise ValueError(f"start_pos must satisfy 0 <= start_pos <= {MAX_SEQ_LEN - T}, got {start_pos}")
    max_visible = (start_pos + T) // COMPRESS_RATIO
    if max_visible > INDEXER_SCORE_CAP:
        raise ValueError(
            f"prefill_indexer needs max_visible={max_visible} compressed slots for start_pos={start_pos}, "
            f"but the standalone score cap is INDEXER_SCORE_CAP={INDEXER_SCORE_CAP}."
        )
    write_count = sum(1 for t in range(num_tokens) if (start_pos + t + 1) % COMPRESS_RATIO == 0)
    if write_count > MAX_CMP_WRITES:
        raise ValueError(f"fixture generated {write_count} compressed writes, cap is {MAX_CMP_WRITES}")

    def init_inner_compress_state_block_table():
        table = torch.full((INNER_STATE_MAX_BLOCKS,), -1, dtype=torch.int32)
        for block in range(INNER_STATE_MAX_BLOCKS):
            table[block] = (block * 17 + 3) % CSA_INNER_STATE_PHYSICAL_BLOCKS
        return table
    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        table = init_inner_compress_state_block_table()
        block = abs_pos // INNER_STATE_BLOCK_SIZE
        intra = abs_pos % INNER_STATE_BLOCK_SIZE
        return int(table[block].item()) * INNER_STATE_BLOCK_SIZE + intra
    def init_x():
        return ((torch.rand(T, D) - 0.5) * 0.1).to(torch.bfloat16)
    def init_freqs_cos():
        return shared_freqs_cos.clone()
    def init_freqs_sin():
        return shared_freqs_sin.clone()
    def init_hadamard():
        h = torch.ones((1, 1))
        while h.shape[0] < IDX_HEAD_DIM:
            h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
        return (h * (IDX_HEAD_DIM ** -0.5)).to(torch.bfloat16)
    def init_inner_compress_state():
        state = torch.zeros(INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM)
        flat = state.view(-1, INNER_COMPRESS_STATE_DIM)
        for abs_pos in range(max(0, start_pos - INNER_STATE_LEN), start_pos):
            row = state_row(abs_pos)
            if row >= 0:
                flat[row] = (torch.rand(INNER_COMPRESS_STATE_DIM) - 0.5) * 0.05
        return state
    # Calibrated to the real DeepSeek-V4-Flash indexer inner compressor (mean l8/l32 of
    # extract_weights_flash): zero-mean Gaussian BF16 weights at the measured std; the RMSNorm
    # gamma centers near the measured mean (not ones / not uniform). Mirrors decode_indexer.
    def init_inner_wkv():
        return torch.randn(INNER_OUT_DIM, D) * 0.0293
    def init_inner_wgate():
        return torch.randn(INNER_OUT_DIM, D) * 0.0512
    def init_inner_ape():
        return torch.randn(COMPRESS_RATIO, INNER_OUT_DIM) * 0.1528
    def init_inner_norm_w():
        return 0.6850 + 0.2610 * torch.randn(INNER_HEAD_DIM)
    # C8 historical index cache: completed compressed slots hold FP8 e4m3 + FP32 dequant scale.
    # Build both from one bf16-rounded random draw so cache and scale stay consistent.
    _idx_hist = {}
    def _build_idx_hist():
        if "cache" in _idx_hist:
            return
        cache_fp8 = torch.zeros(PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM, dtype=torch.float8_e4m3fn)
        scale = torch.zeros(PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, 1)
        c_flat = cache_fp8.view(PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE, IDX_HEAD_DIM)
        s_flat = scale.view(PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE, 1)
        completed = start_pos // COMPRESS_RATIO
        for cmp_slot in range(completed):
            row = idx_row(cmp_slot)
            if row >= PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE:
                raise ValueError("fixture historical compressed slot exceeds standalone idx_kv_cache capacity")
            if row >= 0:
                hist_bf16 = ((torch.rand(IDX_HEAD_DIM) - 0.5) * 0.05).to(torch.bfloat16)
                hfp8, hsc = _fp8_quant_per_row(hist_bf16.float().view(1, IDX_HEAD_DIM))
                c_flat[row] = hfp8.view(IDX_HEAD_DIM)
                s_flat[row] = hsc.view(1)
        _idx_hist["cache"] = cache_fp8
        _idx_hist["scale"] = scale
    def init_idx_kv_cache():
        _build_idx_hist()
        return _idx_hist["cache"].clone()
    def init_idx_kv_scale():
        _build_idx_hist()
        return _idx_hist["scale"].clone()
    def init_idx_block_table():
        table = torch.full((IDX_CACHE_MAX_BLOCKS,), -1, dtype=torch.int32)
        for block in range(IDX_CACHE_MAX_BLOCKS):
            table[block] = block
        return table
    def idx_row(cmp_slot):
        table = init_idx_block_table()
        block = cmp_slot // BLOCK_SIZE
        intra = cmp_slot % BLOCK_SIZE
        phys_block = int(table[block].item())
        if phys_block < 0:
            return -1
        return phys_block * BLOCK_SIZE + intra
    def init_position_ids():
        return torch.arange(start_pos, start_pos + T, dtype=torch.int32)
    def init_idx_slot_mapping():
        mapping = torch.full((T,), -1, dtype=torch.int64)
        for t in range(num_tokens):
            pos = start_pos + t
            if (pos + 1) % COMPRESS_RATIO == 0:
                dst_row = idx_row((pos + 1) // COMPRESS_RATIO - 1)
                if dst_row >= PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE:
                    raise ValueError("fixture compressed slot exceeds standalone idx_kv_cache capacity")
                mapping[t] = dst_row
        return mapping
    def init_inner_state_slot_mapping():
        mapping = torch.full((T,), -1, dtype=torch.int64)
        for t in range(num_tokens):
            mapping[t] = state_row(start_pos + t)
        return mapping
    def init_weights_proj():
        # weights_proj calibrated to the real DeepSeek-V4-Flash indexer weights projection.
        return torch.randn(D, IDX_N_HEADS) * 0.2313
    def init_cos():
        return materialize_half_rope_tables(shared_freqs_cos, shared_freqs_sin, init_position_ids().to(torch.int64))[0]
    def init_sin():
        return materialize_half_rope_tables(shared_freqs_cos, shared_freqs_sin, init_position_ids().to(torch.int64))[1]

    # idx wq_b: MXFP8 Right [Q_LORA, N] + tiled MX_B_NN scale; qr stays INT8 from QKV.
    wq_b, wq_b_scale = gen_mxfp8_weight_kn(
        (Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM),
        dequant_std=0.108,
        chan_cv=0.56,
        n_tile=Q_OUT_TILE,
        k_tile=Q_TILE,
    )
    qr_i8, qr_scale = _int8_quant_per_row(torch.rand(T, Q_LORA))

    return [
        TensorSpec("x", [T, D], torch.bfloat16, init_value=init_x),
        TensorSpec("qr", [T, Q_LORA], torch.int8, init_value=lambda: qr_i8),
        TensorSpec("qr_scale", [T, 1], torch.float32, init_value=lambda: qr_scale),
        TensorSpec("wq_b", [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], torch.float8_e4m3fn, init_value=lambda: wq_b),
        TensorSpec(
            "wq_b_scale", [_WQ_B_SCALE_ROWS, Q_OUT_TILE], torch.float8_e8m0fnu,
            init_value=lambda: wq_b_scale,
        ),
        TensorSpec("weights_proj", [D, IDX_N_HEADS], torch.bfloat16, init_value=init_weights_proj),
        TensorSpec("cos", [T, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_cos),
        TensorSpec("sin", [T, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_sin),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("hadamard", [IDX_HEAD_DIM, IDX_HEAD_DIM], torch.bfloat16, init_value=init_hadamard),
        TensorSpec("inner_compress_state", [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM], torch.float32, init_value=init_inner_compress_state),
        TensorSpec("inner_compress_state_block_table", [INNER_STATE_MAX_BLOCKS], torch.int32, init_value=init_inner_compress_state_block_table),
        TensorSpec("inner_wkv", [INNER_OUT_DIM, D], torch.bfloat16, init_value=init_inner_wkv),
        TensorSpec("inner_wgate", [INNER_OUT_DIM, D], torch.bfloat16, init_value=init_inner_wgate),
        TensorSpec("inner_ape", [COMPRESS_RATIO, INNER_OUT_DIM], torch.float32, init_value=init_inner_ape),
        TensorSpec("inner_norm_w", [INNER_HEAD_DIM], torch.bfloat16, init_value=init_inner_norm_w),
        TensorSpec("idx_kv_cache", [PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], torch.float8_e4m3fn, init_value=init_idx_kv_cache, is_output=True),
        TensorSpec("idx_kv_scale", [PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, 1], torch.float32, init_value=init_idx_kv_scale, is_output=True),
        TensorSpec("idx_block_table", [IDX_CACHE_MAX_BLOCKS], torch.int32, init_value=init_idx_block_table),
        TensorSpec("score", [T, INDEXER_SCORE_CAP], torch.float32, is_output=True),
        TensorSpec("topk_idxs", [T, INDEXER_SCORE_CAP], torch.int32, is_output=True),
        TensorSpec("position_ids", [T], torch.int32, init_value=init_position_ids),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
        TensorSpec("idx_slot_mapping", [T], torch.int64, init_value=init_idx_slot_mapping),
        TensorSpec("inner_state_slot_mapping", [T], torch.int64, init_value=init_inner_state_slot_mapping),
    ]


if __name__ == "__main__":
    import argparse
    import torch
    from golden import ratio_allclose, run_jit, topk_pair_compare

    parser = argparse.ArgumentParser(description="Standalone token-major DeepSeek V4 prefill indexer validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "--compile-only",
        action="store_true",
        default=False,
        help="Compile/codegen only. This is also the implicit behavior on *sim platforms used by CI.",
    )
    parser.add_argument("--start-pos", type=int, default=START_POS,
                        help="Fixture-only absolute position for token 0; lowered into position_ids and dense idx_slot_mapping.")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    def topk_idxs_compare(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        score = actual_outputs["score"]
        a_top = actual[..., :IDX_TOPK]
        e_top = expected[..., :IDX_TOPK]
        invalid_top = a_top < 0
        a_orig = a_top.long().clamp(min=0, max=score.shape[-1] - 1)
        paired = torch.gather(score, dim=-1, index=a_orig)
        paired = torch.where(invalid_top, torch.full_like(paired, -torch.inf), paired)
        synth_actual = {**actual_outputs, "_topk_paired_scores": paired}
        return topk_pair_compare("_topk_paired_scores")(
            a_top, e_top,
            actual_outputs=synth_actual,
            expected_outputs=expected_outputs,
            inputs=inputs,
            rtol=rtol, atol=atol,
        )
    topk_idxs_compare.__name__ = "topk_pair_compare"

    indexer_tol = ATOL_RTOL["indexer_fp8"]

    result = run_jit(
        fn=prefill_indexer_test,
        specs=build_tensor_specs(args.start_pos),
        golden_fn=golden_prefill_indexer,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(platform=args.platform, device_id=args.device, enable_l2_swimlane=args.enable_l2_swimlane),
        rtol=indexer_tol["rtol"],
        atol=indexer_tol["atol"],
        compile_only=args.compile_only,
        compare_fn={
            "score": ratio_allclose(atol=indexer_tol["atol"], rtol=indexer_tol["rtol"]),
            "topk_idxs": topk_idxs_compare,
            # C8 cache: FP8 rows may differ slightly on boundary rows the compressor rewrote.
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
