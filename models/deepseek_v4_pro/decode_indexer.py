# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Indexer (decode). Mirrors model.py Indexer (line 380-433);
golden is a port of forward's decode branch (prefill `start_pos == 0` path is omitted).
The inner Compressor is invoked via golden_compressor (placeholder)."""


import pypto.language as pl

from config import (
    ACTIVE as M,
    DECODE_BATCH,
    DECODE_SEQ,
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    DECODE_IDX_BLOCK_NUM,
    IDX_CACHE_MAX_BLOCKS,
    FP32_NEG_INF,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
)
from decode_indexer_compressor import indexer_compressor

# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
Q_LORA = M.q_lora_rank
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
SCORE_ATOL = 1e-4
SCORE_RTOL = 1.0 / 128
SCORE_HARD_MULTIPLIER = 4.0
SCORE_HARD_ATOL = 5e-3

# tiling
CACHE_TILE = 64
assert BLOCK_SIZE % CACHE_TILE == 0, "CACHE_TILE must not cross a paged idx_kv_cache block"
# matmul/reduce tile over contiguous GM scratch, not the paged KV cache
MAT_TILE = 512
REDUCE_TILE = 128
# score_kv_quant / score_reduce fan the cache-tile loop across NSPLIT extra lanes: T * NSPLIT.
QUANT_NSPLIT = 4
REDUCE_NSPLIT = 4
Q_TILE = 256
# Q_OUT_TILE is the per-task N granularity (sets idx_qr_proj task count); MM_N_TILE
# is the Mat-safe cube N-tile. Q_OUT_TILE fans Q_OUT_TILE // MM_N_TILE cube ops per
# task so task count halves without growing the [Q_TILE, MM_N_TILE] L1 wq load.
Q_OUT_TILE = 1024
MM_N_TILE = 512
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
# The projection split must divide D / D_TILE exactly. Flash has 8 D tiles;
# Pro has 14, and these choices preserve each deployment's existing tuning.
WEIGHTS_OK = 4 if M.name == "flash" else 7
WEIGHTS_K_SLICE = D // WEIGHTS_OK
assert WEIGHTS_K_SLICE % D_TILE == 0
QH_QUANT_TILE = 64
# cube tile for q @ hadamard; L0C caps it at QH_MM_TILE * IDX_HEAD_DIM * 4B <= 64KiB.
QH_MM_TILE = 64
QH_HEAD_DIM_TILE = 64
MX_BLOCK_K = 32
MX_K_TILE = 64
MX_K_SCALE_TILE = MX_K_TILE // MX_BLOCK_K
MX_N_TILE = 256
MX_K_TILES = Q_LORA // MX_K_TILE
WQB_SCALE_ROWS = ((IDX_N_HEADS * IDX_HEAD_DIM) // MX_N_TILE) * MX_K_TILES * MX_K_SCALE_TILE
ROPE_ROW_BLOCK = S * IDX_N_HEADS
# qr_rope SPMD tile == row block: one ROPE_ROW_TILE-row block per SPMD tile.
ROPE_ROW_TILE = 32
# qr_rope gathers with a single-row flat index so the a5 lowering's per-block
# row-base chain is skipped (see the qr_rope_tables scope).
ROPE_SWAP_FLAT_LEN = ROPE_ROW_TILE * ROPE_HEAD_DIM
# float forms: the tracer does not accept float()/int() calls inside a kernel body.
ROPE_HEAD_DIM_F = float(ROPE_HEAD_DIM)
ROPE_HEAD_DIM_INV = 1.0 / ROPE_HEAD_DIM
TOPK_HALF_LEN = SCORE_LEN // 2
TOPK_HALF_PAIR_OFFSET = 2 * TOPK_HALF_LEN
TOPK_PAIR_WIDTH = 2 * IDX_TOPK
assert SCORE_LEN == 2 * TOPK_HALF_LEN, "decode indexer topk expects an even score length"
assert TOPK_HALF_LEN == 2048, "decode indexer 4096-value topk uses two 2048-value halves"
assert IDX_TOPK <= TOPK_HALF_LEN, "per-half candidate list must cover the final topk width"


@pl.jit.inline
def indexer_qr_projection(
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[WQB_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    qr_proj_out: pl.Tensor[[T, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
):
    """Dequantize the legacy QR input and project it with dynamic MXFP8."""
    qr_fp32 = pl.create_tensor([T_PAD, Q_LORA], dtype=pl.FP32)
    for k0 in pl.parallel(0, Q_LORA, Q_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="idx_qr_dequant"):
            qr_i8 = pl.load(qr, [0, k0], [T_PAD, Q_TILE], valid_shape=[T, Q_TILE], target_memory=pl.Mem.Vec)
            qr_i8 = pl.fillpad(qr_i8, pad_value=pl.PadValue.zero)
            # A5 has no direct INT8 -> FP32 tile conversion. Materialize the
            # dequantized input through the native INT8 -> FP16 -> FP32 path.
            qr_tile_fp16 = pl.cast(qr_i8, target_type=pl.FP16, mode="none")
            qr_tile_fp32 = pl.cast(qr_tile_fp16, target_type=pl.FP32, mode="none")
            qr_scale_tile = pl.load(
                qr_scale, [0, 0], [T_PAD, 1], valid_shape=[T, 1], target_memory=pl.Mem.Vec
            )
            qr_scale_tile = pl.fillpad(qr_scale_tile, pad_value=pl.PadValue.zero)
            qr_tile_fp32 = pl.row_expand_mul(qr_tile_fp32, qr_scale_tile)
            pl.store(qr_tile_fp32, [0, k0], qr_fp32)

    qr_mx = pl.create_tensor([T_PAD, Q_LORA], dtype=pl.FP8E4M3FN)
    qr_mx_scale_store = pl.create_tensor([MX_K_TILES, MM_ROW_TILE * MX_K_SCALE_TILE], dtype=pl.FP8E8M0)
    for kb in pl.parallel(MX_K_TILES):
        k0 = kb * MX_K_TILE
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="idx_qr_mx_quant"):
            qr_src = pl.load(qr_fp32, [0, k0], [MM_ROW_TILE, MX_K_TILE], target_memory=pl.Mem.Vec)
            qr_q, qr_mx_scale = pl.quant_mx(qr_src, layout=pl.MX_A_ZZ)
            pl.store(qr_q, [0, k0], qr_mx)
            qr_scale_flat = pl.reshape(qr_mx_scale, [1, MM_ROW_TILE * MX_K_SCALE_TILE])
            pl.store(qr_scale_flat, [kb, 0], qr_mx_scale_store)

    qr_partial = pl.create_tensor([MX_K_TILES * MM_ROW_TILE, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.FP32)
    for task_idx in pl.parallel(MX_K_TILES * ((IDX_N_HEADS * IDX_HEAD_DIM) // MX_N_TILE)):
        kb = task_idx // ((IDX_N_HEADS * IDX_HEAD_DIM) // MX_N_TILE)
        nb = task_idx % ((IDX_N_HEADS * IDX_HEAD_DIM) // MX_N_TILE)
        k0 = kb * MX_K_TILE
        n0 = nb * MX_N_TILE
        qr_scale_slice = qr_mx_scale_store[kb : kb + 1, :]
        qr_scale_mx = pl.tensor.view(qr_scale_slice, [MM_ROW_TILE, MX_K_SCALE_TILE], layout=pl.MX_A_ZZ)
        w_scale_offset = (nb * MX_K_TILES + kb) * MX_K_SCALE_TILE
        w_scale_slice = wq_b_scale[w_scale_offset : w_scale_offset + MX_K_SCALE_TILE, :]
        w_scale_mx = pl.tensor.view(w_scale_slice, [MX_K_SCALE_TILE, MX_N_TILE], layout=pl.MX_B_NN)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="idx_qr_proj_mx"):
            qr_k = pl.move(
                pl.load(qr_mx, [0, k0], [MM_ROW_TILE, MX_K_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.Left,
            )
            qr_scale_k = pl.move(
                pl.load(qr_scale_mx, [0, 0], [MM_ROW_TILE, MX_K_SCALE_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.LeftScale,
            )
            w_k = pl.move(
                pl.load(wq_b, [k0, n0], [MX_K_TILE, MX_N_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.Right,
            )
            w_scale_k = pl.move(
                pl.load(w_scale_mx, [0, 0], [MX_K_SCALE_TILE, MX_N_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.RightScale,
            )
            qr_partial_acc = pl.matmul_mx(qr_k, qr_scale_k, w_k, w_scale_k)
            pl.store(qr_partial_acc, [kb * MM_ROW_TILE, n0], qr_partial)

    for nb in pl.parallel((IDX_N_HEADS * IDX_HEAD_DIM) // MX_N_TILE):
        n0 = nb * MX_N_TILE
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="idx_qr_proj_reduce"):
            qr_sum = pl.tile.full([MM_ROW_TILE, MX_N_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.pipeline(MX_K_TILES, stage=2):
                qr_partial_vec = pl.load(
                    qr_partial, [kb * MM_ROW_TILE, n0], [MM_ROW_TILE, MX_N_TILE], target_memory=pl.Mem.Vec
                )
                qr_sum = pl.add(qr_sum, qr_partial_vec)
            qr_sum_valid = pl.set_validshape(qr_sum, T, MX_N_TILE)
            pl.store(qr_sum_valid, [0, n0], qr_proj_out)
    return qr_proj_out


@pl.jit.inline
def indexer(
    x: pl.Tensor[[B, S, D], pl.BF16],
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[WQB_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],  # shared by q rotation and inner Compressor
    inner_kv: pl.Tensor[[B, S, INNER_HEAD_DIM], pl.FP32],
    inner_compress_state: pl.InOut[
        pl.Tensor[
            [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM],
            pl.FP32,
        ]
    ],
    inner_compress_state_block_table: pl.Tensor[[B, INNER_STATE_MAX_BLOCKS], pl.INT32],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.BF16],
    # C8 indexer cache: INT8 KV (quant-on-write) + per-position FP32 dequant scale; no bf16 cache.
    idx_kv_cache: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
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
    qr_proj = pl.create_tensor([T, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.FP32)
    indexer_qr_projection(qr, qr_scale, wq_b, wq_b_scale, qr_proj)

    qr_proj_flat = pl.reshape(qr_proj, [T * IDX_N_HEADS, IDX_HEAD_DIM])
    # BF16 q for the Hadamard matmul: nope half rounded from the FP32 dequant, rope
    # half rotated then rounded.
    qr_bf16 = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.BF16)
    # The interleave tables are block-invariant: cos_il[j] = cos[b, j>>1], the sign
    # pattern, and the j^1 lane-swap index do not depend on the spmd block. pl.gather
    # lowers to a per-row TGATHER loop, so rebuilding them inside the grid cost
    # 16 blocks x ROPE_ROW_TILE rows x 2 tables of row-gathers per layer. Build them
    # once here and read them as plain loads per block.
    #   out[j] = x[j]*cos_il[j] + x[j^1]*sin_il_signed[j]  (sign folded into sin)
    # Folding the sign into sin is exact: multiplying by +/-1 only flips the sign bit,
    # so (x*sign)*sin and x*(sin*sign) are bit-identical.
    cos_il_t = pl.create_tensor([B, ROPE_HEAD_DIM], dtype=pl.FP32)
    sin_signed_t = pl.create_tensor([B, ROPE_HEAD_DIM], dtype=pl.FP32)
    rope_swap_idx_t = pl.create_tensor([1, ROPE_SWAP_FLAT_LEN], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_rope_tables", allow_early_resolve=True):
        il_col = pl.col_expand_mul(
            pl.full([B, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0),
            pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32))
        il_dup_f = pl.cast(pl.cast(pl.mul(il_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        il_dup_idx = pl.cast(il_dup_f, target_type=pl.INT32)                                    # j>>1
        il_lane = pl.sub(il_col, pl.mul(il_dup_f, 2.0))                                         # j%2
        il_sign = pl.sub(pl.mul(il_lane, 2.0), 1.0)                                             # [-1,+1,...]
        cos_il_t[0:B, 0:ROPE_HEAD_DIM] = pl.gather(
            cos[0:B, 0 : ROPE_HEAD_DIM // 2], dim=-1, index=il_dup_idx)
        sin_signed_t[0:B, 0:ROPE_HEAD_DIM] = pl.mul(
            pl.gather(sin[0:B, 0 : ROPE_HEAD_DIM // 2], dim=-1, index=il_dup_idx), il_sign)
        # The j^1 lane swap permutes data, so it stays an index tensor; it is block
        # invariant all the same. Store it as a SINGLE-ROW *flat* index over the
        # [ROPE_ROW_TILE, ROPE_HEAD_DIM] block: flat[r*RHD + j] = r*RHD + (j^1).
        #
        # Why flat and single-row: on a5 the gather lowering (op_conversion_registry
        # emit_a5_flat_idx) rebuilds a row-base chain -- tile.ci over rows*cols, reshape,
        # divs, muls, add -- inside EVERY spmd block whenever the index has more than one
        # row, and returns the index untouched when rows == 1. The row base is a compile-
        # time constant here, so folding it into this once-per-layer table removes four
        # full-tile INT32 passes per block.
        sw_col = pl.cast(
            pl.arange(0, [1, ROPE_SWAP_FLAT_LEN], dtype=pl.INT32), target_type=pl.FP32)
        sw_rowbase = pl.mul(
            pl.cast(pl.cast(pl.mul(sw_col, ROPE_HEAD_DIM_INV), target_type=pl.INT32, mode="trunc"),
                    target_type=pl.FP32),
            ROPE_HEAD_DIM_F)                                                               # r*RHD
        sw_j = pl.sub(sw_col, sw_rowbase)                                                       # j
        sw_dup_f = pl.cast(pl.cast(pl.mul(sw_j, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        sw_lane = pl.sub(sw_j, pl.mul(sw_dup_f, 2.0))                                           # j%2
        sw_jx = pl.sub(pl.add(sw_j, 1.0), pl.mul(sw_lane, 2.0))                                 # j^1
        rope_swap_idx_t[0:1, 0:ROPE_SWAP_FLAT_LEN] = pl.cast(
            pl.add(sw_rowbase, sw_jx), target_type=pl.INT32)

    # spmd over ROPE_ROW_TILE-row blocks; batch_idx = block base // ROPE_ROW_BLOCK
    # picks the per-batch cos/sin row. col_expand_mul folds the [1, ROPE_HEAD_DIM]
    # row broadcast into the rotation multiply, so no cos_il/sin_il tile is
    # materialized per block.
    for idx in pl.spmd(T * IDX_N_HEADS // ROPE_ROW_TILE, name_hint="qr_rope", allow_early_resolve=True):
        o0 = idx * ROPE_ROW_TILE
        batch_idx = o0 // ROPE_ROW_BLOCK
        rope_swap_flat = rope_swap_idx_t[0:1, 0:ROPE_SWAP_FLAT_LEN]
        cos_row = cos_il_t[batch_idx : batch_idx + 1, 0:ROPE_HEAD_DIM]
        sin_row = sin_signed_t[batch_idx : batch_idx + 1, 0:ROPE_HEAD_DIM]
        qr_nope_slice = qr_proj_flat[o0 : o0 + ROPE_ROW_TILE, 0 : IDX_NOPE_HEAD_DIM]
        qr_rope_slice = qr_proj_flat[o0 : o0 + ROPE_ROW_TILE, IDX_NOPE_HEAD_DIM : IDX_HEAD_DIM]
        # qr_proj_flat is a GM tensor, so this column window lowers to its own
        # tile.load and lands as a dense [ROPE_ROW_TILE, ROPE_HEAD_DIM] tile in UB:
        # the UB row stride is ROPE_HEAD_DIM, which is what the flat index assumes.
        # Slicing a UB *tile* would instead inherit the parent's row stride -- see
        # merge_norm in decode_sparse_attn.py, where the rope half is materialized as
        # an elementwise result for exactly that reason.
        # rows == 1 -> the a5 lowering uses this index as the flat index directly.
        qr_rope_flat = pl.reshape(qr_rope_slice, [1, ROPE_SWAP_FLAT_LEN])
        qr_swapped = pl.reshape(
            pl.gather(qr_rope_flat, dim=-1, index=rope_swap_flat),
            [ROPE_ROW_TILE, ROPE_HEAD_DIM])
        rope_rot = pl.add(
            pl.col_expand_mul(qr_rope_slice, cos_row), pl.col_expand_mul(qr_swapped, sin_row))
        qr_vec = pl.concat(
            pl.cast(qr_nope_slice, target_type=pl.BF16, mode="rint"),
            pl.cast(rope_rot, target_type=pl.BF16, mode="rint"))
        qr_bf16[o0 : o0 + ROPE_ROW_TILE, :] = qr_vec

    # cube-only scope: q @ hadamard lands in GM, keeping the vector amax/quant below
    # in its own scope so the two run as separate cube and vector tasks.
    qh_acc_gm = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.FP32)
    for idx in pl.spmd(T * IDX_N_HEADS // QH_MM_TILE, name_hint="qr_hadamard_matmul", allow_early_resolve=True):
        o0 = idx * QH_MM_TILE
        qh_acc = pl.matmul(qr_bf16[o0 : o0 + QH_MM_TILE, :], hadamard, out_dtype=pl.FP32)
        qh_acc_gm[o0 : o0 + QH_MM_TILE, :] = qh_acc

    qr_hadamard_i8 = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.INT8)
    qr_hadamard_scale_dq = pl.create_tensor([T * IDX_N_HEADS, 1], dtype=pl.FP32)
    for idx in pl.spmd(T * IDX_N_HEADS // QH_QUANT_TILE, name_hint="qr_hadamard_quant", allow_early_resolve=True):
        o0 = idx * QH_QUANT_TILE
        qh_amax = pl.full([1, QH_QUANT_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for h0 in pl.range(0, IDX_HEAD_DIM, QH_HEAD_DIM_TILE):
            qh_a_f32 = qh_acc_gm[o0 : o0 + QH_QUANT_TILE, h0 : h0 + QH_HEAD_DIM_TILE]
            qh_a_abs = pl.maximum(qh_a_f32, pl.neg(qh_a_f32))
            qh_a_max = pl.reshape(pl.row_max(qh_a_abs), [1, QH_QUANT_TILE])
            qh_amax = pl.maximum(qh_amax, qh_a_max)
        qh_scale_quant_row = pl.div(pl.full([1, QH_QUANT_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), qh_amax)
        qh_scale_dq = pl.reshape(pl.recip(qh_scale_quant_row), [QH_QUANT_TILE, 1])
        qr_hadamard_scale_dq[o0 : o0 + QH_QUANT_TILE, :] = qh_scale_dq
        qh_scale_quant = pl.reshape(qh_scale_quant_row, [QH_QUANT_TILE, 1])
        for h1 in pl.range(0, IDX_HEAD_DIM, QH_HEAD_DIM_TILE):
            qh_q_f32 = qh_acc_gm[o0 : o0 + QH_QUANT_TILE, h1 : h1 + QH_HEAD_DIM_TILE]
            qh_q_scaled = pl.row_expand_mul(qh_q_f32, qh_scale_quant)
            qh_q_i32 = pl.cast(qh_q_scaled, target_type=pl.INT32, mode="rint")
            qh_q_half = pl.cast(qh_q_i32, target_type=pl.FP16, mode="round")
            qh_i8 = pl.cast(qh_q_half, target_type=pl.INT8, mode="trunc")
            qr_hadamard_i8[o0 : o0 + QH_QUANT_TILE, h1 : h1 + QH_HEAD_DIM_TILE] = qh_i8

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

    kv_cache_i8_flat = pl.reshape(idx_kv_cache, [IDX_CACHE_BLOCK_NUM * BLOCK_SIZE, IDX_HEAD_DIM])
    kv_scale_flat = pl.reshape(idx_kv_scale, [IDX_CACHE_BLOCK_NUM * BLOCK_SIZE, 1])
    idx_block_table_flat = pl.reshape(idx_block_table, [B * IDX_CACHE_MAX_BLOCKS])
    score_flat = pl.reshape(score, [T, SCORE_LEN])

    # Keep the top-k input statically shaped for PTOAS 0.58. Dynamic valid
    # shapes require an explicit tsort32 workspace that this PyPTO API does not
    # expose, so initialize the invisible tail in GM before writing scores.
    for tg in pl.spmd(T, name_hint="score_init", allow_early_resolve=True):
        score_flat[tg : tg + 1, :] = pl.full([1, SCORE_LEN], dtype=pl.FP32, value=FP32_NEG_INF)

    # Two GM-handoff stages: matmul (cube, reads paged C8 directly) -> reduce (vec).
    score_acc_gm = pl.create_tensor([T * IDX_KV_LEN, IDX_N_HEADS], dtype=pl.INT32)

    # read paged C8 KV one page per tile, matmul with the per-step-quantized query
    for tg in pl.spmd(T, name_hint="score_mat", allow_early_resolve=True):
        b = tg // S
        s = tg - b * S
        clen_b = pl.read(kv_seq_lens, [b]) // COMPRESS_RATIO
        cblk_b = (clen_b + BLOCK_SIZE - 1) // BLOCK_SIZE
        qb = b * S * IDX_N_HEADS
        qr_full = qr_hadamard_i8[qb + s * IDX_N_HEADS : qb + (s + 1) * IDX_N_HEADS, 0 : IDX_HEAD_DIM]
        for cb in pl.pipeline(0, cblk_b, stage=2):
            cache0 = cb * BLOCK_SIZE
            idx_blk_id = pl.cast(
                pl.read(idx_block_table_flat, [b * IDX_CACHE_MAX_BLOCKS + cb]),
                pl.INDEX,
            )
            kv0 = idx_blk_id * BLOCK_SIZE
            base = tg * IDX_KV_LEN + cache0
            kv_i8_mat = kv_cache_i8_flat[kv0 : kv0 + BLOCK_SIZE, :]
            score_acc_mat = pl.matmul(kv_i8_mat, qr_full, out_dtype=pl.INT32, b_trans=True)
            score_acc_gm[base : base + BLOCK_SIZE, :] = score_acc_mat

    for unit in pl.spmd(T * REDUCE_NSPLIT, name_hint="score_reduce", allow_early_resolve=True):
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
            score_tile_red = pl.cast(score_acc_red, target_type=pl.FP32, mode="none")
            # per-position dequant kv_dq_red applied after the head-sum
            score_tile_red = pl.col_expand_mul(score_tile_red, qh_scale_s)
            relu_score_red = pl.maximum(score_tile_red, pl.full([REDUCE_TILE, IDX_N_HEADS], dtype=pl.FP32, value=0.0))
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
    for t in pl.spmd(T, name_hint="topk", allow_early_resolve=True):
        invalid_idxs = pl.full([1, SCORE_LEN], dtype=pl.INT32, value=-1)
        topk_idxs_flat[t : t + 1, :] = invalid_idxs
        batch_idx = t // S
        token_s = t - batch_idx * S
        cache_len_b = pl.read(kv_seq_lens, [batch_idx]) // COMPRESS_RATIO
        pos_t = pl.read(position_ids, [batch_idx, token_s])
        visible_len_t = pl.min(pl.min(cache_len_b, (pos_t + 1) // COMPRESS_RATIO), SCORE_LEN)
        if visible_len_t > 0:
            offset_i32 = pl.cast(offset, target_type=pl.INT32)
            score_full_loaded = score_flat[t : t + 1, 0:SCORE_LEN]
            score_full = score_full_loaded[:, 0:SCORE_LEN]
            idx_init_loaded = pl.arange(0, [1, SCORE_LEN], dtype=pl.UINT32)
            idx_init = idx_init_loaded[:, 0:SCORE_LEN]
            sorted_full = pl.sort32(score_full, idx_init)
            sorted_full = pl.mrgsort(sorted_full, block_len=64)
            sorted_full = pl.mrgsort(sorted_full, block_len=256)
            sorted_full = pl.mrgsort(sorted_full, block_len=1024)

            # After the 1024 merge, the 4096-score row is two sorted 2048-score
            # runs. sort32/mrgsort keeps score/index pairs interleaved, so the
            # second 2048-score run starts at pair-lane offset 2 * 2048.
            # Materialize the candidate subviews so PTOAS sizes the format-2
            # merge workspace from the 2048-lane tiles, not their 8192-lane
            # parent allocation.
            candidate_zero = pl.full([1, TOPK_PAIR_WIDTH], dtype=pl.FP32, value=0.0)
            half0_candidates = pl.add(sorted_full[:, 0:TOPK_PAIR_WIDTH], candidate_zero)
            half1_candidates = pl.add(
                sorted_full[:, TOPK_HALF_PAIR_OFFSET : TOPK_HALF_PAIR_OFFSET + TOPK_PAIR_WIDTH],
                candidate_zero,
            )
            merged_candidates = pl.mrgsort(half0_candidates, half1_candidates)
            topk_pairs = merged_candidates[:, 0:TOPK_PAIR_WIDTH]
            topk_idxs_tile = pl.gather(topk_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            valid_topk = pl.min(IDX_TOPK, visible_len_t)
            topk_idxs_valid = pl.set_validshape(topk_idxs_tile, 1, valid_topk)
            topk_idxs_flat[t : t + 1, 0:IDX_TOPK] = pl.add(topk_idxs_valid, offset_i32)

    return idx_kv_cache, idx_kv_scale, inner_compress_state, score, topk_idxs


@pl.jit
def indexer_test(
    x: pl.Tensor[[B, S, D], pl.BF16],
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[WQB_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_kv: pl.Tensor[[B, S, INNER_HEAD_DIM], pl.FP32],
    inner_compress_state: pl.InOut[
        pl.Tensor[
            [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM],
            pl.FP32,
        ]
    ],
    inner_compress_state_block_table: pl.Tensor[[B, INNER_STATE_MAX_BLOCKS], pl.INT32],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
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
    return score, inner_compress_state, idx_kv_cache, idx_kv_scale, topk_idxs


def _int8_quant_per_row(x):
    """Per-row INT8 symmetric quant matching the runtime W8A8C16 activation path."""
    import torch

    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)


def gen_shared_weight(shape, dequant_std, chan_cv):
    """Synthesize a per-output-channel-symmetric INT8 weight + FP32 scale by simulating the
    real DeepSeek-V4-Flash MXFP8 quant grid (e4m3, 128x128-block E8M0 scale), then re-quantizing
    per-output-channel. Used for the indexer ``idx wq_b`` (and shared by decode_attention_csa),
    which follows the same FP8 grid as the shared experts: ~200 discrete levels, ~1.1% zero
    spike, per-channel scale CV ~0.61. A plain randn INT8 misses that level/scale structure.
    ``chan_cv`` (log-space source-gain std) injects the per-output-channel magnitude spread the
    coarse 128-block scale leaves behind; per-channel INT8 is scale-invariant, so the grid sets
    the level shape and ``dequant_std`` only sets the absolute scale magnitude.

    ``shape`` last dim = reduction (in) dim; leading dims map to the per-output-channel scale
    shape ([out, in] -> scale [out]).
    """
    import torch

    FP8_MAX, TINY = 448.0, 1e-20

    def sim_fp8(W, block=128):   # e4m3 + 128x128-block E8M0 (round-up) scale on (out, in)
        out, inn = W.shape
        Wb = W.reshape(out // block, block, inn // block, block)
        scale = torch.exp2(torch.ceil(torch.log2((Wb.abs().amax(dim=(1, 3), keepdim=True) / FP8_MAX).clamp_min(TINY))))
        q = (Wb / scale).to(torch.float8_e4m3fn).float() * scale
        return q.reshape(out, inn)

    W = torch.randn(*shape) * torch.exp(chan_cv * torch.randn(*shape[:-1], 1))  # per-channel gain
    Wq = sim_fp8(W)
    amax = Wq.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale = amax / INT8_SCALE_MAX
    w_i8 = torch.round(Wq / scale).clamp_(-INT8_SCALE_MAX, INT8_SCALE_MAX).to(torch.int8)
    scale = (scale * (dequant_std / (w_i8.float() * scale).std())).squeeze(-1).float()
    return w_i8, scale


def golden_indexer(tensors):
    """Torch reference for Indexer.forward decode branch; prefill `start_pos == 0` path is omitted."""
    import torch
    from decode_indexer_compressor import golden_compressor
    from expert_shared import _dynamic_mxfp8_matmul, _unpack_b_scale_tiled

    x = tensors["x"].float()
    qr = tensors["qr"]
    qr_scale = tensors["qr_scale"].float()
    wq_b = tensors["wq_b"]
    wq_b_scale = _unpack_b_scale_tiled(tensors["wq_b_scale"], Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM)
    weights_proj = tensors["weights_proj"].float()
    cos = tensors["cos"]
    sin = tensors["sin"]
    hadamard = tensors["hadamard"].float()

    kv_seq_lens = tensors["kv_seq_lens"].to(torch.int64)
    offset = int(tensors["offset"])

    bsz, seqlen, _ = x.shape
    ratio, rd = COMPRESS_RATIO, ROPE_HEAD_DIM

    qr_fp32 = qr.float() * qr_scale
    q = _dynamic_mxfp8_matmul(qr_fp32, wq_b, wq_b_scale).view(B, S, IDX_N_HEADS, IDX_HEAD_DIM)

    x_pair = q[..., -rd:].unflatten(-1, (-1, 2))
    x0, x1 = x_pair[..., 0], x_pair[..., 1]
    cos_v = cos.view(B, 1, 1, -1)
    sin_v = sin.view(B, 1, 1, -1)
    y0 = (x0 * cos_v - x1 * sin_v).to(torch.bfloat16)
    y1 = (x0 * sin_v + x1 * cos_v).to(torch.bfloat16)

    q = torch.cat([q[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1)

    q = q.to(torch.bfloat16).float() @ hadamard
    # W8A8C16: q and Indexer Cache are quantized per row to INT8 for score matmul,
    # then dequantized with q_scale * kv_scale.
    # flash: fp4_act_quant on q (FP4 simulation).

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

    # C8 cache: pre-quantized INT8 KV + per-position dequant scale (no score-time re-quant)
    idx_kv_cache_i8 = tensors["idx_kv_cache"]
    idx_kv_scale = tensors["idx_kv_scale"].float()
    idx_block_table = tensors["idx_block_table"]
    score_full = torch.full((bsz, seqlen, SCORE_LEN), FP32_NEG_INF, dtype=torch.float32)
    topk_idxs = torch.full((bsz, seqlen, SCORE_LEN), -1, dtype=torch.int32)
    q_i8, q_scale = _int8_quant_per_row(q.reshape(B * S * IDX_N_HEADS, IDX_HEAD_DIM))
    q_i8 = q_i8.view(B, S, IDX_N_HEADS, IDX_HEAD_DIM)
    q_scale = q_scale.view(B, S, IDX_N_HEADS, 1)

    for b in range(bsz):
        cache_len = int(kv_seq_lens[b].item()) // ratio
        if cache_len <= 0:
            continue

        kv_i8_rows = []
        kv_scale_rows = []
        for slot in range(cache_len):
            blk_id = int(idx_block_table[b, slot // BLOCK_SIZE].item())
            kv_i8_rows.append(idx_kv_cache_i8[blk_id, slot % BLOCK_SIZE, 0])
            kv_scale_rows.append(idx_kv_scale[blk_id, slot % BLOCK_SIZE, 0, 0])
        kv_i8 = torch.stack(kv_i8_rows, dim=0).view(cache_len, IDX_HEAD_DIM)
        kv_scale = torch.stack(kv_scale_rows, dim=0).view(cache_len, 1)
        score_i32 = torch.einsum("shd,td->sht", q_i8[b].to(torch.int32), kv_i8.to(torch.int32))
        score = score_i32.float() * q_scale[b]
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


def _scalar_input_as_int(value, name):
    """Decode a scalar harness input and return either its value or an error string."""
    if hasattr(value, "numel") and value.numel() != 1:
        return None, f"{name} must be scalar, got shape {tuple(value.shape)}"
    if hasattr(value, "item"):
        value = value.item()
    elif hasattr(value, "value"):
        value = value.value
    try:
        return int(value), None
    except (TypeError, ValueError):
        return None, f"{name} must be integer-like, got {value!r}"


def topk_prefix_contract_error(
    topk_indices,
    position_ids,
    kv_seq_lens,
    offset,
):
    """Return an error string for an invalid decode top-k prefix or ``-1`` tail."""
    import torch

    if topk_indices.ndim != 3 or topk_indices.shape != (B, S, SCORE_LEN):
        return (
            f"top-k tensor must have shape {(B, S, SCORE_LEN)}, "
            f"got {tuple(topk_indices.shape)}"
        )
    if position_ids.ndim != 2 or position_ids.shape != (B, S):
        return f"position_ids must have shape {(B, S)}, got {tuple(position_ids.shape)}"
    if kv_seq_lens.ndim != 1 or kv_seq_lens.shape[0] != B:
        return f"kv_seq_lens must have shape {(B,)}, got {tuple(kv_seq_lens.shape)}"

    for b in range(B):
        cache_len = max(int(kv_seq_lens[b].item()) // COMPRESS_RATIO, 0)
        for s in range(S):
            row = topk_indices[b, s]
            position_visible = max(
                (int(position_ids[b, s].item()) + 1) // COMPRESS_RATIO,
                0,
            )
            visible = min(cache_len, position_visible, SCORE_LEN)
            prefix_len = min(IDX_TOPK, visible)
            prefix = row[:prefix_len]
            if prefix_len:
                out_of_range = (prefix < offset) | (prefix >= offset + visible)
                if out_of_range.any().item():
                    return (
                        f"top-k row [{b},{s}] has "
                        f"{int(out_of_range.count_nonzero().item())} entries outside "
                        f"[{offset}, {offset + visible}) in its active prefix"
                    )
                unique_count = int(torch.unique(prefix).numel())
                if unique_count != prefix_len:
                    return (
                        f"top-k row [{b},{s}] active prefix has "
                        f"{unique_count}/{prefix_len} unique entries"
                    )
            tail_non_padding = int((row[prefix_len:] != -1).count_nonzero().item())
            if tail_non_padding:
                return (
                    f"top-k row [{b},{s}] tail contains "
                    f"{tail_non_padding} non--1 entries"
                )
    return None


def decode_topk_compare(
    actual,
    expected,
    *,
    actual_outputs,
    expected_outputs,
    inputs,
    rtol,
    atol,
):
    """Validate decode top-k structure and allow only score-bounded tie changes."""
    import torch

    max_show = 10

    position_ids = inputs.get("position_ids")
    kv_seq_lens = inputs.get("kv_seq_lens")
    offset_raw = inputs.get("offset", OFFSET)
    if position_ids is None or kv_seq_lens is None:
        return False, (
            "    compare_fn requires position_ids and kv_seq_lens inputs"
        )
    offset, scalar_error = _scalar_input_as_int(offset_raw, "offset")
    if scalar_error:
        return False, f"    {scalar_error}"

    actual = actual.cpu()
    expected = expected.cpu()
    position_ids = position_ids.cpu()
    kv_seq_lens = kv_seq_lens.cpu()
    for label, indices in (("actual", actual), ("expected", expected)):
        contract_error = topk_prefix_contract_error(
            indices,
            position_ids,
            kv_seq_lens,
            offset,
        )
        if contract_error:
            return False, f"    {label} {contract_error}"

    scores = {}
    for label, outputs in (("actual", actual_outputs), ("expected", expected_outputs)):
        score = outputs.get("score")
        if score is None:
            return False, f"    compare_fn misconfigured: missing {label} output 'score'"
        score = score.cpu().to(torch.float32)
        if score.shape != (B, S, SCORE_LEN):
            return False, (
                f"    {label} score must have shape {(B, S, SCORE_LEN)}, "
                f"got {tuple(score.shape)}"
            )
        nonfinite = ~torch.isfinite(score)
        if nonfinite.any().item():
            return False, (
                f"    {label} score contains "
                f"{int(nonfinite.count_nonzero().item())} non-finite value(s)"
            )
        scores[label] = score

    actual_score = scores["actual"]
    expected_score = scores["expected"]
    failures = []
    failure_count = 0

    def record_failure(detail):
        nonlocal failure_count
        failure_count += 1
        if len(failures) < max_show:
            failures.append(detail)

    for b in range(B):
        cache_len = max(int(kv_seq_lens[b].item()) // COMPRESS_RATIO, 0)
        for s in range(S):
            position_visible = max(
                (int(position_ids[b, s].item()) + 1) // COMPRESS_RATIO,
                0,
            )
            visible = min(cache_len, position_visible, SCORE_LEN)
            k = min(IDX_TOPK, visible)
            if k <= 0:
                continue

            actual_order = actual[b, s, :k].to(torch.int64) - offset
            expected_order = expected[b, s, :k].to(torch.int64) - offset
            mismatch = actual_order != expected_order
            if not mismatch.any().item():
                continue

            actual_row = actual_score[b, s, :visible]
            expected_row = expected_score[b, s, :visible]
            candidate_tolerance = SCORE_ATOL + SCORE_RTOL * torch.maximum(
                actual_row.abs(),
                expected_row.abs(),
            )
            candidate_hard_tolerance = torch.maximum(
                SCORE_HARD_MULTIPLIER * candidate_tolerance,
                torch.full_like(candidate_tolerance, SCORE_HARD_ATOL),
            )
            candidate_error = (actual_row - expected_row).abs()

            # A score outlier that fits the score tensor's global ratio budget
            # must not justify a changed selection or order.
            displaced = torch.unique(
                torch.cat((actual_order[mismatch], expected_order[mismatch]))
            )
            bad_candidates = displaced[
                candidate_error[displaced]
                > candidate_hard_tolerance[displaced]
            ]
            for candidate_tensor in bad_candidates:
                candidate = int(candidate_tensor.item())
                record_failure(
                    f"row=[{b},{s}] candidate={candidate} score error "
                    f"{float(candidate_error[candidate].item()):.6g} exceeds "
                    f"candidate hard bound "
                    f"{float(candidate_hard_tolerance[candidate].item()):.6g} "
                    f"(actual={float(actual_row[candidate].item()):.6g}, "
                    f"expected={float(expected_row[candidate].item()):.6g})"
                )

            # Every earlier/later pair emitted by the kernel must remain
            # descending under its own score uncertainty. A prefix minimum of
            # the earlier upper bounds checks all pairs in O(k), without
            # constructing a 1024-by-1024 rank-inversion matrix.
            if k >= 2:
                for label, row in (
                    ("expected", expected_row),
                    ("actual", actual_row),
                ):
                    selected_lower = (
                        row[actual_order]
                        - candidate_hard_tolerance[actual_order]
                    )
                    selected_upper = (
                        row[actual_order]
                        + candidate_hard_tolerance[actual_order]
                    )
                    prior_min_upper = torch.cummin(
                        selected_upper[:-1],
                        dim=0,
                    ).values
                    clear_reverse = (
                        selected_lower[1:] > prior_min_upper
                    ).nonzero(as_tuple=False).flatten()
                    for position_tensor in clear_reverse:
                        later_position = int(position_tensor.item()) + 1
                        earlier_position = int(
                            selected_upper[:later_position].argmin().item()
                        )
                        earlier_candidate = int(
                            actual_order[earlier_position].item()
                        )
                        later_candidate = int(
                            actual_order[later_position].item()
                        )
                        reverse_gap = float(
                            (
                                row[later_candidate]
                                - row[earlier_candidate]
                            ).item()
                        )
                        joint_bound = float(
                            (
                                candidate_hard_tolerance[earlier_candidate]
                                + candidate_hard_tolerance[later_candidate]
                            ).item()
                        )
                        record_failure(
                            f"row=[{b},{s}] clear inversion on {label} "
                            f"scores at positions {earlier_position}/"
                            f"{later_position}: candidate "
                            f"{earlier_candidate} precedes {later_candidate}; "
                            f"reverse_gap={reverse_gap:.6g}, "
                            f"joint_bound={joint_bound:.6g}"
                        )

            # When visible > k, an internally sorted but wholly inferior set
            # would pass an order-only check. Compare candidates crossing the
            # top-k boundary using both the reference and emitted score grids.
            if visible > k:
                missing_mask = ~torch.isin(expected_order, actual_order)
                added_mask = ~torch.isin(actual_order, expected_order)
                missing = expected_order[missing_mask]
                added = actual_order[added_mask]
                if missing.numel() and added.numel():
                    for label, row in (
                        ("expected", expected_row),
                        ("actual", actual_row),
                    ):
                        missing_lower = (
                            row[missing] - candidate_hard_tolerance[missing]
                        )
                        added_upper = (
                            row[added] + candidate_hard_tolerance[added]
                        )
                        best_missing_pos = int(missing_lower.argmax().item())
                        worst_added_pos = int(added_upper.argmin().item())
                        boundary_gap = float(
                            (
                                missing_lower[best_missing_pos]
                                - added_upper[worst_added_pos]
                            ).item()
                        )
                        if boundary_gap > 0:
                            missing_candidate = int(missing[best_missing_pos].item())
                            added_candidate = int(added[worst_added_pos].item())
                            record_failure(
                                f"row=[{b},{s}] clear top-k boundary miss on "
                                f"{label} scores: omitted candidate "
                                f"{missing_candidate}, selected candidate "
                                f"{added_candidate}, uncertainty-adjusted "
                                f"gap={boundary_gap:.6g}"
                            )

    if not failure_count:
        return True, ""
    lines = [
        "    decode top-k differs outside candidate-specific score bounds: "
        f"{failure_count} failure(s) "
        f"(score_atol={SCORE_ATOL} score_rtol={SCORE_RTOL} "
        f"score_hard_multiplier={SCORE_HARD_MULTIPLIER} "
        f"score_hard_atol={SCORE_HARD_ATOL})"
    ]
    lines.extend(f"      {detail}" for detail in failures)
    if failure_count > len(failures):
        lines.append(f"      ... and {failure_count - len(failures)} more")
    return False, "\n".join(lines)


decode_topk_compare.__name__ = "decode_topk_pair_compare"


def score_valid_compare(
    actual,
    expected,
    *,
    actual_outputs,
    expected_outputs,
    inputs,
    rtol,
    atol,
):
    """Compare visible scores with a ratio budget and hard pointwise ceiling."""
    import torch

    from golden import ratio_allclose

    if actual.shape != expected.shape:
        return False, (
            f"    score shape mismatch: actual={tuple(actual.shape)} "
            f"expected={tuple(expected.shape)}"
        )
    actual_f = actual.cpu().to(torch.float32)
    expected_f = expected.cpu().to(torch.float32)
    for label, value in (("actual", actual_f), ("expected", expected_f)):
        nonfinite = ~torch.isfinite(value)
        if nonfinite.any().item():
            return False, (
                f"    {label} score contains "
                f"{int(nonfinite.count_nonzero().item())} non-finite value(s)"
            )
    valid = expected_f != FP32_NEG_INF
    actual_valid = actual_f[valid]
    expected_valid = expected_f[valid]
    base_ok, base_detail = ratio_allclose(atol=SCORE_ATOL, rtol=SCORE_RTOL)(
        actual_valid,
        expected_valid,
        actual_outputs=actual_outputs,
        expected_outputs=expected_outputs,
        inputs=inputs,
        rtol=rtol,
        atol=atol,
    )
    if not base_ok or actual_valid.numel() == 0:
        return base_ok, base_detail

    diff = (actual_valid - expected_valid).abs()
    hard_tolerance = torch.maximum(
        SCORE_HARD_MULTIPLIER * (
            SCORE_ATOL
            + SCORE_RTOL * torch.maximum(
                actual_valid.abs(),
                expected_valid.abs(),
            )
        ),
        torch.full_like(actual_valid, SCORE_HARD_ATOL),
    )
    hard_bad = diff > hard_tolerance
    hard_bad_count = int(hard_bad.count_nonzero().item())
    if not hard_bad_count:
        return True, ""

    flat_bad = hard_bad.nonzero(as_tuple=False).flatten()
    lines = []
    for index in flat_bad[:10].tolist():
        lines.append(
            f"      [{index}] actual={float(actual_valid[index]):.8g} "
            f"expected={float(expected_valid[index]):.8g} "
            f"diff={float(diff[index]):.4g} "
            f"hard_tol={float(hard_tolerance[index]):.4g}"
        )
    return False, (
        f"    score hard bound exceeded at {hard_bad_count} point(s): "
        f"hard_multiplier={SCORE_HARD_MULTIPLIER}, score_atol={SCORE_ATOL}, "
        f"score_rtol={SCORE_RTOL}, hard_atol={SCORE_HARD_ATOL}\n"
        + "\n".join(lines)
    )


score_valid_compare.__name__ = "score_valid_region_compare"


def build_tensor_specs(start_pos=None):
    import torch  # type: ignore[import]
    from expert_shared import _gen_mxfp8_weight_kn
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

    wq_b_fp8, wq_b_scale = _gen_mxfp8_weight_kn(
        (Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM), dequant_std=0.108, chan_cv=0.56
    )
    qr_i8, qr_scale = _int8_quant_per_row(init_qr())

    # C8 indexer cache fixture: INT8 + scale from one bf16-rounded random draw
    idx_kv_cache_bf16 = torch.rand(IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM).to(torch.bfloat16)
    idx_kv_i8, idx_kv_sc = _int8_quant_per_row(
        idx_kv_cache_bf16.float().reshape(IDX_CACHE_BLOCK_NUM * BLOCK_SIZE, IDX_HEAD_DIM))
    idx_kv_i8 = idx_kv_i8.view(IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM)
    idx_kv_sc = idx_kv_sc.view(IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1)

    return [
        TensorSpec("x", [B, S, D], torch.bfloat16, init_value=init_x),
        TensorSpec("qr", [T, Q_LORA], torch.int8, init_value=lambda: qr_i8),
        TensorSpec("qr_scale", [T, 1], torch.float32, init_value=lambda: qr_scale),
        TensorSpec("wq_b", [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], torch.float8_e4m3fn, init_value=lambda: wq_b_fp8),
        TensorSpec("wq_b_scale", [WQB_SCALE_ROWS, MX_N_TILE], torch.float8_e8m0fnu, init_value=lambda: wq_b_scale),
        TensorSpec("weights_proj", [D, IDX_N_HEADS], torch.bfloat16, init_value=init_weights_proj),
        TensorSpec("cos", [B, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_cos),
        TensorSpec("sin", [B, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_sin),
        TensorSpec("hadamard", [IDX_HEAD_DIM, IDX_HEAD_DIM], torch.bfloat16, init_value=init_hadamard),
        TensorSpec("inner_kv", [B, S, INNER_HEAD_DIM], torch.float32),
        TensorSpec("inner_compress_state", [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], torch.float32, init_value=init_inner_compress_state, is_output=True),
        TensorSpec("inner_compress_state_block_table", [B, INNER_STATE_MAX_BLOCKS], torch.int32, init_value=init_inner_compress_state_block_table),
        TensorSpec("inner_wkv", [INNER_OUT_DIM, D], torch.bfloat16, init_value=init_inner_wkv),
        TensorSpec("inner_wgate", [INNER_OUT_DIM, D], torch.bfloat16, init_value=init_inner_wgate),
        TensorSpec("inner_ape", [COMPRESS_RATIO, INNER_OUT_DIM], torch.float32, init_value=init_inner_ape),
        TensorSpec("inner_norm_w", [INNER_HEAD_DIM], torch.bfloat16, init_value=init_inner_norm_w),
        TensorSpec("idx_kv_cache", [IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], torch.int8, init_value=lambda: idx_kv_i8, is_output=True),
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
    from decode_indexer_compressor import (
        mapped_idx_cache_ratio_allclose,
        mapped_inner_state_ratio_allclose,
    )
    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a5",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", type=int, default=0, choices=[0, 1, 2],
                        help="L2 swimlane level: 0=off, 1=AICore timing, 2=+AICPU timing.")
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--start-pos", type=int, default=None,
                        help="Uniform fixture-only start_pos override for all batches; "
                             "default (unset) uses the canonical per-batch CSA set that includes the 8k point.")
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

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
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "score":        score_valid_compare,
            "topk_idxs":    decode_topk_compare,
            "inner_compress_state": mapped_inner_state_ratio_allclose(
                atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            # Ratio budgets apply only to rows written by the current slot mappings;
            # every historical/unallocated physical row must remain bitwise exact.
            "idx_kv_cache": mapped_idx_cache_ratio_allclose(
                atol=1, rtol=0, max_error_ratio=0.01),
            "idx_kv_scale": mapped_idx_cache_ratio_allclose(
                atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
