# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 packed prefill indexer.

This module builds the compressed index KV cache and per-token compressed top-k
indices consumed by packed CSA prefill sparse attention.
"""

import pypto.language as pl

from decode_indexer import (
    _fp8_quant_query_per_row,
    _golden_index_query_projection,
    _native_index_query_fixture,
    project_index_query,
)

from config import (
    ACTIVE as M,
    BLOCK_SIZE,
    CSA_INNER_STATE_PHYSICAL_BLOCKS,
    FP32_NEG_INF,
    IDX_CACHE_MAX_BLOCKS,
    PREFILL_IDX_BLOCK_NUM,
)
from prefill_indexer_compressor import (
    INNER_STATE_BLOCK_NUM,
    INNER_STATE_BLOCK_SIZE,
    INNER_STATE_MAX_BLOCKS,
    STATE_LEN as INNER_STATE_LEN,
    golden_prefill_indexer_compressor,
    mapped_idx_cache_ratio_allclose,
    mapped_inner_state_ratio_allclose,
    prefill_indexer_compressor,
)

# The indexer receives allocator-managed global cache and inner-state pools.
IDX_BLOCK_NUM_DYN = pl.dynamic("PREFILL_IDX_BLOCK_NUM_DYN")
INNER_STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_INNER_STATE_BLOCK_NUM_DYN")

# model config (mirrors decode_indexer)
D = M.hidden_size
ROPE_HEAD_DIM = M.qk_rope_head_dim
IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
IDX_NOPE_HEAD_DIM = M.index_nope_head_dim
Q_LORA = M.q_lora_rank
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
assert INDEXER_TOPK_CAP == INDEXER_SCORE_CAP, (
    "the standalone top-k contract relies on selecting every visible score"
)
MAX_CMP_WRITES = max(1, T // COMPRESS_RATIO)

# Near-zero scores occasionally differ by roughly 5.6e-3 because the CPU and
# AIC reductions quantize at different boundaries. Keep one shared hard floor
# for the score and top-k comparators so their uncertainty models cannot drift.
PREFILL_SCORE_HARD_ATOL = 7e-3

# Q-projection / score tiling (mirrors decode_indexer)
Q_OUT_TILE = 256
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


@pl.jit.inline
def prefill_indexer(
    x: pl.Tensor[[T, D], pl.BF16],
    qr: pl.Tensor[[T, Q_LORA], pl.FP8E4M3FN],
    qr_scale: pl.Tensor[[128, Q_LORA // 32], pl.FP8E8M0, pl.MX_A_ZZ],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[T, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[T, ROPE_HEAD_DIM // 2], pl.FP32],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.InOut[
        pl.Tensor[
            [INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM], pl.FP32
        ]
    ],
    inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.BF16],
    # FP8 cache written by the indexer compressor with per-position FP32 scales.
    idx_kv_cache: pl.Out[pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.FP8E4M3FN]],
    idx_kv_scale: pl.Out[pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    score: pl.Out[pl.Tensor[[T, INDEXER_SCORE_CAP], pl.FP32]],
    cmp_topk_indices: pl.Out[pl.Tensor[[T, IDX_TOPK], pl.INT32]],
    position_ids: pl.Tensor[[T], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    idx_slot_mapping: pl.Tensor[[T], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    qr_proj = pl.create_tensor([T, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.BF16)
    query_done = project_index_query(qr, qr_scale, wq_b, wq_b_scale, qr_proj, late_dep)

    # Apply Q RoPE to one token per task.
    qr_proj_flat = pl.reshape(qr_proj, [T * IDX_N_HEADS, IDX_HEAD_DIM])
    qr_rope_out = pl.create_tensor([T * IDX_N_HEADS, ROPE_HEAD_DIM], dtype=pl.BF16)
    with pl.spmd(T * IDX_N_HEADS // ROPE_ROW_BLOCK, name_hint="prefill_idx_qr_rope", deps=[query_done]) as _query_rope_tid:
        idx = pl.tile.get_block_idx()
        o0 = idx * ROPE_ROW_BLOCK
        token_idx = idx  # ROPE_ROW_BLOCK == IDX_N_HEADS, so one task == one token
        cos_b = cos[token_idx : token_idx + 1, 0 : ROPE_HEAD_DIM // 2]
        sin_b = sin[token_idx : token_idx + 1, 0 : ROPE_HEAD_DIM // 2]
        cos_b32 = pl.col_expand_mul(pl.full([ROPE_ROW_TILE, ROPE_HEAD_DIM // 2], dtype=pl.FP32, value=1.0), cos_b)
        sin_b32 = pl.col_expand_mul(pl.full([ROPE_ROW_TILE, ROPE_HEAD_DIM // 2], dtype=pl.FP32, value=1.0), sin_b)
        sin_b32_neg = pl.neg(sin_b32)
        cos_il = pl.full([ROPE_ROW_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        cos_il = pl.tensor.scatter(cos_b32, mask_pattern=pl.tile.MaskPattern.P0101, dst=cos_il)
        cos_il = pl.tensor.scatter(cos_b32, mask_pattern=pl.tile.MaskPattern.P1010, dst=cos_il)
        sin_signed = pl.full([ROPE_ROW_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        sin_signed = pl.tensor.scatter(sin_b32_neg, mask_pattern=pl.tile.MaskPattern.P0101, dst=sin_signed)
        sin_signed = pl.tensor.scatter(sin_b32, mask_pattern=pl.tile.MaskPattern.P1010, dst=sin_signed)
        for ro in pl.range(0, ROPE_ROW_BLOCK, ROPE_ROW_TILE):
            r0 = o0 + ro
            qr_rope_bf16 = qr_proj_flat[r0 : r0 + ROPE_ROW_TILE, IDX_NOPE_HEAD_DIM : IDX_HEAD_DIM]
            qr_rope_slice = pl.cast(qr_rope_bf16, target_type=pl.FP32)
            qr_even = pl.gather(qr_rope_slice, mask_pattern=pl.tile.MaskPattern.P0101)
            qr_odd = pl.gather(qr_rope_slice, mask_pattern=pl.tile.MaskPattern.P1010)
            qr_swap_zero = pl.full([ROPE_ROW_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
            qr_swapped = pl.tensor.scatter(qr_odd, mask_pattern=pl.tile.MaskPattern.P0101, dst=qr_swap_zero)
            qr_swapped = pl.tensor.scatter(qr_even, mask_pattern=pl.tile.MaskPattern.P1010, dst=qr_swapped)
            rope_rot = pl.add(pl.mul(qr_rope_slice, cos_il), pl.mul(qr_swapped, sin_signed))
            qr_rope_out[r0 : r0 + ROPE_ROW_TILE, :] = pl.cast(rope_rot, target_type=pl.BF16, mode="rint")

    # Finish every native MX query block before the ordinary BF16 matmuls.
    # Q Hadamard BF16 boundary is followed by per-head/token FP8 quantization.
    qr_hadamard_fp8 = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.FP8E4M3FN)
    qr_hadamard_scale_dq = pl.create_tensor([T * IDX_N_HEADS, 1], dtype=pl.FP32)
    with pl.spmd(T * IDX_N_HEADS // QH_QUANT_BLOCK, name_hint="prefill_idx_qr_hadamard_quant", deps=[query_done]) as _query_hadamard_tid:
        idx = pl.tile.get_block_idx()
        o0 = idx * QH_QUANT_BLOCK
        for ro in pl.range(0, QH_QUANT_BLOCK, QH_QUANT_ROW_TILE):
            qh_nope = qr_proj_flat[o0 + ro : o0 + ro + QH_QUANT_ROW_TILE, 0 : IDX_NOPE_HEAD_DIM]
            qh_rope = qr_rope_out[o0 + ro : o0 + ro + QH_QUANT_ROW_TILE, :]
            qh_acc = pl.matmul(qh_nope, hadamard[0 : IDX_NOPE_HEAD_DIM, :], out_dtype=pl.FP32)
            qh_acc = pl.matmul_acc(qh_acc, qh_rope, hadamard[IDX_NOPE_HEAD_DIM : IDX_HEAD_DIM, :])
            qh_bf16 = pl.cast(qh_acc, target_type=pl.BF16, mode="rint")
            qh_values = pl.cast(qh_bf16, target_type=pl.FP32)
            qh_amax = pl.row_max(pl.maximum(qh_values, pl.neg(qh_values)))
            qh_scale = pl.div(qh_amax, 448.0)
            qr_hadamard_scale_dq[o0 + ro : o0 + ro + QH_QUANT_ROW_TILE, :] = qh_scale
            # A zero query retains scale 0 and emits zero FP8 payload.
            qh_nonzero_bits = pl.minimum(pl.reinterpret_view(qh_amax, pl.INT32), 1)
            qh_nonzero = pl.cast(qh_nonzero_bits, target_type=pl.FP32)
            qh_zero_adjustment = pl.sub(qh_nonzero, 1.0)
            qh_safe_scale = pl.sub(qh_scale, qh_zero_adjustment)
            qh_normalized = pl.row_expand_div(qh_values, qh_safe_scale)
            qh_clipped = pl.minimum(pl.maximum(qh_normalized, -448.0), 448.0)
            qh_fp8 = pl.cast(qh_clipped, target_type=pl.FP8E4M3FN, mode="rint")
            qr_hadamard_fp8[o0 + ro : o0 + ro + QH_QUANT_ROW_TILE, :] = qh_fp8

    # === weights projection: (x @ weights_proj) * WEIGHTS_SCALE ===
    weights = pl.create_tensor([T, IDX_N_HEADS], dtype=pl.FP32)
    with pl.spmd(T // WEIGHTS_ROW_TILE, name_hint="prefill_idx_weights_proj", deps=[query_done]) as _weights_tid:
        idx = pl.tile.get_block_idx()
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

    # Finish the BF16 query producers before the inner compressor phase.
    inner_start = pl.system.task_dummy(deps=[_query_hadamard_tid, _weights_tid])
    # === inner compressor: build the paged compressed index KV cache ===
    compressor_completion = pl.array.create(1, pl.TASK_ID)
    prefill_indexer_compressor(
        x,
        inner_compress_state, inner_compress_state_block_table,
        inner_wkv, inner_wgate, inner_ape, inner_norm_w,
        freqs_cos, freqs_sin, hadamard,
        idx_kv_cache, idx_kv_scale, idx_block_table,
        position_ids, num_tokens,
        idx_slot_mapping, inner_state_slot_mapping,
        compressor_completion, inner_start,
    )

    # FP8 Q/cache dot products accumulate in FP32; both row scales and head
    # weights are applied before the causal top-k selection.
    idx_block_num = pl.tensor.dim(idx_kv_cache, 0)
    kv_cache_fp8_flat = pl.reshape(idx_kv_cache, [idx_block_num * BLOCK_SIZE, IDX_HEAD_DIM])
    kv_scale_flat = pl.reshape(idx_kv_scale, [idx_block_num * BLOCK_SIZE, 1])
    score_wide = pl.create_tensor([T, SORT_LEN], dtype=pl.FP32)                                  # wide sort scratch

    for si in pl.parallel(0, T, SCORE_INIT_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_idx_score_init"):
            score_wide[si : si + SCORE_INIT_TILE, :] = pl.full([SCORE_INIT_TILE, SORT_LEN], dtype=pl.FP32, value=FP32_NEG_INF)

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_idx_score",
        deps=[compressor_completion[0]],
    ):
        last_pos = pl.read(position_ids, [num_tokens - 1])
        max_visible = pl.min((last_pos + 1) // COMPRESS_RATIO, INDEXER_SCORE_CAP)
        for cb in pl.range(INDEXER_SCORE_BLOCKS):
            cache0 = cb * CACHE_TILE
            if max_visible > cache0:
                idx_blk_id = pl.cast(pl.read(idx_block_table, [cache0 // BLOCK_SIZE]), pl.INDEX)
                kv_row0 = idx_blk_id * BLOCK_SIZE + (cache0 % BLOCK_SIZE)
                # C8: the compressor stored this block as FP8 + a per-position dequant scale; read
                # both from the paged cache directly (no score-time re-quant).
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

    return idx_kv_cache, idx_kv_scale, inner_compress_state, score, cmp_topk_indices


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

    # Score the paged FP8 index cache and select a causal top-k per token.
    num_tokens = int(tensors["num_tokens"])
    position_ids = tensors["position_ids"].long()
    rd = ROPE_HEAD_DIM
    cmp_topk_indices = torch.full((T, IDX_TOPK), -1, dtype=torch.int32)
    score_full = torch.full((T, INDEXER_SCORE_CAP), FP32_NEG_INF, dtype=torch.float32)
    visible = ((position_ids + 1) // COMPRESS_RATIO).clamp(max=INDEXER_SCORE_CAP)
    max_visible = int(visible[:num_tokens].max().item()) if num_tokens > 0 else 0
    if max_visible == 0:
        return cmp_topk_indices, score_full

    # Native MX query projection rounds to BF16 before interleaved RoPE.
    hadamard = tensors["hadamard"].float()
    cos = tensors["cos"].float().view(T, 1, -1)
    sin = tensors["sin"].float().view(T, 1, -1)
    q = _golden_index_query_projection(tensors).view(T, IDX_N_HEADS, IDX_HEAD_DIM)
    q_pair = q[..., -rd:].unflatten(-1, (-1, 2))
    q0, q1 = q_pair[..., 0], q_pair[..., 1]
    y0 = (q0 * cos - q1 * sin).to(torch.bfloat16)
    y1 = (q0 * sin + q1 * cos).to(torch.bfloat16)
    q = torch.cat([q[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1)
    q = (q.to(torch.bfloat16).float() @ hadamard).to(torch.bfloat16).float()

    weights = (tensors["x"].float() @ tensors["weights_proj"].float()) * WEIGHTS_SCALE  # [T, heads]

    # C8: the compressor already stored FP8 KV + a per-position dequant scale. Gather both in
    # compressed-position order through the paged block table (no score-time re-quant).
    idx_kv_cache = tensors["idx_kv_cache"]
    idx_kv_scale = tensors["idx_kv_scale"]
    cache_flat_fp8 = idx_kv_cache.reshape(idx_kv_cache.shape[0] * BLOCK_SIZE, IDX_HEAD_DIM)
    scale_flat = idx_kv_scale.float().reshape(idx_kv_scale.shape[0] * BLOCK_SIZE, 1)
    idx_block_table = tensors["idx_block_table"]
    rows = [
        int(idx_block_table[c // BLOCK_SIZE].item()) * BLOCK_SIZE + (c % BLOCK_SIZE)
        for c in range(max_visible)
    ]
    kv_fp8 = torch.stack([cache_flat_fp8[r] for r in rows], dim=0).float()  # [max_visible, dim]
    kv_sc = torch.stack([scale_flat[r] for r in rows], dim=0).view(1, 1, max_visible)

    # FP8 dot products and both dequantization scales feed the FP32 weighted reduction.
    q_fp8, q_sc = _fp8_quant_query_per_row(q.reshape(T * IDX_N_HEADS, IDX_HEAD_DIM))
    q_fp8 = q_fp8.view(T, IDX_N_HEADS, IDX_HEAD_DIM).float()
    q_sc = q_sc.view(T, IDX_N_HEADS, 1)
    score_dot = torch.einsum("thd,cd->thc", q_fp8, kv_fp8)
    score = score_dot.float() * q_sc * kv_sc
    score = (torch.relu(score) * weights.unsqueeze(-1)).sum(dim=1)  # [T, max_visible]

    # Per-token causal mask, then top-k over the visible compressed positions.
    col = torch.arange(max_visible).unsqueeze(0)
    score = score.masked_fill(col >= visible.unsqueeze(1), FP32_NEG_INF)
    score_full[:num_tokens, :max_visible] = score[:num_tokens]
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


def topk_prefix_contract_error(topk_indices, position_ids, num_tokens):
    """Return an error string if an active top-k prefix or its -1 padding is invalid."""
    import torch

    if hasattr(num_tokens, "numel") and num_tokens.numel() != 1:
        return f"num_tokens must be scalar, got shape {tuple(num_tokens.shape)}"
    if hasattr(num_tokens, "item"):
        num_tokens = num_tokens.item()
    num_tokens = int(num_tokens)
    if topk_indices.ndim != 2 or topk_indices.shape != (T, INDEXER_TOPK_CAP):
        return (
            f"top-k tensor must have shape {(T, INDEXER_TOPK_CAP)}, "
            f"got {tuple(topk_indices.shape)}"
        )
    if position_ids.ndim != 1 or position_ids.shape[0] != T:
        return f"position_ids must have shape {(T,)}, got {tuple(position_ids.shape)}"
    if not 0 <= num_tokens <= T:
        return f"num_tokens={num_tokens} is outside [0, {T}]"

    for t in range(T):
        row = topk_indices[t]
        if t >= num_tokens:
            non_padding = int((row != -1).count_nonzero().item())
            if non_padding:
                return f"inactive top-k row {t} contains {non_padding} non--1 entries"
            continue

        visible = min(
            max(int((int(position_ids[t].item()) + 1) // COMPRESS_RATIO), 0),
            INDEXER_SCORE_CAP,
        )
        prefix = row[:visible]
        if visible:
            out_of_range = int(
                ((prefix < 0) | (prefix >= visible)).count_nonzero().item()
            )
            if out_of_range:
                return (
                    f"top-k row {t} has {out_of_range} entries outside "
                    f"[0, {visible}) in its visible prefix"
                )
            unique_count = int(torch.unique(prefix).numel())
            if unique_count != visible:
                return (
                    f"top-k row {t} visible prefix has "
                    f"{unique_count}/{visible} unique entries"
                )
        tail_non_padding = int((row[visible:] != -1).count_nonzero().item())
        if tail_non_padding:
            return f"top-k row {t} tail contains {tail_non_padding} non--1 entries"
    return None


def prefill_topk_compare(
    num_tokens,
    *,
    score_atol=1e-4,
    score_rtol=1.0 / 128,
    score_hard_multiplier=4.0,
    score_hard_atol=PREFILL_SCORE_HARD_ATOL,
    max_show=10,
):
    """Validate top-k structure and allow only score-bounded near-tie swaps."""
    import torch

    if score_atol < 0 or score_rtol < 0:
        raise ValueError("top-k score tolerances must be non-negative")
    if score_hard_multiplier < 1:
        raise ValueError("score_hard_multiplier must be at least 1")
    if score_hard_atol < 0:
        raise ValueError("score_hard_atol must be non-negative")
    if max_show < 0:
        raise ValueError("max_show must be non-negative")

    def compare(
        actual,
        expected,
        *,
        actual_outputs,
        expected_outputs,
        inputs,
        rtol,
        atol,
    ):
        position_ids = inputs.get("position_ids")
        if position_ids is None:
            return False, "    compare_fn requires input 'position_ids'"

        actual = actual.cpu()
        expected = expected.cpu()
        position_ids = position_ids.cpu()
        for label, indices in (("actual", actual), ("expected", expected)):
            contract_error = topk_prefix_contract_error(indices, position_ids, num_tokens)
            if contract_error:
                return False, f"    {label} {contract_error}"

        scores = {}
        for label, outputs in (("actual", actual_outputs), ("expected", expected_outputs)):
            score = outputs.get("score")
            if score is None:
                return False, f"    compare_fn misconfigured: missing {label} output 'score'"
            score = score.cpu().to(torch.float32)
            if score.shape != (T, INDEXER_SCORE_CAP):
                return False, (
                    f"    {label} score must have shape {(T, INDEXER_SCORE_CAP)}, "
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

        for token in range(int(num_tokens)):
            visible = min(
                max(
                    int((int(position_ids[token].item()) + 1) // COMPRESS_RATIO),
                    0,
                ),
                INDEXER_SCORE_CAP,
            )
            if visible <= 1:
                continue

            actual_order = actual[token, :visible].long()
            expected_order = expected[token, :visible].long()
            mismatch = actual_order != expected_order
            if not mismatch.any().item():
                continue

            actual_row = actual_score[token, :visible]
            expected_row = expected_score[token, :visible]
            candidate_tolerance = score_atol + score_rtol * torch.maximum(
                actual_row.abs(),
                expected_row.abs(),
            )
            candidate_hard_tolerance = torch.maximum(
                score_hard_multiplier * candidate_tolerance,
                torch.full_like(candidate_tolerance, score_hard_atol),
            )
            candidate_error = (actual_row - expected_row).abs()

            # A score outlier elsewhere in the large score tensor may fit the
            # score output's ratio budget. It must not justify a changed top-k
            # order: every displaced candidate has to satisfy its own bound.
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
                    f"token={token} candidate={candidate} score error "
                    f"{float(candidate_error[candidate].item()):.6g} exceeds "
                    f"candidate hard bound "
                    f"{float(candidate_hard_tolerance[candidate].item()):.6g} "
                    f"(actual={float(actual_row[candidate].item()):.6g}, "
                    f"expected={float(expected_row[candidate].item()):.6g})"
                )

            # Convert the actual permutation to golden ranks. Every inversion
            # is a pair whose order changed. Such a pair is legal only when the
            # reference gap and any discrepancy between the emitted ordering
            # and actual scores both fit the two candidates' joint error band.
            expected_rank = torch.empty(visible, dtype=torch.int64)
            expected_rank[expected_order] = torch.arange(visible)
            rank_in_actual_order = expected_rank[actual_order]
            earlier = rank_in_actual_order.unsqueeze(1)
            later = rank_in_actual_order.unsqueeze(0)
            inversions = torch.triu(earlier > later, diagonal=1).nonzero(
                as_tuple=False
            )
            for inversion in inversions:
                actual_pos = int(inversion[0].item())
                later_pos = int(inversion[1].item())
                lower_candidate = int(actual_order[actual_pos].item())
                higher_candidate = int(actual_order[later_pos].item())
                joint_bound = float(
                    (
                        candidate_hard_tolerance[lower_candidate]
                        + candidate_hard_tolerance[higher_candidate]
                    ).item()
                )
                expected_gap = float(
                    (
                        expected_row[higher_candidate]
                        - expected_row[lower_candidate]
                    ).item()
                )
                actual_order_gap = float(
                    (
                        actual_row[higher_candidate]
                        - actual_row[lower_candidate]
                    ).item()
                )
                if expected_gap > joint_bound or actual_order_gap > joint_bound:
                    record_failure(
                        f"token={token} clear inversion: candidate "
                        f"{lower_candidate} precedes {higher_candidate}; "
                        f"expected_gap={expected_gap:.6g}, "
                        f"actual_order_gap={actual_order_gap:.6g}, "
                        f"joint_bound={joint_bound:.6g}"
                    )

        if not failure_count:
            return True, ""
        lines = [
            "    top-k order differs outside candidate-specific score bounds: "
            f"{failure_count} failure(s) "
            f"(score_atol={score_atol} score_rtol={score_rtol} "
            f"score_hard_multiplier={score_hard_multiplier} "
            f"score_hard_atol={score_hard_atol})"
        ]
        lines.extend(f"      {detail}" for detail in failures)
        if failure_count > len(failures):
            lines.append(f"      ... and {failure_count - len(failures)} more")
        return False, "\n".join(lines)

    compare.__name__ = f"prefill_topk_compare(num_tokens={num_tokens})"
    return compare


def prefill_score_compare(
    num_tokens,
    *,
    score_atol=1e-4,
    score_rtol=1.0 / 128,
    max_error_ratio=0.005,
    hard_multiplier=4.0,
    hard_atol=PREFILL_SCORE_HARD_ATOL,
    max_show=10,
):
    """Apply a ratio budget plus a hard per-score error ceiling."""
    import torch

    from golden import ratio_allclose

    if hard_multiplier < 1:
        raise ValueError("hard_multiplier must be at least 1")
    if hard_atol < 0:
        raise ValueError("hard_atol must be non-negative")
    base_compare = ratio_allclose(
        atol=score_atol,
        rtol=score_rtol,
        max_error_ratio=max_error_ratio,
        valid_rows=num_tokens,
    )

    def compare(
        actual,
        expected,
        *,
        actual_outputs,
        expected_outputs,
        inputs,
        rtol,
        atol,
    ):
        ok, detail = base_compare(
            actual,
            expected,
            actual_outputs=actual_outputs,
            expected_outputs=expected_outputs,
            inputs=inputs,
            rtol=rtol,
            atol=atol,
        )
        if not ok:
            return False, detail

        actual_all = actual.cpu().to(torch.float32)
        expected_all = expected.cpu().to(torch.float32)
        inactive = actual_all[num_tokens:]
        expected_inactive = expected_all[num_tokens:]
        inactive_mismatch = inactive != expected_inactive
        inactive_mismatch_count = int(inactive_mismatch.count_nonzero().item())
        if inactive_mismatch_count:
            return False, (
                f"    inactive score rows differ from the reference sentinel "
                f"at {inactive_mismatch_count} point(s)"
            )

        actual_f = actual_all[:num_tokens]
        expected_f = expected_all[:num_tokens]
        if actual_f.numel() == 0:
            return True, ""
        diff = (actual_f - expected_f).abs()
        hard_tolerance = torch.maximum(
            hard_multiplier * (
                score_atol + score_rtol * torch.maximum(
                    actual_f.abs(),
                    expected_f.abs(),
                )
            ),
            torch.full_like(actual_f, hard_atol),
        )
        hard_bad = diff > hard_tolerance
        hard_bad_count = int(hard_bad.count_nonzero().item())
        if hard_bad_count == 0:
            return True, ""

        flat_bad = hard_bad.flatten().nonzero(as_tuple=False).flatten()
        flat_actual = actual_f.flatten()
        flat_expected = expected_f.flatten()
        flat_diff = diff.flatten()
        flat_tolerance = hard_tolerance.flatten()
        lines = []
        for index in flat_bad[:max_show].tolist():
            lines.append(
                f"      [{index}] actual={float(flat_actual[index]):.8g} "
                f"expected={float(flat_expected[index]):.8g} "
                f"diff={float(flat_diff[index]):.4g} "
                f"hard_tol={float(flat_tolerance[index]):.4g}"
            )
        return False, (
            f"    score hard bound exceeded at {hard_bad_count} point(s): "
            f"hard_multiplier={hard_multiplier}, score_atol={score_atol}, "
            f"score_rtol={score_rtol}, hard_atol={hard_atol}\n"
            + "\n".join(lines)
        )

    compare.__name__ = (
        f"prefill_score_compare(num_tokens={num_tokens},"
        f"max_error_ratio={max_error_ratio},hard_multiplier={hard_multiplier},"
        f"hard_atol={hard_atol})"
    )
    return compare


@pl.jit
def prefill_indexer_test(
    x: pl.Tensor[[T, D], pl.BF16],
    qr: pl.Tensor[[T, Q_LORA], pl.FP8E4M3FN],
    qr_scale: pl.Tensor[[128, Q_LORA // 32], pl.FP8E8M0, pl.MX_A_ZZ],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[T, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[T, ROPE_HEAD_DIM // 2], pl.FP32],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.InOut[
        pl.Tensor[
            [INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM], pl.FP32
        ]
    ],
    inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.InOut[pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.FP8E4M3FN]],
    idx_kv_scale: pl.InOut[pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    score: pl.Out[pl.Tensor[[T, INDEXER_SCORE_CAP], pl.FP32]],
    topk_idxs: pl.Out[pl.Tensor[[T, INDEXER_SCORE_CAP], pl.INT32]],
    position_ids: pl.Tensor[[T], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    idx_slot_mapping: pl.Tensor[[T], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
):
    cmp_topk_indices = pl.create_tensor([T, IDX_TOPK], dtype=pl.INT32)
    late_dep = pl.system.task_dummy(deps=[])
    prefill_indexer(
        x, qr, qr_scale, wq_b, wq_b_scale, weights_proj,
        cos, sin, freqs_cos, freqs_sin, hadamard,
        inner_compress_state, inner_compress_state_block_table,
        inner_wkv, inner_wgate, inner_ape, inner_norm_w,
        idx_kv_cache, idx_kv_scale, idx_block_table,
        score, cmp_topk_indices,
        position_ids, num_tokens,
        idx_slot_mapping, inner_state_slot_mapping, late_dep,
    )
    # Expose the kernel's topk (first INDEXER_SCORE_CAP cols of cmp_topk_indices) as topk_idxs.
    for tb in pl.spmd(T // TOPK_TILE, name_hint="prefill_idx_topk_copy"):
        t0 = tb * TOPK_TILE
        for ti in pl.range(TOPK_TILE):
            t = t0 + ti
            topk_idxs[t : t + 1, 0:INDEXER_SCORE_CAP] = cmp_topk_indices[t : t + 1, 0:INDEXER_SCORE_CAP]
    return score, inner_compress_state, idx_kv_cache, idx_kv_scale, topk_idxs


def build_tensor_specs(start_pos: int = START_POS, num_tokens: int = T):
    import torch
    from golden import ScalarSpec, TensorSpec
    from indexer_cache_quant import reference_indexer_cache_quant as quant_cache
    from rope_tables import build_deepseek_v4_rope_tables, materialize_half_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_deepseek_v4_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)

    if not 1 <= num_tokens <= T:
        raise ValueError(f"num_tokens must satisfy 1 <= num_tokens <= {T}, got {num_tokens}")
    if start_pos < 0 or start_pos + T > MAX_SEQ_LEN:
        raise ValueError(f"start_pos must satisfy 0 <= start_pos <= {MAX_SEQ_LEN - T}, got {start_pos}")
    max_visible = (start_pos + num_tokens) // COMPRESS_RATIO
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
    # C8 historical index cache: completed compressed slots hold FP8 + a per-position dequant scale.
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
                hi8, hsc = quant_cache(hist_bf16.float().view(1, IDX_HEAD_DIM))
                c_flat[row] = hi8.view(IDX_HEAD_DIM)
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

    qr_fp8, qr_scale, wq_b_fp8, wq_b_scale = _native_index_query_fixture(T)

    return [
        TensorSpec("x", [T, D], torch.bfloat16, init_value=init_x),
        TensorSpec("qr", [T, Q_LORA], torch.float8_e4m3fn, init_value=lambda: qr_fp8),
        TensorSpec("qr_scale", [128, Q_LORA // 32], torch.float8_e8m0fnu, init_value=lambda: qr_scale),
        TensorSpec("wq_b", [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], torch.float8_e4m3fn, init_value=lambda: wq_b_fp8),
        TensorSpec("wq_b_scale", [Q_LORA // 32, IDX_N_HEADS * IDX_HEAD_DIM], torch.float8_e8m0fnu, init_value=lambda: wq_b_scale),
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
        TensorSpec("idx_kv_cache", [PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], torch.float8_e4m3fn, init_value=init_idx_kv_cache),
        TensorSpec("idx_kv_scale", [PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, 1], torch.float32, init_value=init_idx_kv_scale),
        TensorSpec("idx_block_table", [IDX_CACHE_MAX_BLOCKS], torch.int32, init_value=init_idx_block_table),
        TensorSpec("score", [T, INDEXER_SCORE_CAP], torch.float32),
        TensorSpec("topk_idxs", [T, INDEXER_SCORE_CAP], torch.int32),
        TensorSpec("position_ids", [T], torch.int32, init_value=init_position_ids),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
        TensorSpec("idx_slot_mapping", [T], torch.int64, init_value=init_idx_slot_mapping),
        TensorSpec("inner_state_slot_mapping", [T], torch.int64, init_value=init_inner_state_slot_mapping),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser(description="Standalone token-major DeepSeek V4 prefill indexer validation.")
    parser.add_argument("-p", "--platform", type=str, default="a5",
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
    parser.add_argument("--num-tokens", type=int, default=T,
                        help="Active token prefix; inactive top-k rows and slot mappings remain -1.")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        fn=prefill_indexer_test,
        specs=build_tensor_specs(args.start_pos, args.num_tokens),
        golden_fn=golden_prefill_indexer,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        config=dict(
            dump_passes=args.dump_passes,
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compile_only=args.compile_only,
        compare_fn={
            "score": prefill_score_compare(num_tokens=args.num_tokens),
            "topk_idxs": prefill_topk_compare(args.num_tokens),
            "inner_compress_state": mapped_inner_state_ratio_allclose(
                num_tokens=args.num_tokens, atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            # Apply the ratio budget only to rows written by the active mapping;
            # every historical/unallocated row must remain bitwise exact.
            "idx_kv_cache": mapped_idx_cache_ratio_allclose(
                num_tokens=args.num_tokens, atol=0.001953125, rtol=0.125,
                max_error_ratio=0.01),
            "idx_kv_scale": mapped_idx_cache_ratio_allclose(
                num_tokens=args.num_tokens, atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
