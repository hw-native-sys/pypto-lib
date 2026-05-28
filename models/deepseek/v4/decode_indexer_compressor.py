# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 KV Compressor (decode incremental, ratio=4 overlap).

Uses overlapping state layout with 8 slots.
Front slots 0-3 at columns [0:HEAD_DIM], back slots 4-7 at columns [HEAD_DIM:OUT_DIM].
Tree reduction for softmax+pool. State shift after compression."""


import pypto.language as pl

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ, BLOCK_SIZE, C4A_COMPRESSOR_BLOCK_SIZE, FP32_NEG_INF


# model config
B = DECODE_BATCH
S = DECODE_SEQ
EPS = M.rms_norm_eps
D = M.hidden_size
HEAD_DIM = M.index_head_dim
HEAD_DIM_INV = 1.0 / HEAD_DIM
ROPE_HEAD_DIM = M.qk_rope_head_dim
NOPE_HEAD_DIM = M.index_nope_head_dim
MAX_SEQ_LEN = M.max_position_embeddings

# kernel-local (ratio-4 overlapping compressor)
COMPRESS_RATIO = 4
ROTATE = True
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
OUT_DIM = COFF * HEAD_DIM
STATE_LEN = COFF * COMPRESS_RATIO
IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
COMPRESS_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
COMPRESS_STATE_MAX_BLOCKS = 64
COMPRESS_STATE_BLOCK_NUM = B * COMPRESS_STATE_MAX_BLOCKS
COMPRESS_STATE_DIM = 2 * OUT_DIM
IDX_CACHE_MAX_BLOCKS = 64
IDX_CACHE_BLOCK_NUM = B * IDX_CACHE_MAX_BLOCKS
START_POS = COMPRESS_RATIO - 1       # ScalarSpec default exercises S-token window-boundary scatter

# tiling
ROPE_CHUCK = 32
K_CHUNK = 512
OUT_CHUNK = 64
# kv_score_proj output-column tile = 64 (4 parallel tasks). This scope is parallelism/overlap-
# limited, NOT HBM-bandwidth-limited: each task is a small M=B*S matmul over sequential K-tiles
# and cube is only lightly busy, so the lever is running MORE tasks in parallel to overlap with
# the concurrent rope/qr_hadamard work. A wider N (fewer tasks) cuts x re-reads but under-fills
# the cores and regresses Total, so keep 4-way (N=64).
KV_SCORE_PROJ_N = 64
assert OUT_DIM % KV_SCORE_PROJ_N == 0, "OUT_DIM must be divisible by KV_SCORE_PROJ_N"
HEAD_CHUNK = 32
HEAD_DIM_CHUCK = 128
K_BLOCKS = D // K_CHUNK
OUT_BLOCKS = OUT_DIM // OUT_CHUNK
HEAD_BLOCKS = HEAD_DIM // HEAD_CHUNK
# TODO: Remove this post-processing row padding once RMSNorm/RoPE/Hadamard are
# lowered as true row-level vector ops instead of relying on boxed matmul tiles.
POST_CHUNK = 16
# Block-table state/cache writes are per-row physical scatters, so keep this
# loop row-granular for now.
BATCH_CHUNK_1 = 1
assert B % BATCH_CHUNK_1 == 0, "B must be divisible by BATCH_CHUNK_1"
assert BATCH_CHUNK_1 <= POST_CHUNK, "BATCH_CHUNK_1 must not exceed the POST_CHUNK matmul box"


@pl.jit.inline
def indexer_compressor(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Tensor[[B, S, HEAD_DIM], pl.FP32],
    compress_state: pl.Tensor[[COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[B, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    even_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    odd_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    idx_block_table: pl.Tensor[[B, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    start_pos: pl.Tensor[[B], pl.INT32],
    rotate: pl.Scalar[pl.BOOL],
):
    x_flat = pl.reshape(x, [B * S, D])
    idx_cmp_kv_proj_scratch = pl.create_tensor([B * S, OUT_DIM], dtype=pl.FP32)
    idx_cmp_score_proj_scratch = pl.create_tensor([B * S, OUT_DIM], dtype=pl.FP32)
    # Hand-off buffers that decouple the block-table-bound scatter/cache passes from the
    # pure-compute pass. pooled_all carries the softmax-pooled KV (Pass 1 -> Pass 2);
    # kv_final_all carries the rmsnorm+rope+hadamard result (Pass 2 -> Pass 3). Both are
    # contiguous per-batch (no paged indirection), so the compute pass that only reads
    # pooled_all and writes kv_final_all has no in-place read+write and no block-table
    # handle, letting its 64 batches spread across cores instead of serializing.
    pooled_all = pl.create_tensor([B, HEAD_DIM], dtype=pl.FP32)
    kv_final_all = pl.create_tensor([B, HEAD_DIM], dtype=pl.FP32)
    compress_state_flat = pl.reshape(compress_state, [COMPRESS_STATE_BLOCK_NUM * COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM])
    compress_state_block_table_flat = pl.reshape(compress_state_block_table, [B * COMPRESS_STATE_MAX_BLOCKS])
    kv_flat = pl.reshape(kv, [B * S, HEAD_DIM])
    idx_kv_cache_flat = pl.reshape(idx_kv_cache, [IDX_CACHE_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    idx_block_table_flat = pl.reshape(idx_block_table, [B * IDX_CACHE_MAX_BLOCKS])

    # pl.parallel (not pl.range): the KV_SCORE_PROJ_N output-column chunks are independent
    # (disjoint wkv/wgate columns + disjoint scratch assemble, x read-only), so they spread
    # across cube cores instead of running serially. As pl.range they ran ~serially, and this
    # is the compressor's first matmul, delaying everything downstream that reads the proj.
    for o0 in pl.parallel(0, OUT_DIM, KV_SCORE_PROJ_N):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_score_proj"):
            x_tile = x_flat[:, 0 : K_CHUNK]
            wkv_tile = wkv[0 : K_CHUNK, o0 : o0 + KV_SCORE_PROJ_N]
            wgate_tile = wgate[0 : K_CHUNK, o0 : o0 + KV_SCORE_PROJ_N]
            kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32)
            score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32)

            for k0 in pl.range(K_CHUNK, D, K_CHUNK):
                x_tile = x_flat[:, k0 : k0 + K_CHUNK]
                wkv_tile = wkv[k0 : k0 + K_CHUNK, o0 : o0 + KV_SCORE_PROJ_N]
                wgate_tile = wgate[k0 : k0 + K_CHUNK, o0 : o0 + KV_SCORE_PROJ_N]
                kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile)
                score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile)

            idx_cmp_kv_proj_scratch = pl.assemble(idx_cmp_kv_proj_scratch, kv_acc, [0, o0])
            idx_cmp_score_proj_scratch = pl.assemble(idx_cmp_score_proj_scratch, score_acc, [0, o0])

    idx_cmp_kv_proj_by_batch = pl.reshape(idx_cmp_kv_proj_scratch, [B, S * OUT_DIM])
    idx_cmp_score_proj_by_batch = pl.reshape(idx_cmp_score_proj_scratch, [B, S * OUT_DIM])

    # Zero-init pooled_all: Pass 2 is coarsened to batch groups and computes every row
    # unconditionally (the per-batch should_compress gate lives only in Pass 1/3), so the
    # non-compressing rows must read 0 -- not uninitialized GM -- to avoid NaN propagating
    # through the rmsnorm/rope/hadamard chain. Pass 3 still gates the actual cache write, so
    # the garbage-free 0 rows are computed but never stored.
    for c0 in pl.parallel(0, B, POST_CHUNK):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="pooled_init"):
            pooled_all = pl.assemble(
                pooled_all, pl.full([POST_CHUNK, HEAD_DIM], dtype=pl.FP32, value=0.0), [c0, 0]
            )

    # Pass 1 (per-batch, block-table-bound): scatter projected kv/score into paged
    # compress_state (o0 loop folded into one scope -> 1 task/batch, spreads to ~40 cores; the
    # block-table WAW is not a hard serializer, the old 4-scope-per-batch structure was), then
    # softmax-pool the completed window into contiguous pooled_all. Kept fused: both splitting
    # softmax into its own pass and folding its spmd->range were tried and regress (softmax
    # stays spmd-pinned to ~4 cores; isolating it or folding to a range drops it to 1 core).
    # The fused folded-scatter + softmax-spmd is the best measured structure.
    for c_idx in pl.parallel(0, B, BATCH_CHUNK_1):
        start_pos_b = pl.read(start_pos, [c_idx])
        pos_b = start_pos_b % COMPRESS_RATIO
        ape_row_b = pl.cast(pos_b, target_type=pl.INDEX)
        pre_tokens_b = COMPRESS_RATIO - pos_b
        boundary_end_b = start_pos_b + pre_tokens_b - 1
        cur_window_start_b = boundary_end_b - COMPRESS_RATIO + 1
        prev_window_start_b = cur_window_start_b - COMPRESS_RATIO

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_scatter_paged"):
            for o0 in pl.range(0, OUT_DIM, OUT_CHUNK):
                for s in pl.range(S):
                    abs_pos_s = start_pos_b + s
                    state_blk_off = abs_pos_s // COMPRESS_STATE_BLOCK_SIZE
                    state_intra = abs_pos_s % COMPRESS_STATE_BLOCK_SIZE
                    state_blk_id = pl.cast(
                        pl.read(compress_state_block_table_flat, [c_idx * COMPRESS_STATE_MAX_BLOCKS + state_blk_off]),
                        pl.INDEX,
                    )
                    state_row = state_blk_id * COMPRESS_STATE_BLOCK_SIZE + state_intra
                    proj_col0 = s * OUT_DIM + o0
                    token_ape_row = (ape_row_b + s) % COMPRESS_RATIO
                    kv_tile = idx_cmp_kv_proj_by_batch[c_idx : c_idx + BATCH_CHUNK_1, proj_col0 : proj_col0 + OUT_CHUNK]
                    score_tile = idx_cmp_score_proj_by_batch[c_idx : c_idx + BATCH_CHUNK_1, proj_col0 : proj_col0 + OUT_CHUNK]
                    ape_tile = ape[token_ape_row : token_ape_row + 1, o0 : o0 + OUT_CHUNK]
                    ape_base = pl.full([BATCH_CHUNK_1, OUT_CHUNK], dtype=pl.FP32, value=0.0)
                    score_tile = pl.add(score_tile, pl.col_expand(ape_base, ape_tile))
                    compress_state_flat = pl.assemble(compress_state_flat, kv_tile, [state_row, o0])
                    compress_state_flat = pl.assemble(compress_state_flat, score_tile, [state_row, OUT_DIM + o0])

        if pos_b + S >= COMPRESS_RATIO:

            for hb in pl.spmd(HEAD_BLOCKS, name_hint="softmax_pool"):
                h0 = hb * HEAD_CHUNK
                last_abs = cur_window_start_b + COMPRESS_RATIO - 1
                last_blk_off = last_abs // COMPRESS_STATE_BLOCK_SIZE
                last_intra = last_abs % COMPRESS_STATE_BLOCK_SIZE
                last_blk_id = pl.cast(
                    pl.read(compress_state_block_table_flat, [c_idx * COMPRESS_STATE_MAX_BLOCKS + last_blk_off]),
                    pl.INDEX,
                )
                last_row = last_blk_id * COMPRESS_STATE_BLOCK_SIZE + last_intra
                last_col0 = OUT_DIM + HEAD_DIM + h0
                mi = compress_state_flat[last_row : last_row + 1, last_col0 : last_col0 + HEAD_CHUNK]
                li = pl.exp(pl.sub(mi, mi))
                oi = compress_state_flat[last_row : last_row + 1, HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_CHUNK]

                for s in pl.range(0, COMPRESS_RATIO):
                    prev_abs = prev_window_start_b + s
                    if prev_abs >= 0:
                        prev_blk_off = prev_abs // COMPRESS_STATE_BLOCK_SIZE
                        prev_intra = prev_abs % COMPRESS_STATE_BLOCK_SIZE
                        prev_blk_id = pl.cast(
                            pl.read(compress_state_block_table_flat, [c_idx * COMPRESS_STATE_MAX_BLOCKS + prev_blk_off]),
                            pl.INDEX,
                        )
                        prev_row = prev_blk_id * COMPRESS_STATE_BLOCK_SIZE + prev_intra
                        front_score = compress_state_flat[prev_row : prev_row + 1, OUT_DIM + h0 : OUT_DIM + h0 + HEAD_CHUNK]
                        front_kv = compress_state_flat[prev_row : prev_row + 1, h0 : h0 + HEAD_CHUNK]
                        mi_next_front = pl.maximum(mi, front_score)
                        alpha_front = pl.exp(pl.sub(mi, mi_next_front))
                        beta_front = pl.exp(pl.sub(front_score, mi_next_front))
                        li = pl.add(pl.mul(alpha_front, li), beta_front)
                        oi = pl.add(pl.mul(oi, alpha_front), pl.mul(front_kv, beta_front))
                        mi = mi_next_front

                for s in pl.range(0, COMPRESS_RATIO - 1):
                    cur_abs = cur_window_start_b + s
                    cur_blk_off = cur_abs // COMPRESS_STATE_BLOCK_SIZE
                    cur_intra = cur_abs % COMPRESS_STATE_BLOCK_SIZE
                    cur_blk_id = pl.cast(
                        pl.read(compress_state_block_table_flat, [c_idx * COMPRESS_STATE_MAX_BLOCKS + cur_blk_off]),
                        pl.INDEX,
                    )
                    cur_row = cur_blk_id * COMPRESS_STATE_BLOCK_SIZE + cur_intra
                    back_col0 = OUT_DIM + HEAD_DIM + h0
                    back_score = compress_state_flat[cur_row : cur_row + 1, back_col0 : back_col0 + HEAD_CHUNK]
                    back_kv = compress_state_flat[cur_row : cur_row + 1, HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_CHUNK]
                    mi_next_back = pl.maximum(mi, back_score)
                    alpha_back = pl.exp(pl.sub(mi, mi_next_back))
                    beta_back = pl.exp(pl.sub(back_score, mi_next_back))
                    li = pl.add(pl.mul(alpha_back, li), beta_back)
                    oi = pl.add(pl.mul(oi, alpha_back), pl.mul(back_kv, beta_back))
                    mi = mi_next_back

                pooled_chunk = pl.div(oi, li)
                pooled_all = pl.assemble(pooled_all, pooled_chunk, [c_idx, h0])

    # Pass 2 (batch-group coarsened, NO block-table): rmsnorm + rope + hadamard. Reads
    # pooled_all read-only and writes contiguous kv_final_all -- no in-place RW, no paged
    # handle. Because there is no block-table indirection here, the per-batch loop can be
    # coarsened to groups of POST_CHUNK: the rmsnorm/rope/hadamard matmuls are already boxed to
    # POST_CHUNK rows, so packing POST_CHUNK real batches fills what used to be 15/16 padding
    # for free (1 valid row -> POST_CHUNK valid rows), cutting task count B/POST_CHUNK-fold.
    # The should_compress gate is dropped here (can't gate a mixed-start_pos group); every row
    # is computed and Pass 3 gates the store. cos/sin are now per-row (POST_CHUNK real rows).
    for c0 in pl.parallel(0, B, POST_CHUNK):
        cos_g = cos[c0 : c0 + POST_CHUNK, 0 : ROPE_HEAD_DIM // 2]
        sin_g = sin[c0 : c0 + POST_CHUNK, 0 : ROPE_HEAD_DIM // 2]
        pooled_kv = pl.create_tensor([POST_CHUNK, HEAD_DIM], dtype=pl.FP32)
        normed_kv = pl.create_tensor([POST_CHUNK, HEAD_DIM], dtype=pl.BF16)
        kv_final = pl.create_tensor([POST_CHUNK, HEAD_DIM], dtype=pl.FP32)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="pooled_load"):
            pooled_kv = pl.assemble(pooled_kv, pooled_all[c0 : c0 + POST_CHUNK, :], [0, 0])

        norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
        kv_rope = pl.create_tensor([POST_CHUNK, ROPE_HEAD_DIM], dtype=pl.BF16)
        # rmsnorm stays its own (NONE) scope: folding it into the UP_DOWN rope_fused scope
        # below fails codegen -- kv_rope built as an in-scope intermediate (assemble) and then
        # fed to the slice matmul has no resolvable tile view after the row-split
        # ("Tensor view not found for parameter kv_rope_inline"). rope_fused works because
        # kv_rope arrives as a clean scope input from here.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm_rope_slice"):
            partial_sq = pl.full([1, POST_CHUNK], dtype=pl.FP32, value=0.0)
            for k0 in pl.range(0, HEAD_DIM, HEAD_CHUNK):
                # Golden applies rmsnorm to kv.to(torch.bfloat16), then casts to FP32 inside rmsnorm.
                kv_rms_chunk = pl.cast(
                    pl.cast(pooled_kv[:, k0 : k0 + HEAD_CHUNK], target_type=pl.BF16, mode="rint"),
                    target_type=pl.FP32,
                )
                partial_sq = pl.add(
                    partial_sq,
                    pl.reshape(pl.row_sum(pl.mul(kv_rms_chunk, kv_rms_chunk)), [1, POST_CHUNK]),
                )

            variance = pl.reshape(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS), [POST_CHUNK, 1])
            inv_rms = pl.recip(pl.sqrt(variance))
            for k0 in pl.range(0, HEAD_DIM, HEAD_CHUNK):
                kv_norm_chunk = pl.cast(
                    pl.cast(pooled_kv[:, k0 : k0 + HEAD_CHUNK], target_type=pl.BF16, mode="rint"),
                    target_type=pl.FP32,
                )
                gamma = norm_w_2d[:, k0 : k0 + HEAD_CHUNK]
                normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
                normed_kv = pl.assemble(normed_kv, pl.cast(normed_chunk, target_type=pl.BF16, mode="rint"), [0, k0])
            kv_rope = pl.assemble(kv_rope, normed_kv[:, NOPE_HEAD_DIM : HEAD_DIM], [0, 0])

        # Mix: slice matmul (cube) + cos/sin rotate (vec) + assemble matmul (cube) + BF16
        # write (vec), all in one UP_DOWN scope. Row-halving keeps each AIV subblock's Vec
        # bounded; the assemble feeds rope_even/rope_odd whole (K=ROPE_HEAD_DIM//2=32, no
        # column subview) so it never trips the UP_DOWN valid_row mismatch.
        with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.UP_DOWN)], name_hint="rope_fused"):
            even_acc = pl.matmul(kv_rope, even_select, out_dtype=pl.FP32)
            odd_acc = pl.matmul(kv_rope, odd_select, out_dtype=pl.FP32)
            rope_even = pl.cast(pl.sub(pl.mul(even_acc, cos_g), pl.mul(odd_acc, sin_g)), target_type=pl.BF16, mode="rint")
            rope_odd = pl.cast(pl.add(pl.mul(even_acc, sin_g), pl.mul(odd_acc, cos_g)), target_type=pl.BF16, mode="rint")
            rope_acc = pl.matmul(rope_even, even_select, out_dtype=pl.FP32, b_trans=True)
            rope_acc = pl.matmul_acc(rope_acc, rope_odd, odd_select, b_trans=True)
            normed_kv = pl.assemble(normed_kv, pl.cast(rope_acc, target_type=pl.BF16, mode="rint"), [0, NOPE_HEAD_DIM])

        if rotate:
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_hadamard"):
                kv_proj_tile = normed_kv[:, 0 : HEAD_DIM]
                for o0 in pl.range(0, HEAD_DIM, OUT_CHUNK):
                    hadamard_tile = hadamard[0 : HEAD_DIM, o0 : o0 + OUT_CHUNK]
                    kv_hadamard_acc = pl.matmul(kv_proj_tile, hadamard_tile, out_dtype=pl.FP32)
                    kv_final = pl.assemble(kv_final, kv_hadamard_acc, [0, o0])
        else:
            for oi in pl.spmd(HEAD_DIM // OUT_CHUNK, name_hint="kv_write"):
                o0 = oi * OUT_CHUNK
                kv_out_tile = normed_kv[:, o0 : o0 + OUT_CHUNK]
                kv_final = pl.assemble(kv_final, pl.cast(kv_out_tile, target_type=pl.FP32), [0, o0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_final_store"):
            kv_final_all = pl.assemble(kv_final_all, kv_final[0 : POST_CHUNK, :], [c0, 0])

    # Pass 3 (per-batch, block-table-bound): scatter the compressed row into kv_flat and the
    # paged idx_kv_cache. Reads kv_final_all read-only; the paged writes keep this serialized
    # but the per-batch body is tiny.
    for c_idx in pl.parallel(0, B, BATCH_CHUNK_1):
        start_pos_b = pl.read(start_pos, [c_idx])
        pos_b = start_pos_b % COMPRESS_RATIO
        if pos_b + S >= COMPRESS_RATIO:
            cache_col = start_pos_b // COMPRESS_RATIO
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_and_cache_write"):
                kv_row_fp32 = kv_final_all[c_idx : c_idx + BATCH_CHUNK_1, 0 : HEAD_DIM]
                kv_flat = pl.assemble(kv_flat, kv_row_fp32, [c_idx * S, 0])
                idx_blk_off = cache_col // BLOCK_SIZE
                idx_intra = cache_col % BLOCK_SIZE
                idx_blk_id = pl.cast(
                    pl.read(idx_block_table_flat, [c_idx * IDX_CACHE_MAX_BLOCKS + idx_blk_off]),
                    pl.INDEX,
                )
                cache_row = idx_blk_id * BLOCK_SIZE + idx_intra
                idx_kv_cache_flat = pl.assemble(
                    idx_kv_cache_flat,
                    pl.cast(kv_row_fp32, target_type=pl.BF16, mode="rint"),
                    [cache_row, 0],
                )

    compress_state = pl.reshape(compress_state_flat, [COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM])
    idx_kv_cache = pl.reshape(idx_kv_cache_flat, [IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM])
    kv = pl.reshape(kv_flat, [B, S, HEAD_DIM])
    return kv, compress_state, idx_kv_cache


@pl.jit
def compressor_test(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Out[pl.Tensor[[B, S, HEAD_DIM], pl.FP32]],
    compress_state: pl.Out[pl.Tensor[[COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[B, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    even_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    odd_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.Out[pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    idx_block_table: pl.Tensor[[B, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    start_pos: pl.Tensor[[B], pl.INT32],
    rotate: pl.Scalar[pl.BOOL],
):
    kv, compress_state, idx_kv_cache = indexer_compressor(
        x,
        kv,
        compress_state,
        compress_state_block_table,
        wkv,
        wgate,
        ape,
        norm_w,
        cos,
        sin,
        even_select,
        odd_select,
        hadamard,
        idx_kv_cache,
        idx_block_table,
        start_pos,
        rotate,
    )
    return kv, compress_state, idx_kv_cache


def golden_compressor(tensors):
    """Torch reference for Compressor.forward (decode branch, ratio=4 overlap)."""
    import torch

    x = tensors["x"].float()
    compress_state = tensors["compress_state"]
    compress_state_block_table = tensors["compress_state_block_table"]
    wkv = tensors["wkv"].float()
    wgate = tensors["wgate"].float()
    ape = tensors["ape"]
    norm_w = tensors["norm_w"]
    cos = tensors["cos"]
    sin = tensors["sin"]
    hadamard = tensors["hadamard"].float()
    idx_kv_cache = tensors["idx_kv_cache"]
    idx_block_table = tensors["idx_block_table"]
    start_pos_t = tensors["start_pos"]
    rotate = bool(tensors["rotate"])
    bsz, _, _ = x.shape
    ratio, d, rd = COMPRESS_RATIO, hadamard.shape[0], tensors["even_select"].shape[0]

    kv = x @ wkv                        # [B, S, OUT_DIM]
    score = x @ wgate                   # [B, S, OUT_DIM]

    pooled = torch.zeros(bsz, 1, HEAD_DIM, dtype=torch.float32, device=x.device)
    should_compress_rows = torch.zeros(bsz, dtype=torch.bool, device=x.device)

    for b in range(bsz):
        start_pos = int(start_pos_t[b].item())
        pre_tokens = min(S, ratio - (start_pos % ratio))
        should_compress = pre_tokens < S or (start_pos + S) % ratio == 0
        boundary_end = start_pos + pre_tokens - 1
        cur_window_start = boundary_end - ratio + 1
        prev_window_start = cur_window_start - ratio

        # Per-token ape add + state scatter through the compressor-state block table.
        ape_row_g = start_pos % ratio
        for s in range(S):
            token_ape_row = (ape_row_g + s) % ratio
            score[b, s, :] = score[b, s, :] + ape[token_ape_row]
            abs_pos = start_pos + s
            blk_id = int(compress_state_block_table[b, abs_pos // COMPRESS_STATE_BLOCK_SIZE].item())
            intra = abs_pos % COMPRESS_STATE_BLOCK_SIZE
            compress_state[blk_id, intra, :OUT_DIM] = kv[b, s, :]
            compress_state[blk_id, intra, OUT_DIM:] = score[b, s, :]

        if should_compress:
            should_compress_rows[b] = True
            kv_rows = []
            score_rows = []
            for s in range(ratio):
                abs_pos = prev_window_start + s
                if abs_pos < 0:
                    kv_rows.append(torch.zeros(HEAD_DIM, dtype=torch.float32, device=x.device))
                    score_rows.append(torch.full((HEAD_DIM,), float("-inf"), dtype=torch.float32, device=x.device))
                    continue
                blk_id = int(compress_state_block_table[b, abs_pos // COMPRESS_STATE_BLOCK_SIZE].item())
                intra = abs_pos % COMPRESS_STATE_BLOCK_SIZE
                kv_rows.append(compress_state[blk_id, intra, :HEAD_DIM])
                score_rows.append(compress_state[blk_id, intra, OUT_DIM:OUT_DIM + HEAD_DIM])
            for s in range(ratio):
                abs_pos = cur_window_start + s
                blk_id = int(compress_state_block_table[b, abs_pos // COMPRESS_STATE_BLOCK_SIZE].item())
                intra = abs_pos % COMPRESS_STATE_BLOCK_SIZE
                kv_rows.append(compress_state[blk_id, intra, HEAD_DIM:OUT_DIM])
                score_rows.append(compress_state[blk_id, intra, OUT_DIM + HEAD_DIM:])
            kvs = torch.stack(kv_rows, dim=0).unsqueeze(0)
            scs = torch.stack(score_rows, dim=0).unsqueeze(0)
            pooled[b : b + 1] = (kvs * scs.softmax(dim=1)).sum(dim=1, keepdim=True)

    tensors["compress_state"][:] = compress_state

    if not bool(should_compress_rows.any()):
        return

    def rmsnorm(x, w):
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + EPS)
        return (w * x).to(torch.bfloat16)

    for b in range(bsz):
        if not bool(should_compress_rows[b]):
            continue
        start_pos = int(start_pos_t[b].item())
        kv_b = rmsnorm(pooled[b : b + 1].to(torch.bfloat16), norm_w)

        x_pair = kv_b[..., -rd:].unflatten(-1, (-1, 2))
        x0, x1 = x_pair[..., 0], x_pair[..., 1]
        cos_v, sin_v = cos[b].view(-1), sin[b].view(-1)
        y0 = (x0 * cos_v - x1 * sin_v).to(torch.bfloat16)
        y1 = (x0 * sin_v + x1 * cos_v).to(torch.bfloat16)

        kv_b = torch.cat([kv_b[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1).float()

        if rotate:
            kv_b = kv_b @ hadamard
        # Kernel writes pooled result only to kv[:, 0, :]; leave kv[:, 1:, :] = 0.
        tensors["kv"][b : b + 1, 0:1, :] = kv_b

        cache_col = start_pos // ratio
        blk_id = int(idx_block_table[b, cache_col // BLOCK_SIZE].item())
        idx_kv_cache[blk_id, cache_col % BLOCK_SIZE, 0] = kv_b[0, 0]

    tensors["idx_kv_cache"][:] = idx_kv_cache


def build_tensor_specs(start_pos: int = START_POS, hetero_start_pos: bool = False):
    import torch  # type: ignore[import]
    from golden import ScalarSpec, TensorSpec

    def init_x():
        return torch.rand(B, S, D)
    def init_compress_state():
        state = torch.zeros(COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM)
        state[:, :, OUT_DIM:] = FP32_NEG_INF
        return state
    def init_compress_state_block_table():
        tbl = torch.full((B, COMPRESS_STATE_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(COMPRESS_STATE_MAX_BLOCKS):
                tbl[b, j] = b * COMPRESS_STATE_MAX_BLOCKS + j
        return tbl
    def init_wkv():
        return torch.rand(D, OUT_DIM)
    def init_wgate():
        return torch.rand(D, OUT_DIM)
    def init_ape():
        return torch.rand(COMPRESS_RATIO, OUT_DIM)
    def init_norm_w():
        return torch.ones(HEAD_DIM)
    def init_cos():
        return torch.rand(B, ROPE_HEAD_DIM // 2)
    def init_sin():
        return torch.rand(B, ROPE_HEAD_DIM // 2)
    def init_odd_select():
        M = torch.zeros((ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2))
        for i in range(ROPE_HEAD_DIM // 2):
            M[2*i+1, i] = 1
        return M
    def init_even_select():
        M = torch.zeros((ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2))
        for i in range(ROPE_HEAD_DIM // 2):
            M[2*i, i] = 1
        return M
    def init_hadamard():
        return torch.rand(HEAD_DIM, HEAD_DIM) * (HEAD_DIM ** -0.5)
    def init_idx_kv_cache():
        return torch.zeros(IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
    def init_idx_block_table():
        tbl = torch.full((B, IDX_CACHE_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(IDX_CACHE_MAX_BLOCKS):
                tbl[b, j] = b * IDX_CACHE_MAX_BLOCKS + j
        return tbl
    def init_start_pos():
        vals = torch.full((B,), start_pos, dtype=torch.int32)
        if hetero_start_pos:
            pattern = torch.tensor([0, COMPRESS_RATIO - S, COMPRESS_RATIO - 1, COMPRESS_RATIO * 2 - 1], dtype=torch.int32)
            for b in range(B):
                vals[b] = pattern[(b // BATCH_CHUNK_1) % int(pattern.numel())]
        return vals

    return [
        TensorSpec("x", [B, S, D], torch.bfloat16, init_value=init_x),
        TensorSpec("kv", [B, S, HEAD_DIM], torch.float32, is_output=True),
        TensorSpec("compress_state", [COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], torch.float32, init_value=init_compress_state, is_output=True),
        TensorSpec("compress_state_block_table", [B, COMPRESS_STATE_MAX_BLOCKS], torch.int32, init_value=init_compress_state_block_table),
        TensorSpec("wkv", [D, OUT_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [D, OUT_DIM], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.float32, init_value=init_norm_w),
        TensorSpec("cos", [B, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_cos),
        TensorSpec("sin", [B, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_sin),
        TensorSpec("even_select", [ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], torch.bfloat16, init_value=init_even_select),
        TensorSpec("odd_select", [ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], torch.bfloat16, init_value=init_odd_select),
        TensorSpec("hadamard", [HEAD_DIM, HEAD_DIM], torch.bfloat16, init_value=init_hadamard),
        TensorSpec("idx_kv_cache", [IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_idx_kv_cache, is_output=True),
        TensorSpec("idx_block_table", [B, IDX_CACHE_MAX_BLOCKS], torch.int32, init_value=init_idx_block_table),
        TensorSpec("start_pos", [B], torch.int32, init_value=init_start_pos),
        ScalarSpec("rotate", torch.bool, ROTATE),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--start-pos", type=int, default=START_POS,
                        help="Decode start position for no-compression/aligned/crossing coverage.")
    parser.add_argument("--hetero-start-pos", action="store_true", default=False,
                        help="Use a per-batch start_pos pattern for no-compress/exact/crossing coverage.")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=compressor_test,
        specs=build_tensor_specs(args.start_pos, args.hetero_start_pos),
        golden_fn=golden_compressor,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "kv":          ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
            "compress_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "idx_kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.005 / (IDX_CACHE_BLOCK_NUM * BLOCK_SIZE)),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
