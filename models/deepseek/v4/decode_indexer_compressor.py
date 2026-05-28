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

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ


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
START_POS = COMPRESS_RATIO - 1       # ScalarSpec default exercises S-token window-boundary scatter

# tiling
ROPE_CHUCK = 32
K_CHUNK = 512
OUT_CHUNK = 64
HEAD_CHUNK = 32
HEAD_DIM_CHUCK = 128
K_BLOCKS = D // K_CHUNK
OUT_BLOCKS = OUT_DIM // OUT_CHUNK
HEAD_BLOCKS = HEAD_DIM // HEAD_CHUNK
# TODO: Remove this post-processing row padding once RMSNorm/RoPE/Hadamard are
# lowered as true row-level vector ops instead of relying on boxed matmul tiles.
POST_CHUNK = 16
# Batches processed per parallel iteration of the per-batch compressor loop. The
# post-processing matmuls (rmsnorm/rope/hadamard) are boxed to POST_CHUNK rows
# regardless, so at BATCH_CHUNK_1=1 they burn 15/16 of their work on padding; grouping
# fills that padding with real batches and collapses the swimlane's fragmented 1-4us
# task storm into a handful of ~50us tasks. Cap at POST_CHUNK (the matmul box). All
# grouped batches must share start_pos -- the scalar control flow (branch / ape_row /
# cache_col) is read from the group's first batch; the hetero test groups start_pos by
# BATCH_CHUNK_1 to honor this. cos/sin and the kv/score state are per-row, so those
# scale freely.
BATCH_CHUNK_1 = POST_CHUNK
assert B % BATCH_CHUNK_1 == 0, "B must be divisible by BATCH_CHUNK_1"
assert BATCH_CHUNK_1 <= POST_CHUNK, "BATCH_CHUNK_1 must not exceed the POST_CHUNK matmul box"


@pl.jit.inline
def indexer_compressor(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Tensor[[B, S, HEAD_DIM], pl.FP32],
    kv_state: pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32],
    score_state: pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    even_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    odd_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    kv_cache: pl.Tensor[[B, IDX_KV_LEN, HEAD_DIM], pl.BF16],
    start_pos: pl.Tensor[[B], pl.INT32],
    rotate: pl.Scalar[pl.BOOL],
):
    x_flat = pl.reshape(x, [B * S, D])
    idx_cmp_kv_proj_scratch = pl.create_tensor([B * S, OUT_DIM], dtype=pl.FP32)
    idx_cmp_score_proj_scratch = pl.create_tensor([B * S, OUT_DIM], dtype=pl.FP32)
    kv_state_flat = pl.reshape(kv_state, [B, STATE_LEN * OUT_DIM])
    score_state_flat = pl.reshape(score_state, [B, STATE_LEN * OUT_DIM])
    kv_flat = pl.reshape(kv, [B * S, HEAD_DIM])
    kv_cache_flat = pl.reshape(kv_cache, [B * IDX_KV_LEN, HEAD_DIM])

    for o0 in pl.range(0, OUT_DIM, OUT_CHUNK):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_score_proj"):
            x_tile = x_flat[:, 0 : K_CHUNK]
            wkv_tile = wkv[0 : K_CHUNK, o0 : o0 + OUT_CHUNK]
            wgate_tile = wgate[0 : K_CHUNK, o0 : o0 + OUT_CHUNK]
            kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32)
            score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32)

            for k0 in pl.range(K_CHUNK, D, K_CHUNK):
                x_tile = x_flat[:, k0 : k0 + K_CHUNK]
                wkv_tile = wkv[k0 : k0 + K_CHUNK, o0 : o0 + OUT_CHUNK]
                wgate_tile = wgate[k0 : k0 + K_CHUNK, o0 : o0 + OUT_CHUNK]
                kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile)
                score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile)

            idx_cmp_kv_proj_scratch = pl.assemble(idx_cmp_kv_proj_scratch, kv_acc, [0, o0])
            idx_cmp_score_proj_scratch = pl.assemble(idx_cmp_score_proj_scratch, score_acc, [0, o0])

    idx_cmp_kv_proj_by_batch = pl.reshape(idx_cmp_kv_proj_scratch, [B, S * OUT_DIM])
    idx_cmp_score_proj_by_batch = pl.reshape(idx_cmp_score_proj_scratch, [B, S * OUT_DIM])

    for c_idx in pl.parallel(0, B, BATCH_CHUNK_1):
        start_pos_b = pl.read(start_pos, [c_idx])
        pos_b = start_pos_b % COMPRESS_RATIO
        ape_row_b = pl.cast(pos_b, target_type=pl.INDEX)
        pre_tokens_b = COMPRESS_RATIO - pos_b
        cos_b = cos[c_idx : c_idx + BATCH_CHUNK_1, 0 : ROPE_HEAD_DIM // 2]
        sin_b = sin[c_idx : c_idx + BATCH_CHUNK_1, 0 : ROPE_HEAD_DIM // 2]
        pooled_kv = pl.create_tensor([POST_CHUNK, HEAD_DIM], dtype=pl.FP32)
        normed_kv = pl.create_tensor([POST_CHUNK, HEAD_DIM], dtype=pl.BF16)
        kv_final = pl.create_tensor([POST_CHUNK, HEAD_DIM], dtype=pl.FP32)

        if pos_b + S < COMPRESS_RATIO:
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_scatter_no_compress"):
                for o0 in pl.range(0, OUT_DIM, OUT_CHUNK):
                    for s in pl.range(S):
                        proj_col0 = s * OUT_DIM + o0
                        token_ape_row = (ape_row_b + s) % COMPRESS_RATIO
                        kv_tile = idx_cmp_kv_proj_by_batch[c_idx : c_idx + BATCH_CHUNK_1, proj_col0 : proj_col0 + OUT_CHUNK]
                        score_tile = idx_cmp_score_proj_by_batch[c_idx : c_idx + BATCH_CHUNK_1, proj_col0 : proj_col0 + OUT_CHUNK]
                        ape_tile = ape[token_ape_row : token_ape_row + 1, o0 : o0 + OUT_CHUNK]
                        ape_base = pl.full([BATCH_CHUNK_1, OUT_CHUNK], dtype=pl.FP32, value=0.0)
                        score_tile = pl.add(score_tile, pl.col_expand(ape_base, ape_tile))
                        slot_col0_s = (COMPRESS_RATIO + token_ape_row) * OUT_DIM
                        kv_state_flat = pl.assemble(kv_state_flat, kv_tile, [c_idx, slot_col0_s + o0])
                        score_state_flat = pl.assemble(score_state_flat, score_tile, [c_idx, slot_col0_s + o0])
        else:
            if pos_b + S == COMPRESS_RATIO:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_scatter_exact_boundary"):
                    for o0 in pl.range(0, OUT_DIM, OUT_CHUNK):
                        for s in pl.range(S):
                            proj_col0 = s * OUT_DIM + o0
                            token_ape_row = (ape_row_b + s) % COMPRESS_RATIO
                            kv_tile = idx_cmp_kv_proj_by_batch[c_idx : c_idx + BATCH_CHUNK_1, proj_col0 : proj_col0 + OUT_CHUNK]
                            score_tile = idx_cmp_score_proj_by_batch[c_idx : c_idx + BATCH_CHUNK_1, proj_col0 : proj_col0 + OUT_CHUNK]
                            ape_tile = ape[token_ape_row : token_ape_row + 1, o0 : o0 + OUT_CHUNK]
                            ape_base = pl.full([BATCH_CHUNK_1, OUT_CHUNK], dtype=pl.FP32, value=0.0)
                            score_tile = pl.add(score_tile, pl.col_expand(ape_base, ape_tile))
                            slot_col0_s = (COMPRESS_RATIO + token_ape_row) * OUT_DIM
                            kv_state_flat = pl.assemble(kv_state_flat, kv_tile, [c_idx, slot_col0_s + o0])
                            score_state_flat = pl.assemble(score_state_flat, score_tile, [c_idx, slot_col0_s + o0])
            else:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_scatter_crossing_pre"):
                    for o0 in pl.range(0, OUT_DIM, OUT_CHUNK):
                        for s in pl.range(pre_tokens_b):
                            proj_col0 = s * OUT_DIM + o0
                            token_ape_row = (ape_row_b + s) % COMPRESS_RATIO
                            kv_tile = idx_cmp_kv_proj_by_batch[c_idx : c_idx + BATCH_CHUNK_1, proj_col0 : proj_col0 + OUT_CHUNK]
                            score_tile = idx_cmp_score_proj_by_batch[c_idx : c_idx + BATCH_CHUNK_1, proj_col0 : proj_col0 + OUT_CHUNK]
                            ape_tile = ape[token_ape_row : token_ape_row + 1, o0 : o0 + OUT_CHUNK]
                            ape_base = pl.full([BATCH_CHUNK_1, OUT_CHUNK], dtype=pl.FP32, value=0.0)
                            score_tile = pl.add(score_tile, pl.col_expand(ape_base, ape_tile))
                            slot_col0_s = (COMPRESS_RATIO + token_ape_row) * OUT_DIM
                            kv_state_flat = pl.assemble(kv_state_flat, kv_tile, [c_idx, slot_col0_s + o0])
                            score_state_flat = pl.assemble(score_state_flat, score_tile, [c_idx, slot_col0_s + o0])

            for hb in pl.spmd(HEAD_BLOCKS, name_hint="softmax_pool"):
                h0 = hb * HEAD_CHUNK
                last_col0 = (STATE_LEN - 1) * OUT_DIM + HEAD_DIM + h0
                mi = score_state_flat[c_idx : c_idx + BATCH_CHUNK_1, last_col0 : last_col0 + HEAD_CHUNK]
                li = pl.exp(pl.sub(mi, mi))
                oi = kv_state_flat[c_idx : c_idx + BATCH_CHUNK_1, last_col0 : last_col0 + HEAD_CHUNK]

                for s in pl.range(0, COMPRESS_RATIO):
                    front_col0 = s * OUT_DIM + h0
                    front_score = score_state_flat[c_idx : c_idx + BATCH_CHUNK_1, front_col0 : front_col0 + HEAD_CHUNK]
                    front_kv = kv_state_flat[c_idx : c_idx + BATCH_CHUNK_1, front_col0 : front_col0 + HEAD_CHUNK]
                    mi_next_front = pl.maximum(mi, front_score)
                    alpha_front = pl.exp(pl.sub(mi, mi_next_front))
                    beta_front = pl.exp(pl.sub(front_score, mi_next_front))
                    li = pl.add(pl.mul(alpha_front, li), beta_front)
                    oi = pl.add(pl.mul(oi, alpha_front), pl.mul(front_kv, beta_front))
                    mi = mi_next_front

                for s in pl.range(COMPRESS_RATIO, STATE_LEN - 1):
                    back_col0 = s * OUT_DIM + HEAD_DIM + h0
                    back_score = score_state_flat[c_idx : c_idx + BATCH_CHUNK_1, back_col0 : back_col0 + HEAD_CHUNK]
                    back_kv = kv_state_flat[c_idx : c_idx + BATCH_CHUNK_1, back_col0 : back_col0 + HEAD_CHUNK]
                    mi_next_back = pl.maximum(mi, back_score)
                    alpha_back = pl.exp(pl.sub(mi, mi_next_back))
                    beta_back = pl.exp(pl.sub(back_score, mi_next_back))
                    li = pl.add(pl.mul(alpha_back, li), beta_back)
                    oi = pl.add(pl.mul(oi, alpha_back), pl.mul(back_kv, beta_back))
                    mi = mi_next_back

                pooled_chunk = pl.div(oi, li)
                pooled_kv = pl.assemble(pooled_kv, pooled_chunk, [0, h0])

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_shift"):
                for s in pl.range(0, COMPRESS_RATIO, 1):
                    src_col0 = (COMPRESS_RATIO + s) * OUT_DIM
                    dst_col0 = s * OUT_DIM
                    for o0 in pl.range(0, OUT_DIM, OUT_CHUNK):
                        dep_col0 = o0 % HEAD_DIM
                        dep_tile = pooled_kv[0 : BATCH_CHUNK_1, dep_col0 : dep_col0 + OUT_CHUNK]
                        dep_zero = pl.sub(dep_tile, dep_tile)
                        kv_tile = kv_state_flat[c_idx : c_idx + BATCH_CHUNK_1, src_col0 + o0 : src_col0 + o0 + OUT_CHUNK]
                        score_tile = score_state_flat[c_idx : c_idx + BATCH_CHUNK_1, src_col0 + o0 : src_col0 + o0 + OUT_CHUNK]
                        kv_tile = pl.add(kv_tile, dep_zero)
                        score_tile = pl.add(score_tile, dep_zero)
                        kv_state_flat = pl.assemble(kv_state_flat, kv_tile, [c_idx, dst_col0 + o0])
                        score_state_flat = pl.assemble(score_state_flat, score_tile, [c_idx, dst_col0 + o0])

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
                rope_even = pl.cast(pl.sub(pl.mul(even_acc, cos_b), pl.mul(odd_acc, sin_b)), target_type=pl.BF16, mode="rint")
                rope_odd = pl.cast(pl.add(pl.mul(even_acc, sin_b), pl.mul(odd_acc, cos_b)), target_type=pl.BF16, mode="rint")
                rope_acc = pl.matmul(rope_even, even_select, out_dtype=pl.FP32, b_trans=True)
                rope_acc = pl.matmul_acc(rope_acc, rope_odd, odd_select, b_trans=True)
                normed_kv = pl.assemble(normed_kv, pl.cast(rope_acc, target_type=pl.BF16, mode="rint"), [0, NOPE_HEAD_DIM])

            if rotate:
                # TODO: Match pypto2.0 by moving Hadamard into a separate
                # batch-level post pass over compressed rows instead of this
                # per-row matmul that depends on POST_CHUNK padding.
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

            cache_col = start_pos_b // COMPRESS_RATIO
            # kv_flat (row stride S) and kv_cache_flat (row stride IDX_KV_LEN) place one
            # pooled row per batch at batch-strided locations, so the group's rows can't be
            # written as one contiguous tile -- scatter each batch row individually. One
            # CORE_GROUP task still covers the whole group; cache_col is shared (homogeneous
            # start_pos within the group).
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_and_cache_write"):
                for r in pl.range(0, BATCH_CHUNK_1):
                    kv_row_fp32 = kv_final[r : r + 1, 0 : HEAD_DIM]
                    kv_flat = pl.assemble(kv_flat, kv_row_fp32, [(c_idx + r) * S, 0])
                    cache_row = (c_idx + r) * IDX_KV_LEN + cache_col
                    kv_cache_flat = pl.assemble(
                        kv_cache_flat,
                        pl.cast(kv_row_fp32, target_type=pl.BF16, mode="rint"),
                        [cache_row, 0],
                    )

            if pos_b + S > COMPRESS_RATIO:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_scatter_crossing_next"):
                    for o0 in pl.range(0, OUT_DIM, OUT_CHUNK):
                        for s in pl.range(pre_tokens_b, S):
                            proj_col0 = s * OUT_DIM + o0
                            shift_dep_col0 = (COMPRESS_RATIO - 1) * OUT_DIM + o0
                            dep_zero = pl.sub(
                                kv_state_flat[c_idx : c_idx + BATCH_CHUNK_1, shift_dep_col0 : shift_dep_col0 + OUT_CHUNK],
                                kv_state_flat[c_idx : c_idx + BATCH_CHUNK_1, shift_dep_col0 : shift_dep_col0 + OUT_CHUNK],
                            )
                            token_ape_row = (ape_row_b + s) % COMPRESS_RATIO
                            kv_tile = idx_cmp_kv_proj_by_batch[c_idx : c_idx + BATCH_CHUNK_1, proj_col0 : proj_col0 + OUT_CHUNK]
                            score_tile = idx_cmp_score_proj_by_batch[c_idx : c_idx + BATCH_CHUNK_1, proj_col0 : proj_col0 + OUT_CHUNK]
                            kv_tile = pl.add(kv_tile, dep_zero)
                            score_tile = pl.add(score_tile, dep_zero)
                            ape_tile = ape[token_ape_row : token_ape_row + 1, o0 : o0 + OUT_CHUNK]
                            ape_base = pl.full([BATCH_CHUNK_1, OUT_CHUNK], dtype=pl.FP32, value=0.0)
                            score_tile = pl.add(score_tile, pl.col_expand(ape_base, ape_tile))
                            slot_col0_s = (COMPRESS_RATIO + token_ape_row) * OUT_DIM
                            kv_state_flat = pl.assemble(kv_state_flat, kv_tile, [c_idx, slot_col0_s + o0])
                            score_state_flat = pl.assemble(score_state_flat, score_tile, [c_idx, slot_col0_s + o0])

    kv_cache = pl.reshape(kv_cache_flat, [B, IDX_KV_LEN, HEAD_DIM])

    kv_state = pl.reshape(kv_state_flat, [B, STATE_LEN, OUT_DIM])
    score_state = pl.reshape(score_state_flat, [B, STATE_LEN, OUT_DIM])
    kv = pl.reshape(kv_flat, [B, S, HEAD_DIM])
    return kv, kv_state, score_state, kv_cache


@pl.jit
def compressor_test(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Out[pl.Tensor[[B, S, HEAD_DIM], pl.FP32]],
    kv_state: pl.Out[pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32]],
    score_state: pl.Out[pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32]],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    even_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    odd_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    kv_cache: pl.Out[pl.Tensor[[B, IDX_KV_LEN, HEAD_DIM], pl.BF16]],
    start_pos: pl.Tensor[[B], pl.INT32],
    rotate: pl.Scalar[pl.BOOL],
):
    kv, kv_state, score_state, kv_cache = indexer_compressor(
        x, kv, kv_state, score_state, wkv, wgate, ape, norm_w, cos, sin, even_select, odd_select, hadamard, kv_cache, start_pos, rotate
    )
    return kv, kv_state, score_state, kv_cache


def golden_compressor(tensors):
    """Torch reference for Compressor.forward (decode branch, ratio=4 overlap)."""
    import torch

    x = tensors["x"].float()
    kv_state = tensors["kv_state"]
    score_state = tensors["score_state"]
    wkv = tensors["wkv"].float()
    wgate = tensors["wgate"].float()
    ape = tensors["ape"]
    norm_w = tensors["norm_w"]
    cos = tensors["cos"]
    sin = tensors["sin"]
    hadamard = tensors["hadamard"].float()
    kv_cache = tensors["kv_cache"]
    start_pos_t = tensors["start_pos"]
    rotate = bool(tensors["rotate"])
    bsz, _, _ = x.shape
    ratio, d, rd = COMPRESS_RATIO, hadamard.shape[0], tensors["even_select"].shape[0]

    kv = x @ wkv                        # [B, S, OUT_DIM]
    score = x @ wgate                   # [B, S, OUT_DIM]
    kv_proj = kv

    pooled = torch.zeros(bsz, 1, HEAD_DIM, dtype=torch.float32, device=x.device)
    should_compress_rows = torch.zeros(bsz, dtype=torch.bool, device=x.device)

    for b in range(bsz):
        start_pos = int(start_pos_t[b].item())
        pre_tokens = min(S, ratio - (start_pos % ratio))
        should_compress = pre_tokens < S or (start_pos + S) % ratio == 0

        # Per-token ape add + state scatter; slots wrap when the S-token step crosses a window boundary.
        ape_row_g = start_pos % ratio
        for s in range(pre_tokens):
            token_ape_row = (ape_row_g + s) % ratio
            score[b, s, :] = score[b, s, :] + ape[token_ape_row]
            kv_state[b, ratio + token_ape_row] = kv[b, s, :]
            score_state[b, ratio + token_ape_row] = score[b, s, :]

        if should_compress:
            should_compress_rows[b] = True
            kvs = torch.cat([kv_state[b : b + 1, :ratio, :d], kv_state[b : b + 1, ratio:, d:]], dim=1)
            scs = torch.cat([score_state[b : b + 1, :ratio, :d], score_state[b : b + 1, ratio:, d:]], dim=1)
            pooled[b : b + 1] = (kvs * scs.softmax(dim=1)).sum(dim=1, keepdim=True)
            kv_state[b : b + 1, :ratio] = kv_state[b : b + 1, ratio:]
            score_state[b : b + 1, :ratio] = score_state[b : b + 1, ratio:]

        if pre_tokens < S:
            for s in range(pre_tokens, S):
                token_ape_row = (ape_row_g + s) % ratio
                score[b, s, :] = score[b, s, :] + ape[token_ape_row]
                kv_state[b, ratio + token_ape_row] = kv_proj[b, s, :]
                score_state[b, ratio + token_ape_row] = score[b, s, :]

    tensors["kv_state"][:] = kv_state
    tensors["score_state"][:] = score_state

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

        kv_cache[b, start_pos // ratio] = kv_b[0, 0]

    tensors["kv_cache"][:] = kv_cache


def build_tensor_specs(start_pos: int = START_POS, hetero_start_pos: bool = False):
    import torch  # type: ignore[import]
    from golden import ScalarSpec, TensorSpec

    def init_x():
        return torch.rand(B, S, D)
    def init_kv_state():
        return torch.zeros(B, STATE_LEN, OUT_DIM)
    def init_score_state():
        return torch.zeros(B, STATE_LEN, OUT_DIM)
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
    def init_kv_cache():
        return torch.zeros(B, IDX_KV_LEN, HEAD_DIM)
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
        TensorSpec("kv_state", [B, STATE_LEN, OUT_DIM], torch.float32, init_value=init_kv_state, is_output=True),
        TensorSpec("score_state", [B, STATE_LEN, OUT_DIM], torch.float32, init_value=init_score_state, is_output=True),
        TensorSpec("wkv", [D, OUT_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [D, OUT_DIM], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.float32, init_value=init_norm_w),
        TensorSpec("cos", [B, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_cos),
        TensorSpec("sin", [B, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_sin),
        TensorSpec("even_select", [ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], torch.bfloat16, init_value=init_even_select),
        TensorSpec("odd_select", [ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], torch.bfloat16, init_value=init_odd_select),
        TensorSpec("hadamard", [HEAD_DIM, HEAD_DIM], torch.bfloat16, init_value=init_hadamard),
        TensorSpec("kv_cache", [B, IDX_KV_LEN, HEAD_DIM], torch.bfloat16, init_value=init_kv_cache, is_output=True),
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
            "kv_state":    ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "score_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "kv_cache":    ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.005 / IDX_KV_LEN),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
