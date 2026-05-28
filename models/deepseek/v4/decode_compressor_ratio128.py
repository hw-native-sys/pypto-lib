# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 KV Compressor (decode incremental, ratio=128 non-overlap).

Uses non-overlapping state layout with 128 slots.
Softmax+pool over all slots. No state shift needed."""

import pypto.language as pl

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ

# model config
B = DECODE_BATCH
S = DECODE_SEQ
EPS = M.rms_norm_eps
D = M.hidden_size
HEAD_DIM = M.head_dim
HEAD_DIM_INV = 1.0 / HEAD_DIM
ROPE_HEAD_DIM = M.qk_rope_head_dim
NOPE_HEAD_DIM = M.nope_head_dim
MAX_SEQ_LEN = M.max_position_embeddings

# kernel-local (ratio-128 non-overlap compressor)
COMPRESS_RATIO = 128
IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
COFF = 1
OUT_DIM = COFF * HEAD_DIM          # 512
STATE_LEN = COFF * COMPRESS_RATIO  # 128
START_POS = COMPRESS_RATIO - 1       # default exercises S-token window-boundary scatter

# tiling
ROPE_TILE = 32
K_TILE = 512
OUT_TILE = 64

HEAD_TILE = 64 if B * S >= 64 else 128
B_TILE = 64
# TODO: Remove this post-processing row padding once RMSNorm/RoPE are lowered
# as true row-level vector ops instead of relying on boxed tile matmul shapes.
POST_TILE = 16

# Paged cache layout (mirrors recipes sfa_kv_state / sfa_cmp_kv contract).
# kv_state_pool merges kv and score into a single FP32 pool with [..., 0:OUT_DIM]=kv,
# [..., OUT_DIM:2*OUT_DIM]=score. cmp_kv_pool is the long-term compressed BF16 pool.
# Both are addressed via per-request block_table indirection.
BLOCK_SIZE = 128
STATE_BLOCKS_PER_BATCH = (STATE_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
CMP_BLOCKS_PER_BATCH = (IDX_KV_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
STATE_BLOCK_NUM = B * STATE_BLOCKS_PER_BATCH
CMP_BLOCK_NUM = B * CMP_BLOCKS_PER_BATCH
STATE_CHANNELS = 2 * OUT_DIM


@pl.jit.inline
def compressor(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Tensor[[B, S, HEAD_DIM], pl.FP32],
    kv_state_pool: pl.Tensor[[STATE_BLOCK_NUM, BLOCK_SIZE, STATE_CHANNELS], pl.FP32],
    state_block_table: pl.Tensor[[B, STATE_BLOCKS_PER_BATCH], pl.INT32],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    even_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    odd_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    cmp_kv_pool: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_BLOCKS_PER_BATCH], pl.INT32],
    start_pos: pl.Tensor[[B], pl.INT32],
):
    x_flat = pl.reshape(x, [B * S, D])
    # State pool flat view: one paged block = one row of length BLOCK_SIZE * STATE_CHANNELS.
    # Slot s within a block occupies cols [s*STATE_CHANNELS, (s+1)*STATE_CHANNELS); inside that
    # span, [0:OUT_DIM] is kv channel, [OUT_DIM:2*OUT_DIM] is score channel.
    kv_state_pool_flat = pl.reshape(kv_state_pool, [STATE_BLOCK_NUM, BLOCK_SIZE * STATE_CHANNELS])
    kv_flat = pl.reshape(kv, [B * S, HEAD_DIM])
    # Cmp pool flat view: physical rows are linearized across all blocks.
    cmp_kv_pool_flat = pl.reshape(cmp_kv_pool, [CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])

    cmp128_kv_proj_scratch = pl.create_tensor([B * S, OUT_DIM], dtype=pl.FP32)
    cmp128_score_proj_scratch = pl.create_tensor([B * S, OUT_DIM], dtype=pl.FP32)

    for global_row0 in pl.parallel(0, B * S, B_TILE):
        for o0 in pl.parallel(0, OUT_DIM, OUT_TILE):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_score_proj"):
                kv_acc = pl.create_tensor([B_TILE, OUT_TILE], dtype=pl.FP32)
                score_acc = pl.create_tensor([B_TILE, OUT_TILE], dtype=pl.FP32)
                for kb in pl.pipeline(0, D // K_TILE, stage=2):
                    k0 = kb * K_TILE
                    x_tile = x_flat[global_row0 : global_row0 + B_TILE, k0 : k0 + K_TILE]
                    wkv_tile = wkv[k0 : k0 + K_TILE, o0 : o0 + OUT_TILE]
                    wgate_tile = wgate[k0 : k0 + K_TILE, o0 : o0 + OUT_TILE]
                    if k0 == 0:
                        kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32)
                        score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32)
                    else:
                        kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile)
                        score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile)

                cmp128_kv_proj_scratch[global_row0 : global_row0 + B_TILE, o0 : o0 + OUT_TILE] = kv_acc
                cmp128_score_proj_scratch[global_row0 : global_row0 + B_TILE, o0 : o0 + OUT_TILE] = score_acc

    cmp128_kv_proj_by_batch = pl.reshape(cmp128_kv_proj_scratch, [B, S * OUT_DIM])
    cmp128_score_proj_by_batch = pl.reshape(cmp128_score_proj_scratch, [B, S * OUT_DIM])

    for global_c_idx in pl.parallel(B):
        start_pos_b = pl.read(start_pos, [global_c_idx])
        pos_b = start_pos_b % COMPRESS_RATIO
        ape_row_b = pl.cast(pos_b, target_type=pl.INDEX)
        pre_tokens_b = COMPRESS_RATIO - pos_b
        # Paged indirection: STATE_LEN == BLOCK_SIZE so each batch owns exactly one state block;
        # cache_col < BLOCK_SIZE in supported configs so logical cmp-block index is 0.
        state_blk_id = pl.cast(pl.read(state_block_table, [global_c_idx, 0]), target_type=pl.INDEX)
        cmp_blk_id = pl.cast(pl.read(cmp_block_table, [global_c_idx, 0]), target_type=pl.INDEX)
        cos_b = cos[global_c_idx : global_c_idx + 1, 0 : ROPE_HEAD_DIM // 2]
        sin_b = sin[global_c_idx : global_c_idx + 1, 0 : ROPE_HEAD_DIM // 2]
        pooled_kv = pl.create_tensor([POST_TILE, HEAD_DIM], dtype=pl.FP32)
        normed_kv = pl.create_tensor([POST_TILE, HEAD_DIM], dtype=pl.BF16)
        kv_final = pl.create_tensor([POST_TILE, HEAD_DIM], dtype=pl.FP32)

        if pos_b + S < COMPRESS_RATIO:
            for o0 in pl.parallel(0, OUT_DIM, OUT_TILE):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_scatter_no_compress"):
                    for s in pl.range(S):
                        proj_col0 = s * OUT_DIM + o0
                        token_ape_row = (ape_row_b + s) % COMPRESS_RATIO
                        ape_tile = ape[token_ape_row : token_ape_row + 1, o0 : o0 + OUT_TILE]
                        slot_col0_s = token_ape_row * STATE_CHANNELS
                        kv_tile = cmp128_kv_proj_by_batch[global_c_idx : global_c_idx + 1, proj_col0 : proj_col0 + OUT_TILE]
                        score_tile = cmp128_score_proj_by_batch[global_c_idx : global_c_idx + 1, proj_col0 : proj_col0 + OUT_TILE]
                        ape_base = pl.full([1, OUT_TILE], dtype=pl.FP32, value=0.0)
                        score_tile = pl.add(score_tile, pl.col_expand(ape_base, ape_tile))
                        kv_state_pool_flat = pl.assemble(kv_state_pool_flat, kv_tile, [state_blk_id, slot_col0_s + o0])
                        kv_state_pool_flat = pl.assemble(kv_state_pool_flat, score_tile, [state_blk_id, slot_col0_s + OUT_DIM + o0])
        else:
            if pos_b + S == COMPRESS_RATIO:
                for o0 in pl.parallel(0, OUT_DIM, OUT_TILE):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_scatter_exact_boundary"):
                        for s in pl.range(S):
                            proj_col0 = s * OUT_DIM + o0
                            token_ape_row = (ape_row_b + s) % COMPRESS_RATIO
                            ape_tile = ape[token_ape_row : token_ape_row + 1, o0 : o0 + OUT_TILE]
                            slot_col0_s = token_ape_row * STATE_CHANNELS
                            kv_tile = cmp128_kv_proj_by_batch[global_c_idx : global_c_idx + 1, proj_col0 : proj_col0 + OUT_TILE]
                            score_tile = cmp128_score_proj_by_batch[global_c_idx : global_c_idx + 1, proj_col0 : proj_col0 + OUT_TILE]
                            ape_base = pl.full([1, OUT_TILE], dtype=pl.FP32, value=0.0)
                            score_tile = pl.add(score_tile, pl.col_expand(ape_base, ape_tile))
                            kv_state_pool_flat = pl.assemble(kv_state_pool_flat, kv_tile, [state_blk_id, slot_col0_s + o0])
                            kv_state_pool_flat = pl.assemble(kv_state_pool_flat, score_tile, [state_blk_id, slot_col0_s + OUT_DIM + o0])
            else:
                for o0 in pl.parallel(0, OUT_DIM, OUT_TILE):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_scatter_crossing_pre"):
                        for s in pl.range(pre_tokens_b):
                            proj_col0 = s * OUT_DIM + o0
                            token_ape_row = (ape_row_b + s) % COMPRESS_RATIO
                            ape_tile = ape[token_ape_row : token_ape_row + 1, o0 : o0 + OUT_TILE]
                            slot_col0_s = token_ape_row * STATE_CHANNELS
                            kv_tile = cmp128_kv_proj_by_batch[global_c_idx : global_c_idx + 1, proj_col0 : proj_col0 + OUT_TILE]
                            score_tile = cmp128_score_proj_by_batch[global_c_idx : global_c_idx + 1, proj_col0 : proj_col0 + OUT_TILE]
                            ape_base = pl.full([1, OUT_TILE], dtype=pl.FP32, value=0.0)
                            score_tile = pl.add(score_tile, pl.col_expand(ape_base, ape_tile))
                            kv_state_pool_flat = pl.assemble(kv_state_pool_flat, kv_tile, [state_blk_id, slot_col0_s + o0])
                            kv_state_pool_flat = pl.assemble(kv_state_pool_flat, score_tile, [state_blk_id, slot_col0_s + OUT_DIM + o0])

            for h0 in pl.parallel(0, HEAD_DIM, HEAD_TILE):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="softmax_pool"):
                    softmax_score_state = pl.create_tensor([STATE_LEN, HEAD_TILE], dtype=pl.FP32)
                    softmax_kv_state = pl.create_tensor([STATE_LEN, HEAD_TILE], dtype=pl.FP32)
                    for s in pl.range(STATE_LEN):
                        kv_col0 = s * STATE_CHANNELS + h0
                        score_col0 = s * STATE_CHANNELS + OUT_DIM + h0
                        slot_score = kv_state_pool_flat[state_blk_id : state_blk_id + 1, score_col0 : score_col0 + HEAD_TILE]
                        slot_kv = kv_state_pool_flat[state_blk_id : state_blk_id + 1, kv_col0 : kv_col0 + HEAD_TILE]
                        softmax_score_state = pl.assemble(softmax_score_state, slot_score, [s, 0])
                        softmax_kv_state = pl.assemble(softmax_kv_state, slot_kv, [s, 0])

                    softmax_score_state_t = pl.transpose(softmax_score_state, axis1=0, axis2=1)
                    softmax_kv_state_t = pl.transpose(softmax_kv_state, axis1=0, axis2=1)
                    score_max = pl.row_max(softmax_score_state_t)
                    score_exp = pl.exp(pl.row_expand_sub(softmax_score_state_t, score_max))
                    score_sum = pl.row_sum(score_exp)
                    score_prob = pl.row_expand_div(score_exp, score_sum)
                    pooled_chunk_t = pl.row_sum(pl.mul(softmax_kv_state_t, score_prob))
                    pooled_chunk = pl.reshape(pooled_chunk_t, [1, HEAD_TILE])
                    pooled_kv = pl.assemble(pooled_kv, pooled_chunk, [0, h0])

        # No state shift for non-overlap

        # RMSNorm with BF16 intermediate
        norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
        kv_rope = pl.create_tensor([POST_TILE, ROPE_HEAD_DIM], dtype=pl.BF16)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm_rope_slice"):
            partial_sq = pl.full([1, POST_TILE], dtype=pl.FP32, value=0.0)
            for rms_kb in pl.pipeline(HEAD_DIM // HEAD_TILE, stage=4):
                kv_rms_chunk = pl.cast(
                    pl.cast(pooled_kv[:, rms_kb * HEAD_TILE : (rms_kb + 1) * HEAD_TILE], target_type=pl.BF16, mode="rint"),
                    target_type=pl.FP32,
                )
                partial_sq = pl.add(
                    partial_sq,
                    pl.reshape(pl.row_sum(pl.mul(kv_rms_chunk, kv_rms_chunk)), [1, POST_TILE]),
                )

            variance = pl.reshape(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS), [POST_TILE, 1])
            inv_rms = pl.recip(pl.sqrt(variance))
            for rms_kb in pl.pipeline(HEAD_DIM // HEAD_TILE, stage=4):
                kv_norm_chunk = pl.cast(
                    pl.cast(pooled_kv[:, rms_kb * HEAD_TILE : (rms_kb + 1) * HEAD_TILE], target_type=pl.BF16, mode="rint"),
                    target_type=pl.FP32,
                )
                gamma = norm_w_2d[:, rms_kb * HEAD_TILE : (rms_kb + 1) * HEAD_TILE]
                normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
                normed_kv = pl.assemble(normed_kv, pl.cast(normed_chunk, target_type=pl.BF16, mode="rint"), [0, rms_kb * HEAD_TILE])
            kv_rope = pl.assemble(kv_rope, normed_kv[:, NOPE_HEAD_DIM : HEAD_DIM], [0, 0])

        # Selector-based RoPE
        kv_proj_even = pl.create_tensor([POST_TILE, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
        kv_proj_odd = pl.create_tensor([POST_TILE, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
        rope_even = pl.create_tensor([POST_TILE, ROPE_HEAD_DIM // 2], dtype=pl.BF16)
        rope_odd = pl.create_tensor([POST_TILE, ROPE_HEAD_DIM // 2], dtype=pl.BF16)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_slice"):
            even_acc = pl.matmul(kv_rope, even_select, out_dtype=pl.FP32)
            odd_acc = pl.matmul(kv_rope, odd_select, out_dtype=pl.FP32)
            kv_proj_even = pl.assemble(kv_proj_even, even_acc, [0, 0])
            kv_proj_odd = pl.assemble(kv_proj_odd, odd_acc, [0, 0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_apply"):
            even_tile = kv_proj_even[:, :]
            odd_tile = kv_proj_odd[:, :]
            rope_even_acc = pl.cast(pl.sub(pl.col_expand_mul(even_tile, cos_b), pl.col_expand_mul(odd_tile, sin_b)), target_type=pl.BF16, mode="rint")
            rope_odd_acc = pl.cast(pl.add(pl.col_expand_mul(even_tile, sin_b), pl.col_expand_mul(odd_tile, cos_b)), target_type=pl.BF16, mode="rint")
            rope_even = pl.assemble(rope_even, rope_even_acc, [0, 0])
            rope_odd = pl.assemble(rope_odd, rope_odd_acc, [0, 0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_assemble"):
            rope_acc = pl.matmul(rope_even, even_select, out_dtype=pl.FP32, b_trans=True)
            rope_acc = pl.matmul_acc(rope_acc, rope_odd, odd_select, b_trans=True)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_write"):
            normed_kv = pl.assemble(normed_kv, pl.cast(rope_acc, target_type=pl.BF16, mode="rint"), [0, NOPE_HEAD_DIM])

        cache_col = start_pos_b // COMPRESS_RATIO
        for o0 in pl.parallel(0, HEAD_DIM, OUT_TILE):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_write"):
                kv_out_tile = normed_kv[:, o0 : o0 + OUT_TILE]
                kv_out_fp32 = pl.cast(kv_out_tile, target_type=pl.FP32)
                kv_final = pl.assemble(kv_final, kv_out_fp32, [0, o0])

        if pos_b + S >= COMPRESS_RATIO:
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_and_cache_write"):
                kv_row_fp32 = kv_final[0 : 1, 0 : HEAD_DIM]
                kv_flat = pl.assemble(kv_flat, kv_row_fp32, [global_c_idx * S, 0])
                phys_cmp_row = cmp_blk_id * BLOCK_SIZE + cache_col
                cmp_kv_pool_flat = pl.assemble(
                    cmp_kv_pool_flat,
                    pl.cast(kv_row_fp32, target_type=pl.BF16, mode="rint"),
                    [phys_cmp_row, 0],
                )

        if pos_b + S > COMPRESS_RATIO:
            for o0 in pl.parallel(0, OUT_DIM, OUT_TILE):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_scatter_next"):
                    for s in pl.range(pre_tokens_b, S):
                        proj_col0 = s * OUT_DIM + o0
                        token_ape_row = (ape_row_b + s) % COMPRESS_RATIO
                        ape_tile = ape[token_ape_row : token_ape_row + 1, o0 : o0 + OUT_TILE]
                        slot_col0_s = token_ape_row * STATE_CHANNELS
                        kv_tile = cmp128_kv_proj_by_batch[global_c_idx : global_c_idx + 1, proj_col0 : proj_col0 + OUT_TILE]
                        score_tile = cmp128_score_proj_by_batch[global_c_idx : global_c_idx + 1, proj_col0 : proj_col0 + OUT_TILE]
                        dep_tile = kv_final[0 : 1, o0 : o0 + OUT_TILE]
                        dep_zero = pl.sub(dep_tile, dep_tile)
                        kv_tile = pl.add(kv_tile, dep_zero)
                        score_tile = pl.add(score_tile, dep_zero)
                        ape_base = pl.full([1, OUT_TILE], dtype=pl.FP32, value=0.0)
                        score_tile = pl.add(score_tile, pl.col_expand(ape_base, ape_tile))
                        kv_state_pool_flat = pl.assemble(kv_state_pool_flat, kv_tile, [state_blk_id, slot_col0_s + o0])
                        kv_state_pool_flat = pl.assemble(kv_state_pool_flat, score_tile, [state_blk_id, slot_col0_s + OUT_DIM + o0])

    kv_state_pool = pl.reshape(kv_state_pool_flat, [STATE_BLOCK_NUM, BLOCK_SIZE, STATE_CHANNELS])
    kv = pl.reshape(kv_flat, [B, S, HEAD_DIM])
    cmp_kv_pool = pl.reshape(cmp_kv_pool_flat, [CMP_BLOCK_NUM, BLOCK_SIZE, HEAD_DIM])
    return kv, kv_state_pool, cmp_kv_pool


@pl.jit
def compressor_test(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Out[pl.Tensor[[B, S, HEAD_DIM], pl.FP32]],
    kv_state_pool: pl.Out[pl.Tensor[[STATE_BLOCK_NUM, BLOCK_SIZE, STATE_CHANNELS], pl.FP32]],
    state_block_table: pl.Tensor[[B, STATE_BLOCKS_PER_BATCH], pl.INT32],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    even_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    odd_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    cmp_kv_pool: pl.Out[pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[B, CMP_BLOCKS_PER_BATCH], pl.INT32],
    start_pos: pl.Tensor[[B], pl.INT32],
):
    kv, kv_state_pool, cmp_kv_pool = compressor(
        x, kv, kv_state_pool, state_block_table, wkv, wgate, ape, norm_w, cos, sin,
        even_select, odd_select, cmp_kv_pool, cmp_block_table, start_pos,
    )
    return kv, kv_state_pool, cmp_kv_pool


def golden_compressor(tensors):
    """Torch reference for Compressor.forward (decode branch, ratio=128 non-overlap).

    Operates on paged caches: kv_state_pool (kv + score channels merged) and cmp_kv_pool,
    each addressed via the corresponding block_table.
    """
    import torch

    x = tensors["x"].float()
    state_block_table = tensors["state_block_table"]
    cmp_block_table = tensors["cmp_block_table"]
    # Unpack paged state into per-batch [STATE_LEN, OUT_DIM] views.
    kv_state_pool = tensors["kv_state_pool"]
    kv_state = torch.zeros(B, STATE_LEN, OUT_DIM, dtype=torch.float32, device=x.device)
    score_state = torch.zeros(B, STATE_LEN, OUT_DIM, dtype=torch.float32, device=x.device)
    for b in range(B):
        sblk = int(state_block_table[b, 0].item())
        kv_state[b] = kv_state_pool[sblk, :STATE_LEN, :OUT_DIM]
        score_state[b] = kv_state_pool[sblk, :STATE_LEN, OUT_DIM:2 * OUT_DIM]

    wkv = tensors["wkv"].float()
    wgate = tensors["wgate"].float()
    ape = tensors["ape"]
    norm_w = tensors["norm_w"]
    cos = tensors["cos"]
    sin = tensors["sin"]
    cmp_kv_pool = tensors["cmp_kv_pool"]
    start_pos_t = tensors["start_pos"]
    bsz, _, _ = x.shape
    ratio, rd = COMPRESS_RATIO, ROPE_HEAD_DIM

    kv = x @ wkv                        # [B, S, OUT_DIM]
    score = x @ wgate                   # [B, S, OUT_DIM]
    kv_proj = kv
    pooled = torch.zeros(bsz, 1, HEAD_DIM, dtype=torch.float32, device=x.device)
    should_compress_rows = torch.zeros(bsz, dtype=torch.bool, device=x.device)

    for b in range(bsz):
        start_pos = int(start_pos_t[b].item())
        pre_tokens = min(S, ratio - (start_pos % ratio))
        should_compress = pre_tokens < S or (start_pos + S) % ratio == 0
        ape_row_g = start_pos % ratio

        for s in range(pre_tokens):
            token_ape_row = (ape_row_g + s) % ratio
            score[b, s, :] = score[b, s, :] + ape[token_ape_row]
            kv_state[b, token_ape_row] = kv[b, s, :]
            score_state[b, token_ape_row] = score[b, s, :]

        if should_compress:
            should_compress_rows[b] = True
            pooled[b : b + 1] = (kv_state[b : b + 1] * score_state[b : b + 1].softmax(dim=1)).sum(dim=1, keepdim=True)

        if pre_tokens < S:
            for s in range(pre_tokens, S):
                token_ape_row = (ape_row_g + s) % ratio
                score[b, s, :] = score[b, s, :] + ape[token_ape_row]
                kv_state[b, token_ape_row] = kv_proj[b, s, :]
                score_state[b, token_ape_row] = score[b, s, :]

    # Repack updated per-batch state back into the paged pool.
    for b in range(B):
        sblk = int(state_block_table[b, 0].item())
        kv_state_pool[sblk, :STATE_LEN, :OUT_DIM] = kv_state[b]
        kv_state_pool[sblk, :STATE_LEN, OUT_DIM:2 * OUT_DIM] = score_state[b]
    tensors["kv_state_pool"][:] = kv_state_pool

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

        # Kernel writes pooled result only to kv[:, 0, :]; leave kv[:, 1:, :] = 0.
        tensors["kv"][b : b + 1, 0:1, :] = kv_b

        cache_col = start_pos // ratio
        logical_blk = cache_col // BLOCK_SIZE
        intra_offset = cache_col % BLOCK_SIZE
        cblk = int(cmp_block_table[b, logical_blk].item())
        cmp_kv_pool[cblk, intra_offset] = kv_b[0, 0]

    tensors["cmp_kv_pool"][:] = cmp_kv_pool


def build_tensor_specs(start_pos: int = START_POS, hetero_start_pos: bool = False):
    import torch  # type: ignore[import]
    from golden import TensorSpec

    def init_x():
        return torch.rand(B, S, D)
    def init_kv_state_pool():
        return torch.zeros(STATE_BLOCK_NUM, BLOCK_SIZE, STATE_CHANNELS)
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
    def init_cmp_kv_pool():
        return torch.zeros(CMP_BLOCK_NUM, BLOCK_SIZE, HEAD_DIM)
    def init_state_block_table():
        # Identity logical->physical mapping: batch b uses block b.
        return torch.arange(B * STATE_BLOCKS_PER_BATCH, dtype=torch.int32).view(B, STATE_BLOCKS_PER_BATCH)
    def init_cmp_block_table():
        return torch.arange(B * CMP_BLOCKS_PER_BATCH, dtype=torch.int32).view(B, CMP_BLOCKS_PER_BATCH)
    def init_start_pos():
        vals = torch.full((B,), start_pos, dtype=torch.int32)
        if hetero_start_pos:
            pattern = torch.tensor([0, COMPRESS_RATIO - S, COMPRESS_RATIO - 1, COMPRESS_RATIO * 2 - 1], dtype=torch.int32)
            for b in range(B):
                vals[b] = pattern[b % int(pattern.numel())]
        return vals
    return [
        TensorSpec("x", [B, S, D], torch.bfloat16, init_value=init_x),
        TensorSpec("kv", [B, S, HEAD_DIM], torch.float32, is_output=True),
        TensorSpec("kv_state_pool", [STATE_BLOCK_NUM, BLOCK_SIZE, STATE_CHANNELS], torch.float32, init_value=init_kv_state_pool, is_output=True),
        TensorSpec("state_block_table", [B, STATE_BLOCKS_PER_BATCH], torch.int32, init_value=init_state_block_table),
        TensorSpec("wkv", [D, OUT_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [D, OUT_DIM], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.float32, init_value=init_norm_w),
        TensorSpec("cos", [B, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_cos),
        TensorSpec("sin", [B, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_sin),
        TensorSpec("even_select", [ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], torch.bfloat16, init_value=init_even_select),
        TensorSpec("odd_select", [ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], torch.bfloat16, init_value=init_odd_select),
        TensorSpec("cmp_kv_pool", [CMP_BLOCK_NUM, BLOCK_SIZE, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv_pool, is_output=True),
        TensorSpec("cmp_block_table", [B, CMP_BLOCKS_PER_BATCH], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("start_pos", [B], torch.int32, init_value=init_start_pos),
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
                        help="Use a grouped per-batch start_pos pattern.")
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
            "kv":            ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
            "kv_state_pool": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "cmp_kv_pool":   ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.005 / BLOCK_SIZE),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
