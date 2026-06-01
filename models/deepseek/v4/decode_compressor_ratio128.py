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

from config import FLASH as M, BLOCK_SIZE, C128_COMPRESSOR_BLOCK_SIZE, DECODE_BATCH, DECODE_SEQ

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
OUT_DIM = COFF * HEAD_DIM
STATE_LEN = COFF * COMPRESS_RATIO
# Paged contract:
# - compress_state_block_table is indexed by absolute token position blocks,
#   matching vLLM's DeepSeek V4 compressor state cache contract:
#     state_logical_block = absolute_position // COMPRESS_STATE_BLOCK_SIZE
#     state_intra        = absolute_position % COMPRESS_STATE_BLOCK_SIZE
# - APE remains ratio-local:
#     ape_row = absolute_position % COMPRESS_RATIO
# - COMPRESS_STATE_MAX_BLOCKS=64 is only enough for the current tests
#   (64 * 8 positions per request), not full MAX_SEQ_LEN coverage.
COMPRESS_STATE_MAX_BLOCKS = 64
COMPRESS_STATE_BLOCK_NUM = B * COMPRESS_STATE_MAX_BLOCKS
COMPRESS_STATE_BLOCK_SIZE = C128_COMPRESSOR_BLOCK_SIZE
COMPRESS_STATE_DIM = 2 * OUT_DIM
CMP_MAX_BLOCKS = 64
CMP_BLOCK_NUM = B * CMP_MAX_BLOCKS
START_POS = COMPRESS_RATIO - 1
if IDX_KV_LEN > CMP_MAX_BLOCKS * BLOCK_SIZE:
    raise ValueError("ratio128 compressed KV cache capacity is smaller than max compressed sequence length")

# tiling
ROPE_TILE = 32
K_TILE = 512
OUT_TILE = 64
HEAD_TILE = 64
B_TILE = 64
RMS_TILE = 8


@pl.jit.inline
def compressor(
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
    cmp_kv_cache: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    start_pos: pl.Tensor[[B], pl.INT32],
):
    x_flat = pl.reshape(x, [B * S, D])
    # Flatten paged block dim into row. Slot s in a block occupies cols
    # [s*COMPRESS_STATE_DIM, (s+1)*COMPRESS_STATE_DIM); kv channel is [:OUT_DIM],
    # score channel is [OUT_DIM:].
    compress_state_flat = pl.reshape(compress_state, [COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE * COMPRESS_STATE_DIM])
    kv_flat = pl.reshape(kv, [B * S, HEAD_DIM])
    cmp_kv_cache_flat = pl.reshape(cmp_kv_cache, [CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])

    kv_proj_scratch = pl.create_tensor([B * S, OUT_DIM], dtype=pl.FP32)
    score_proj_scratch = pl.create_tensor([B * S, OUT_DIM], dtype=pl.FP32)

    for idx in pl.spmd(B * S * OUT_DIM // (B_TILE * OUT_TILE), name_hint="kv_score_proj"):
        global_row0 = (idx // (OUT_DIM // OUT_TILE)) * B_TILE
        o0 = (idx % (OUT_DIM // OUT_TILE)) * OUT_TILE
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

        kv_proj_scratch[global_row0 : global_row0 + B_TILE, o0 : o0 + OUT_TILE] = kv_acc
        score_proj_scratch[global_row0 : global_row0 + B_TILE, o0 : o0 + OUT_TILE] = score_acc

    kv_proj_by_batch = pl.reshape(kv_proj_scratch, [B, S * OUT_DIM])
    score_proj_by_batch = pl.reshape(score_proj_scratch, [B, S * OUT_DIM])
    pooled_kv = pl.create_tensor([B, HEAD_DIM], dtype=pl.FP32)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_scatter_pre"):
        for global_c_idx in pl.range(B):
            start_pos_b = pl.read(start_pos, [global_c_idx])
            pos_b = start_pos_b % COMPRESS_RATIO
            ape_row_b = pl.cast(pos_b, target_type=pl.INDEX)
            pre_tokens_b = COMPRESS_RATIO - pos_b
            if pos_b + S > COMPRESS_RATIO:
                scatter_n = pre_tokens_b
            else:
                scatter_n = S
            for s in pl.pipeline(scatter_n, stage=2):
                proj_col0 = s * OUT_DIM
                token_ape_row = (ape_row_b + s) % COMPRESS_RATIO
                # State cache uses absolute-position block-table addressing;
                # only APE indexing wraps modulo COMPRESS_RATIO.
                state_pos = start_pos_b + s
                state_logical_blk = state_pos // COMPRESS_STATE_BLOCK_SIZE
                state_intra = state_pos % COMPRESS_STATE_BLOCK_SIZE
                state_blk_id = pl.cast(pl.read(compress_state_block_table, [global_c_idx, state_logical_blk]), target_type=pl.INDEX)
                slot_col0_s = state_intra * COMPRESS_STATE_DIM
                ape_row = ape[token_ape_row : token_ape_row + 1, 0 : OUT_DIM]
                kv_row = kv_proj_by_batch[global_c_idx : global_c_idx + 1, proj_col0 : proj_col0 + OUT_DIM]
                score_row = score_proj_by_batch[global_c_idx : global_c_idx + 1, proj_col0 : proj_col0 + OUT_DIM]
                score_row = pl.add(score_row, ape_row)
                compress_state_flat[state_blk_id : state_blk_id + 1, slot_col0_s : slot_col0_s + OUT_DIM] = kv_row
                compress_state_flat[state_blk_id : state_blk_id + 1, slot_col0_s + OUT_DIM : slot_col0_s + 2 * OUT_DIM] = score_row

    # Unconditional — gating pooled_kv inside pl.spmd hits pypto #1569;
    # non-compress batches' pooled value is discarded by downstream gates.
    for idx in pl.spmd(B * HEAD_DIM // HEAD_TILE, name_hint="softmax_pool"):
        global_c_idx = idx // (HEAD_DIM // HEAD_TILE)
        h0 = (idx % (HEAD_DIM // HEAD_TILE)) * HEAD_TILE

        softmax_score_state = pl.create_tensor([STATE_LEN, HEAD_TILE], dtype=pl.FP32)
        softmax_kv_state = pl.create_tensor([STATE_LEN, HEAD_TILE], dtype=pl.FP32)
        for s in pl.pipeline(STATE_LEN, stage=2):
            start_pos_b = pl.read(start_pos, [global_c_idx])
            pos_b = start_pos_b % COMPRESS_RATIO
            # Read the absolute state window ending at the next compression
            # boundary. This mirrors vLLM's `start = position - ratio + 1`.
            compress_pos = start_pos_b + (COMPRESS_RATIO - 1 - pos_b)
            state_pos = compress_pos - (COMPRESS_RATIO - 1) + s
            state_logical_blk = state_pos // COMPRESS_STATE_BLOCK_SIZE
            state_intra = state_pos % COMPRESS_STATE_BLOCK_SIZE
            state_blk_id = pl.cast(pl.read(compress_state_block_table, [global_c_idx, state_logical_blk]), target_type=pl.INDEX)
            kv_col0 = state_intra * COMPRESS_STATE_DIM + h0
            score_col0 = state_intra * COMPRESS_STATE_DIM + OUT_DIM + h0
            slot_score = compress_state_flat[state_blk_id : state_blk_id + 1, score_col0 : score_col0 + HEAD_TILE]
            slot_kv = compress_state_flat[state_blk_id : state_blk_id + 1, kv_col0 : kv_col0 + HEAD_TILE]
            softmax_score_state[s : s + 1, :] = slot_score
            softmax_kv_state[s : s + 1, :] = slot_kv

        softmax_score_state_t = pl.transpose(softmax_score_state, axis1=0, axis2=1)
        softmax_kv_state_t = pl.transpose(softmax_kv_state, axis1=0, axis2=1)
        score_max = pl.row_max(softmax_score_state_t)
        score_exp = pl.exp(pl.row_expand_sub(softmax_score_state_t, score_max))
        score_sum = pl.row_sum(score_exp)
        score_prob = pl.row_expand_div(score_exp, score_sum)
        pooled_chunk_t = pl.row_sum(pl.mul(softmax_kv_state_t, score_prob))
        pooled_chunk = pl.reshape(pooled_chunk_t, [1, HEAD_TILE])
        pooled_kv[global_c_idx : global_c_idx + 1, h0 : h0 + HEAD_TILE] = pooled_chunk

    # No state shift for non-overlap

    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    normed_kv = pl.create_tensor([B, HEAD_DIM], dtype=pl.FP32)

    # Fused RMSNorm + gather/scatter-based RoPE, processing RMS_TILE real
    # batches per spmd block so all vec col-vectors hit ptoas's 32B-aligned
    # row stride without per-batch row padding.
    for batch_base_idx in pl.spmd(B // RMS_TILE, name_hint="rmsnorm_rope"):
        batch_base = batch_base_idx * RMS_TILE
        cos_b = cos[batch_base : batch_base + RMS_TILE, 0 : ROPE_HEAD_DIM // 2]
        sin_b = sin[batch_base : batch_base + RMS_TILE, 0 : ROPE_HEAD_DIM // 2]
        partial_sq = pl.full([1, RMS_TILE], dtype=pl.FP32, value=0.0)
        for rms_kb in pl.pipeline(HEAD_DIM // HEAD_TILE, stage=2):
            kv_rms_chunk = pooled_kv[batch_base : batch_base + RMS_TILE, rms_kb * HEAD_TILE : (rms_kb + 1) * HEAD_TILE]
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            kv_rms_rowsum = pl.reshape(pl.row_sum(kv_rms_sq), [1, RMS_TILE])
            partial_sq = pl.add(partial_sq, kv_rms_rowsum)

        variance = pl.reshape(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS), [RMS_TILE, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        # Normalize the NOPE span; the rope span [NOPE_HEAD_DIM:HEAD_DIM] is
        # written once below by the RoPE'd rope_buf.
        for rms_kb in pl.pipeline(NOPE_HEAD_DIM // HEAD_TILE, stage=2):
            kv_norm_chunk = pooled_kv[batch_base : batch_base + RMS_TILE, rms_kb * HEAD_TILE : (rms_kb + 1) * HEAD_TILE]
            gamma = norm_w_2d[:, rms_kb * HEAD_TILE : (rms_kb + 1) * HEAD_TILE]
            normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
            normed_kv[batch_base : batch_base + RMS_TILE, rms_kb * HEAD_TILE : (rms_kb + 1) * HEAD_TILE] = normed_chunk

        # Re-derive the RoPE input slice (the HEAD_TILE chunk the loop above
        # skipped) from pooled_kv with the same inv_rms/gamma.
        kv_rope_norm = pooled_kv[batch_base : batch_base + RMS_TILE, NOPE_HEAD_DIM : HEAD_DIM]
        gamma_rope = norm_w_2d[:, NOPE_HEAD_DIM : HEAD_DIM]
        rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
        even_tile = pl.gather(rope_normed, mask_pattern=pl.tile.MaskPattern.P0101)
        odd_tile = pl.gather(rope_normed, mask_pattern=pl.tile.MaskPattern.P1010)
        rope_even = pl.sub(pl.mul(even_tile, cos_b), pl.mul(odd_tile, sin_b))
        rope_odd = pl.add(pl.mul(even_tile, sin_b), pl.mul(odd_tile, cos_b))
        rope_buf = pl.full([RMS_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        # Mask-form scatter: re-interleave rotated even/odd halves into the
        # P0101/P1010-selected columns (inverse of the gathers above). Drops the
        # explicit even/odd index tiles the index-form scatter required. A3/sim only.
        rope_buf = pl.tensor.scatter(rope_even, mask_pattern=pl.tile.MaskPattern.P0101, dst=rope_buf)
        rope_buf = pl.tensor.scatter(rope_odd, mask_pattern=pl.tile.MaskPattern.P1010, dst=rope_buf)
        normed_kv[batch_base : batch_base + RMS_TILE, NOPE_HEAD_DIM : HEAD_DIM] = rope_buf

    # Keep this scope separate from rmsnorm_rope: merging would put 3 outer
    # carries (kv_flat / cmp_kv_cache_flat / compress_state_flat) on one task,
    # which hits pypto #1573 (orchestration phi cross-assignment).
    for batch_base_idx in pl.spmd(B // RMS_TILE, name_hint="kv_finalize"):
        batch_base = batch_base_idx * RMS_TILE
        for inner in pl.range(RMS_TILE):
            global_c_idx = batch_base + inner
            start_pos_b = pl.read(start_pos, [global_c_idx])
            pos_b = start_pos_b % COMPRESS_RATIO
            ape_row_b = pl.cast(pos_b, target_type=pl.INDEX)
            pre_tokens_b = COMPRESS_RATIO - pos_b
            cache_col = start_pos_b // COMPRESS_RATIO
            if pos_b + S >= COMPRESS_RATIO:
                kv_row = normed_kv[global_c_idx : global_c_idx + 1, 0 : HEAD_DIM]
                kv_flat[global_c_idx * S : global_c_idx * S + 1, :] = kv_row
                cmp_logical_blk = cache_col // BLOCK_SIZE
                cmp_intra = cache_col % BLOCK_SIZE
                cmp_blk_id = pl.cast(pl.read(cmp_block_table, [global_c_idx, cmp_logical_blk]), target_type=pl.INDEX)
                phys_cmp_row = cmp_blk_id * BLOCK_SIZE + cmp_intra
                cmp_kv_cache_flat[phys_cmp_row : phys_cmp_row + 1, :] = pl.cast(kv_row, target_type=pl.BF16, mode="rint")

                for s in pl.pipeline(pre_tokens_b, S, stage=2):
                    proj_col0 = s * OUT_DIM
                    token_ape_row = (ape_row_b + s) % COMPRESS_RATIO
                    # Tokens after the boundary start the next absolute state
                    # window; they are not written to a modulo-128 ring slot.
                    state_pos = start_pos_b + s
                    state_logical_blk = state_pos // COMPRESS_STATE_BLOCK_SIZE
                    state_intra = state_pos % COMPRESS_STATE_BLOCK_SIZE
                    state_blk_id = pl.cast(pl.read(compress_state_block_table, [global_c_idx, state_logical_blk]), target_type=pl.INDEX)
                    slot_col0_s = state_intra * COMPRESS_STATE_DIM
                    ape_row = ape[token_ape_row : token_ape_row + 1, 0 : OUT_DIM]
                    kv_row = kv_proj_by_batch[global_c_idx : global_c_idx + 1, proj_col0 : proj_col0 + OUT_DIM]
                    score_row = score_proj_by_batch[global_c_idx : global_c_idx + 1, proj_col0 : proj_col0 + OUT_DIM]
                    score_row = pl.add(score_row, ape_row)
                    compress_state_flat[state_blk_id : state_blk_id + 1, slot_col0_s : slot_col0_s + OUT_DIM] = kv_row
                    compress_state_flat[state_blk_id : state_blk_id + 1, slot_col0_s + OUT_DIM : slot_col0_s + 2 * OUT_DIM] = score_row

    compress_state = pl.reshape(compress_state_flat, [COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM])
    kv = pl.reshape(kv_flat, [B, S, HEAD_DIM])
    cmp_kv_cache = pl.reshape(cmp_kv_cache_flat, [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM])
    return kv, compress_state, cmp_kv_cache


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
    cmp_kv_cache: pl.Out[pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    start_pos: pl.Tensor[[B], pl.INT32],
):
    kv, compress_state, cmp_kv_cache = compressor(
        x, kv, compress_state, compress_state_block_table, wkv, wgate, ape, norm_w, cos, sin,
        cmp_kv_cache, cmp_block_table, start_pos,
    )
    return kv, compress_state, cmp_kv_cache


def golden_compressor(tensors):
    """Torch reference for Compressor.forward (decode branch, ratio=128 non-overlap).

    Operates on paged caches: compress_state (kv + score channels merged) and cmp_kv_cache,
    each addressed via the corresponding block_table.
    """
    import torch

    x = tensors["x"].float()
    compress_state_block_table = tensors["compress_state_block_table"]
    cmp_block_table = tensors["cmp_block_table"]
    # Read/write state cache through absolute-position block-table addressing,
    # matching vLLM's DeepSeek V4 compressor contract. APE remains modulo ratio.
    compress_state = tensors["compress_state"]

    def read_state_row(b, pos):
        logical_blk = pos // COMPRESS_STATE_BLOCK_SIZE
        intra = pos % COMPRESS_STATE_BLOCK_SIZE
        sblk = int(compress_state_block_table[b, logical_blk].item())
        return (
            compress_state[sblk, intra, :OUT_DIM],
            compress_state[sblk, intra, OUT_DIM:2 * OUT_DIM],
        )

    def write_state_row(b, pos, kv_row, score_row):
        logical_blk = pos // COMPRESS_STATE_BLOCK_SIZE
        intra = pos % COMPRESS_STATE_BLOCK_SIZE
        sblk = int(compress_state_block_table[b, logical_blk].item())
        compress_state[sblk, intra, :OUT_DIM] = kv_row
        compress_state[sblk, intra, OUT_DIM:2 * OUT_DIM] = score_row

    wkv = tensors["wkv"].float()
    wgate = tensors["wgate"].float()
    ape = tensors["ape"]
    norm_w = tensors["norm_w"]
    cos = tensors["cos"]
    sin = tensors["sin"]
    cmp_kv_cache = tensors["cmp_kv_cache"]
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
            write_state_row(b, start_pos + s, kv[b, s, :], score[b, s, :])

        if should_compress:
            should_compress_rows[b] = True
            compress_pos = start_pos + (ratio - 1 - (start_pos % ratio))
            kv_rows = []
            score_rows = []
            for pos in range(compress_pos - ratio + 1, compress_pos + 1):
                kv_row, score_row = read_state_row(b, pos)
                kv_rows.append(kv_row)
                score_rows.append(score_row)
            kv_state = torch.stack(kv_rows, dim=0).unsqueeze(0)
            score_state = torch.stack(score_rows, dim=0).unsqueeze(0)
            pooled[b : b + 1] = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)

        if pre_tokens < S:
            for s in range(pre_tokens, S):
                token_ape_row = (ape_row_g + s) % ratio
                score[b, s, :] = score[b, s, :] + ape[token_ape_row]
                write_state_row(b, start_pos + s, kv_proj[b, s, :], score[b, s, :])
    tensors["compress_state"][:] = compress_state

    if not bool(should_compress_rows.any()):
        return

    def rmsnorm(x, w):
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + EPS)
        return w * x

    for b in range(bsz):
        if not bool(should_compress_rows[b]):
            continue
        start_pos = int(start_pos_t[b].item())
        kv_b = rmsnorm(pooled[b : b + 1], norm_w)

        x_pair = kv_b[..., -rd:].unflatten(-1, (-1, 2))
        x0, x1 = x_pair[..., 0], x_pair[..., 1]
        cos_v, sin_v = cos[b].view(-1), sin[b].view(-1)
        y0 = x0 * cos_v - x1 * sin_v
        y1 = x0 * sin_v + x1 * cos_v

        kv_b = torch.cat([kv_b[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1)

        # Kernel writes pooled result only to kv[:, 0, :]; leave kv[:, 1:, :] = 0.
        tensors["kv"][b : b + 1, 0:1, :] = kv_b

        cache_col = start_pos // ratio
        logical_blk = cache_col // BLOCK_SIZE
        intra_offset = cache_col % BLOCK_SIZE
        cblk = int(cmp_block_table[b, logical_blk].item())
        cmp_kv_cache[cblk, intra_offset, 0] = kv_b[0, 0]

    tensors["cmp_kv_cache"][:] = cmp_kv_cache


def build_tensor_specs(
    start_pos: int = START_POS,
    hetero_start_pos: bool = False,
    non_contiguous_state_blocks: bool = False,
    non_contiguous_cmp_blocks: bool = False,
):
    import torch  # type: ignore[import]
    from golden import TensorSpec

    def init_x():
        return torch.rand(B, S, D)
    def init_compress_state():
        return torch.zeros(COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM)
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
    def init_cmp_kv_cache():
        return torch.zeros(CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
    def init_compress_state_block_table():
        tbl = torch.arange(B * COMPRESS_STATE_MAX_BLOCKS, dtype=torch.int32).view(B, COMPRESS_STATE_MAX_BLOCKS)
        if non_contiguous_state_blocks:
            for b in range(B):
                tbl[b, 0], tbl[b, 31] = tbl[b, 31].clone(), tbl[b, 0].clone()
        return tbl
    def init_cmp_block_table():
        tbl = torch.arange(B * CMP_MAX_BLOCKS, dtype=torch.int32).view(B, CMP_MAX_BLOCKS)
        if non_contiguous_cmp_blocks:
            for b in range(B):
                tbl[b, 0], tbl[b, 1] = tbl[b, 1].clone(), tbl[b, 0].clone()
        return tbl
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
        TensorSpec("compress_state", [COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], torch.float32, init_value=init_compress_state, is_output=True),
        TensorSpec("compress_state_block_table", [B, COMPRESS_STATE_MAX_BLOCKS], torch.int32, init_value=init_compress_state_block_table),
        TensorSpec("wkv", [D, OUT_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [D, OUT_DIM], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.float32, init_value=init_norm_w),
        TensorSpec("cos", [B, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_cos),
        TensorSpec("sin", [B, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_sin),
        TensorSpec("cmp_kv_cache", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv_cache, is_output=True),
        TensorSpec("cmp_block_table", [B, CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
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
    parser.add_argument("--hetero-start-pos", action=argparse.BooleanOptionalAction, default=True,
                        help="Use a grouped per-batch start_pos pattern.")
    parser.add_argument("--non-contiguous-state-blocks", action=argparse.BooleanOptionalAction, default=False,
                        help="Swap state logical blocks in the fixture to validate state block-table addressing.")
    parser.add_argument("--non-contiguous-cmp-blocks", action=argparse.BooleanOptionalAction, default=False,
                        help="Swap logical compressed blocks in the fixture to validate block-table addressing.")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=compressor_test,
        specs=build_tensor_specs(
            args.start_pos,
            args.hetero_start_pos,
            args.non_contiguous_state_blocks,
            args.non_contiguous_cmp_blocks,
        ),
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
            "compress_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "cmp_kv_cache":   ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.005 / (CMP_BLOCK_NUM * BLOCK_SIZE)),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
