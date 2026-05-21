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
Online softmax+pool over all slots. No state shift needed."""

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
START_POS = COMPRESS_RATIO - 1       # ScalarSpec default exercises S-token window-boundary scatter

# tiling
ROPE_CHUNK = 32
K_CHUNK = 512
OUT_CHUNK = 64

HEAD_CHUNK = 64 if B * S >= 64 else 128
K_BLOCKS = D // K_CHUNK            # 8
OUT_BLOCKS = OUT_DIM // OUT_CHUNK  # 8
HEAD_BLOCKS = HEAD_DIM // HEAD_CHUNK  # 4
BATCH_CHUNK_0 = 64
BATCH_CHUNK_1 = 16


@pl.jit.inline
def compressor(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Tensor[[B, S, HEAD_DIM], pl.FP32],
    kv_state: pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32],
    score_state: pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.FP32],
    even_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    odd_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    kv_cache: pl.Tensor[[B, IDX_KV_LEN, HEAD_DIM], pl.BF16],
    start_pos: pl.Scalar[pl.INT32],
    rotate: pl.Scalar[pl.BOOL],
):
    x_flat = pl.reshape(x, [B * S, D])
    ape_row = pl.cast(start_pos % COMPRESS_RATIO, target_type=pl.INDEX)
    pre_tokens = COMPRESS_RATIO - (start_pos % COMPRESS_RATIO)
    scatter_stop = S
    if pre_tokens < S:
        scatter_stop = pre_tokens
    # Non-overlap: scatter into the wrapped slot for each token.
    kv_state_flat = pl.reshape(kv_state, [B, STATE_LEN * OUT_DIM])
    score_state_flat = pl.reshape(score_state, [B, STATE_LEN * OUT_DIM])

    cmp128_kv_proj_scratch = pl.create_tensor([B * S, OUT_DIM], dtype=pl.FP32)
    cmp128_score_proj_scratch = pl.create_tensor([B * S, OUT_DIM], dtype=pl.FP32)
    for o0 in pl.parallel(0, OUT_DIM, OUT_CHUNK):
        for b_idx in pl.parallel(0, B * S, BATCH_CHUNK_0):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_score_proj"):
                kv_acc = pl.create_tensor([BATCH_CHUNK_0, OUT_CHUNK], dtype=pl.FP32)
                score_acc = pl.create_tensor([BATCH_CHUNK_0, OUT_CHUNK], dtype=pl.FP32)
                for kb in pl.pipeline(0, K_BLOCKS, stage=2):
                    k0 = kb * K_CHUNK
                    x_tile = x_flat[b_idx : b_idx + BATCH_CHUNK_0, k0 : k0 + K_CHUNK]
                    wkv_tile = wkv[k0 : k0 + K_CHUNK, o0 : o0 + OUT_CHUNK]
                    wgate_tile = wgate[k0 : k0 + K_CHUNK, o0 : o0 + OUT_CHUNK]
                    if k0 == 0:
                        kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32)
                        score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32)
                    else:
                        kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile)
                        score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile)

                cmp128_kv_proj_scratch = pl.assemble(cmp128_kv_proj_scratch, kv_acc, [b_idx, o0])
                cmp128_score_proj_scratch = pl.assemble(cmp128_score_proj_scratch, score_acc, [b_idx, o0])

        cmp128_kv_proj_by_batch = pl.reshape(cmp128_kv_proj_scratch, [B, S * OUT_DIM])
        cmp128_score_proj_by_batch = pl.reshape(cmp128_score_proj_scratch, [B, S * OUT_DIM])
        for s in pl.range(scatter_stop):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="score_ape_state_scatter"):
                proj_col0 = s * OUT_DIM + o0
                kv_tile = cmp128_kv_proj_by_batch[:, proj_col0 : proj_col0 + OUT_CHUNK]
                score_tile = cmp128_score_proj_by_batch[:, proj_col0 : proj_col0 + OUT_CHUNK]
                token_ape_row = (ape_row + s) % COMPRESS_RATIO
                ape_tile = ape[token_ape_row : token_ape_row + 1, o0 : o0 + OUT_CHUNK]
                ape_base = pl.full([B, OUT_CHUNK], dtype=pl.FP32, value=0.0)
                score_tile = pl.add(score_tile, pl.col_expand(ape_base, ape_tile))
                slot_col0_s = token_ape_row * OUT_DIM
                kv_state_flat = pl.assemble(kv_state_flat, kv_tile, [0, slot_col0_s + o0])
                score_state_flat = pl.assemble(score_state_flat, score_tile, [0, slot_col0_s + o0])

    pooled_kv = pl.create_tensor([B, HEAD_DIM], dtype=pl.FP32)
    normed_kv = pl.create_tensor([B, HEAD_DIM], dtype=pl.BF16)
    kv_final = pl.create_tensor([B, HEAD_DIM], dtype=pl.FP32)
    kv_flat = pl.reshape(kv, [B * S, HEAD_DIM])

    if (start_pos % COMPRESS_RATIO) + S >= COMPRESS_RATIO:
        for b_idx in pl.parallel(0, B, BATCH_CHUNK_1):
            for hb in pl.parallel(0, HEAD_BLOCKS):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="softmax_pool"):
                    h0 = hb * HEAD_CHUNK
                    # Initialize m/l/o from last slot
                    last_col0 = (STATE_LEN - 1) * OUT_DIM + h0
                    mi = score_state_flat[b_idx : b_idx + BATCH_CHUNK_1, last_col0 : last_col0 + HEAD_CHUNK]
                    oi = kv_state_flat[b_idx : b_idx + BATCH_CHUNK_1, last_col0 : last_col0 + HEAD_CHUNK]
                    if pre_tokens < S:
                        pre_s = pre_tokens - 1
                        token_ape_row = (ape_row + pre_s) % COMPRESS_RATIO
                        mi_seed = pl.create_tensor([BATCH_CHUNK_1, HEAD_CHUNK], dtype=pl.FP32)
                        oi_seed = pl.create_tensor([BATCH_CHUNK_1, HEAD_CHUNK], dtype=pl.FP32)
                        for b in pl.range(b_idx, b_idx + BATCH_CHUNK_1):
                            scratch_row = b * S + pre_s
                            mi_seed = pl.assemble(
                                mi_seed,
                                cmp128_score_proj_scratch[scratch_row : scratch_row + 1, h0 : h0 + HEAD_CHUNK],
                                [b - b_idx, 0],
                            )
                            oi_seed = pl.assemble(
                                oi_seed,
                                cmp128_kv_proj_scratch[scratch_row : scratch_row + 1, h0 : h0 + HEAD_CHUNK],
                                [b - b_idx, 0],
                            )
                        mi = mi_seed
                        pool_ape_tile = ape[token_ape_row : token_ape_row + 1, h0 : h0 + HEAD_CHUNK]
                        pool_ape_base = pl.full([BATCH_CHUNK_1, HEAD_CHUNK], dtype=pl.FP32, value=0.0)
                        mi = pl.add(mi, pl.col_expand(pool_ape_base, pool_ape_tile))
                        oi = oi_seed
                    li = pl.exp(pl.sub(mi, mi))

                    # Online softmax over all remaining slots
                    for s in pl.pipeline(0, STATE_LEN - 1, stage=2):
                        col0 = s * OUT_DIM + h0
                        slot_score = score_state_flat[b_idx : b_idx + BATCH_CHUNK_1, col0 : col0 + HEAD_CHUNK]
                        slot_kv = kv_state_flat[b_idx : b_idx + BATCH_CHUNK_1, col0 : col0 + HEAD_CHUNK]
                        mi_next = pl.maximum(mi, slot_score)
                        alpha = pl.exp(pl.sub(mi, mi_next))
                        beta = pl.exp(pl.sub(slot_score, mi_next))
                        li = pl.add(pl.mul(alpha, li), beta)
                        oi = pl.add(pl.mul(oi, alpha), pl.mul(slot_kv, beta))
                        mi = mi_next

                    pooled_chunk = pl.div(oi, li)
                    pooled_kv = pl.assemble(pooled_kv, pooled_chunk, [b_idx, h0])

        # No state shift for non-overlap

        # RMSNorm with BF16 intermediate
        norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
        kv_rope = pl.create_tensor([B, ROPE_HEAD_DIM], dtype=pl.BF16)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm_rope_slice"):
            partial_sq = pl.full([1, B], dtype=pl.FP32, value=0.0)
            for rms_kb in pl.pipeline(HEAD_BLOCKS, stage=4):
                kv_rms_chunk = pl.cast(
                    pl.cast(pooled_kv[:, rms_kb * HEAD_CHUNK : (rms_kb + 1) * HEAD_CHUNK], target_type=pl.BF16, mode="rint"),
                    target_type=pl.FP32,
                )
                partial_sq = pl.add(
                    partial_sq,
                    pl.reshape(pl.row_sum(pl.mul(kv_rms_chunk, kv_rms_chunk)), [1, B]),
                )

            variance = pl.reshape(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS), [B, 1])
            inv_rms = pl.recip(pl.sqrt(variance))
            for rms_kb in pl.pipeline(HEAD_BLOCKS, stage=4):
                kv_norm_chunk = pl.cast(
                    pl.cast(pooled_kv[:, rms_kb * HEAD_CHUNK : (rms_kb + 1) * HEAD_CHUNK], target_type=pl.BF16, mode="rint"),
                    target_type=pl.FP32,
                )
                gamma = norm_w_2d[:, rms_kb * HEAD_CHUNK : (rms_kb + 1) * HEAD_CHUNK]
                normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
                normed_kv = pl.assemble(normed_kv, pl.cast(normed_chunk, target_type=pl.BF16, mode="rint"), [0, rms_kb * HEAD_CHUNK])
            kv_rope = pl.assemble(kv_rope, normed_kv[:, NOPE_HEAD_DIM : HEAD_DIM], [0, 0])

        # Selector-based RoPE
        kv_proj_even = pl.create_tensor([B, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
        kv_proj_odd = pl.create_tensor([B, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
        rope_even = pl.create_tensor([B, ROPE_HEAD_DIM // 2], dtype=pl.BF16)
        rope_odd = pl.create_tensor([B, ROPE_HEAD_DIM // 2], dtype=pl.BF16)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_slice"):
            even_acc = pl.matmul(kv_rope, even_select, out_dtype=pl.FP32)
            odd_acc = pl.matmul(kv_rope, odd_select, out_dtype=pl.FP32)
            kv_proj_even = pl.assemble(kv_proj_even, even_acc, [0, 0])
            kv_proj_odd = pl.assemble(kv_proj_odd, odd_acc, [0, 0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_apply"):
            even_tile = kv_proj_even[:, :]
            odd_tile = kv_proj_odd[:, :]
            rope_even_acc = pl.cast(pl.sub(pl.col_expand_mul(even_tile, cos), pl.col_expand_mul(odd_tile, sin)), target_type=pl.BF16, mode="rint")
            rope_odd_acc = pl.cast(pl.add(pl.col_expand_mul(even_tile, sin), pl.col_expand_mul(odd_tile, cos)), target_type=pl.BF16, mode="rint")
            rope_even = pl.assemble(rope_even, rope_even_acc, [0, 0])
            rope_odd = pl.assemble(rope_odd, rope_odd_acc, [0, 0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_assemble"):
            rope_acc = pl.matmul(rope_even, even_select, out_dtype=pl.FP32, b_trans=True)
            rope_acc = pl.matmul_acc(rope_acc, rope_odd, odd_select, b_trans=True)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_write"):
            normed_kv = pl.assemble(normed_kv, pl.cast(rope_acc, target_type=pl.BF16, mode="rint"), [0, NOPE_HEAD_DIM])

        if rotate:
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_hadamard"):
                kv_proj_tile = normed_kv[:, 0 : HEAD_DIM]
                for o0 in pl.range(0, HEAD_DIM, OUT_CHUNK):
                    hadamard_tile = hadamard[0 : HEAD_DIM, o0 : o0 + OUT_CHUNK]
                    kv_hadamard_acc = pl.matmul(kv_proj_tile, hadamard_tile, out_dtype=pl.FP32)
                    kv_final = pl.assemble(kv_final, kv_hadamard_acc, [0, o0])
        else:
            for o0 in pl.parallel(0, HEAD_DIM, OUT_CHUNK):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_write"):
                    kv_out_tile = normed_kv[:, o0 : o0 + OUT_CHUNK]
                    kv_final = pl.assemble(kv_final, pl.cast(kv_out_tile, target_type=pl.FP32), [0, o0])

        # Per-batch fan-out: write kv_final[b] to kv[b, 0, :] (row b*S of kv_flat).
        kv_cache_flat = pl.reshape(kv_cache, [B * IDX_KV_LEN, HEAD_DIM])
        cache_col = start_pos // COMPRESS_RATIO
        with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk], name_hint="kv_and_cache_write"):
            for b_idx in pl.parallel(0, B, chunk=16):
                kv_row_fp32 = kv_final[b_idx : b_idx + 1, 0 : HEAD_DIM]
                kv_flat = pl.assemble(kv_flat, kv_row_fp32, [b_idx * S, 0])
                cache_row = b_idx * IDX_KV_LEN + cache_col
                kv_cache_flat = pl.assemble(
                    kv_cache_flat,
                    pl.cast(kv_row_fp32, target_type=pl.BF16, mode="rint"),
                    [cache_row, 0],
                )
        kv_cache = pl.reshape(kv_cache_flat, [B, IDX_KV_LEN, HEAD_DIM])

    if pre_tokens < S:
        cmp128_kv_proj_by_batch = pl.reshape(cmp128_kv_proj_scratch, [B, S * OUT_DIM])
        cmp128_score_proj_by_batch = pl.reshape(cmp128_score_proj_scratch, [B, S * OUT_DIM])
        for o0 in pl.parallel(0, OUT_DIM, OUT_CHUNK):
            for s in pl.range(pre_tokens, S):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_scatter_next"):
                    proj_col0 = s * OUT_DIM + o0
                    kv_tile = cmp128_kv_proj_by_batch[:, proj_col0 : proj_col0 + OUT_CHUNK]
                    score_tile = cmp128_score_proj_by_batch[:, proj_col0 : proj_col0 + OUT_CHUNK]
                    dep_tile = kv_final[:, o0 : o0 + OUT_CHUNK]
                    dep_zero = pl.sub(dep_tile, dep_tile)
                    kv_tile = pl.add(kv_tile, dep_zero)
                    score_tile = pl.add(score_tile, dep_zero)
                    token_ape_row = (ape_row + s) % COMPRESS_RATIO
                    ape_tile = ape[token_ape_row : token_ape_row + 1, o0 : o0 + OUT_CHUNK]
                    ape_base = pl.full([B, OUT_CHUNK], dtype=pl.FP32, value=0.0)
                    score_tile = pl.add(score_tile, pl.col_expand(ape_base, ape_tile))
                    slot_col0_s = token_ape_row * OUT_DIM
                    kv_state_flat = pl.assemble(kv_state_flat, kv_tile, [0, slot_col0_s + o0])
                    score_state_flat = pl.assemble(score_state_flat, score_tile, [0, slot_col0_s + o0])

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
    cos: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.FP32],
    even_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    odd_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    kv_cache: pl.Out[pl.Tensor[[B, IDX_KV_LEN, HEAD_DIM], pl.BF16]],
    start_pos: pl.Scalar[pl.INT32],
    rotate: pl.Scalar[pl.BOOL],
):
    kv, kv_state, score_state, kv_cache = compressor(
        x, kv, kv_state, score_state, wkv, wgate, ape, norm_w, cos, sin, even_select, odd_select, hadamard, kv_cache, start_pos, rotate
    )
    return kv, kv_state, score_state, kv_cache


def golden_compressor(tensors):
    """Torch reference for Compressor.forward (decode branch, ratio=128 non-overlap)."""
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
    start_pos = int(tensors["start_pos"])
    rotate = bool(tensors["rotate"])
    bsz, _, _ = x.shape
    ratio, rd = COMPRESS_RATIO, ROPE_HEAD_DIM

    kv = x @ wkv                        # [B, S, OUT_DIM]
    score = x @ wgate                   # [B, S, OUT_DIM]
    kv_proj = kv

    pre_tokens = min(S, ratio - (start_pos % ratio))
    should_compress = pre_tokens < S or (start_pos + S) % ratio == 0

    # Non-overlap: per-token ape add + scatter to wrapped slots.
    ape_row_g = start_pos % ratio
    for s in range(pre_tokens):
        token_ape_row = (ape_row_g + s) % ratio
        score[:, s, :] = score[:, s, :] + ape[token_ape_row]
        kv_state[:bsz, token_ape_row] = kv[:, s, :]
        score_state[:bsz, token_ape_row] = score[:, s, :]

    if should_compress:
        kv = (kv_state[:bsz] * score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)   # [B, 1, HEAD_DIM]

    if pre_tokens < S:
        for s in range(pre_tokens, S):
            token_ape_row = (ape_row_g + s) % ratio
            score[:, s, :] = score[:, s, :] + ape[token_ape_row]
            kv_state[:bsz, token_ape_row] = kv_proj[:, s, :]
            score_state[:bsz, token_ape_row] = score[:, s, :]

    tensors["kv_state"][:] = kv_state
    tensors["score_state"][:] = score_state

    if not should_compress:
        return

    def rmsnorm(x, w):
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + EPS)
        return (w * x).to(torch.bfloat16)

    kv = rmsnorm(kv.to(torch.bfloat16), norm_w)

    x_pair = kv[..., -rd:].unflatten(-1, (-1, 2))
    x0, x1 = x_pair[..., 0], x_pair[..., 1]
    cos_v, sin_v = cos.view(-1), sin.view(-1)
    y0 = (x0 * cos_v - x1 * sin_v).to(torch.bfloat16)
    y1 = (x0 * sin_v + x1 * cos_v).to(torch.bfloat16)

    kv = torch.cat([kv[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1).float()

    if rotate:
        kv = kv @ hadamard
    # Kernel writes pooled result only to kv[:, 0, :]; leave kv[:, 1:, :] = 0.
    tensors["kv"][:, 0:1, :] = kv

    kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)

    tensors["kv_cache"][:] = kv_cache


def build_tensor_specs(start_pos: int = START_POS):
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
        return torch.rand(1, ROPE_HEAD_DIM // 2)
    def init_sin():
        return torch.rand(1, ROPE_HEAD_DIM // 2)
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

    return [
        TensorSpec("x", [B, S, D], torch.bfloat16, init_value=init_x),
        TensorSpec("kv", [B, S, HEAD_DIM], torch.float32, is_output=True),
        TensorSpec("kv_state", [B, STATE_LEN, OUT_DIM], torch.float32, init_value=init_kv_state, is_output=True),
        TensorSpec("score_state", [B, STATE_LEN, OUT_DIM], torch.float32, init_value=init_score_state, is_output=True),
        TensorSpec("wkv", [D, OUT_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [D, OUT_DIM], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.float32, init_value=init_norm_w),
        TensorSpec("cos", [1, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_cos),
        TensorSpec("sin", [1, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_sin),
        TensorSpec("even_select", [ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], torch.bfloat16, init_value=init_even_select),
        TensorSpec("odd_select", [ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], torch.bfloat16, init_value=init_odd_select),
        TensorSpec("hadamard", [HEAD_DIM, HEAD_DIM], torch.bfloat16, init_value=init_hadamard),
        TensorSpec("kv_cache", [B, IDX_KV_LEN, HEAD_DIM], torch.bfloat16, init_value=init_kv_cache, is_output=True),
        ScalarSpec("start_pos", torch.int32, start_pos),
        ScalarSpec("rotate", torch.bool, True),
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
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=compressor_test,
        specs=build_tensor_specs(args.start_pos),
        golden_fn=golden_compressor,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
            enable_pmu=2,
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
