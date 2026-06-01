# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F401,F403,F405,F821
"""DeepSeek-V4 prefill indexer compressor for ratio-4 overlapping KV cache."""

import pypto.language as pl

from config import FP32_NEG_INF
from decode_indexer_compressor import *  # noqa: F401,F403

B = 1
S = 128
START_POS = 0
PREFILL_COMPRESSED_LEN = S // COMPRESS_RATIO
PREFILL_ROWS = B * PREFILL_COMPRESSED_LEN
HEAD_CHUNK = 32
HEAD_BLOCKS = HEAD_DIM // HEAD_CHUNK
K_CHUNK = 512
OUT_CHUNK = 64
ROPE_CHUCK = 32

@pl.jit.inline
def prefill_indexer_compressor(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Tensor[[B, PREFILL_COMPRESSED_LEN, HEAD_DIM], pl.FP32],
    kv_state: pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32],
    score_state: pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[PREFILL_COMPRESSED_LEN, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[PREFILL_COMPRESSED_LEN, ROPE_HEAD_DIM // 2], pl.FP32],
    even_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    odd_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    kv_cache: pl.Tensor[[B, IDX_KV_LEN, HEAD_DIM], pl.BF16],
    start_pos: pl.Scalar[pl.INT32],
    rotate: pl.Scalar[pl.BOOL],
):
    x_flat = pl.reshape(x, [B * S, D])
    kv_proj_scratch = pl.create_tensor([B * S, OUT_DIM], dtype=pl.FP32)
    score_proj_scratch = pl.create_tensor([B * S, OUT_DIM], dtype=pl.FP32)
    kv_state_flat = pl.reshape(kv_state, [B, STATE_LEN * OUT_DIM])
    score_state_flat = pl.reshape(score_state, [B, STATE_LEN * OUT_DIM])
    kv_flat = pl.reshape(kv, [B * PREFILL_COMPRESSED_LEN, HEAD_DIM])
    kv_cache_flat = pl.reshape(kv_cache, [B * IDX_KV_LEN, HEAD_DIM])

    for o0 in pl.parallel(0, OUT_DIM, OUT_CHUNK):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_kv_score_proj"):
            kv_acc = pl.create_tensor([B * S, OUT_CHUNK], dtype=pl.FP32)
            score_acc = pl.create_tensor([B * S, OUT_CHUNK], dtype=pl.FP32)
            for kb in pl.pipeline(0, D // K_CHUNK, stage=2):
                k0 = kb * K_CHUNK
                x_tile = x_flat[:, k0 : k0 + K_CHUNK]
                wkv_tile = wkv[k0 : k0 + K_CHUNK, o0 : o0 + OUT_CHUNK]
                wgate_tile = wgate[k0 : k0 + K_CHUNK, o0 : o0 + OUT_CHUNK]
                if k0 == 0:
                    kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32)
                    score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32)
                else:
                    kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile)
                    score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile)

            kv_proj_scratch[:, o0 : o0 + OUT_CHUNK] = kv_acc
            score_proj_scratch[:, o0 : o0 + OUT_CHUNK] = score_acc

    kv_proj = pl.reshape(kv_proj_scratch, [B, S, OUT_DIM])
    score_proj = pl.reshape(score_proj_scratch, [B, S, OUT_DIM])

    for o0 in pl.parallel(0, OUT_DIM, OUT_CHUNK):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_final_state_write"):
            for s in pl.range(COMPRESS_RATIO):
                src_token = S - COMPRESS_RATIO + s
                state_col0 = s * OUT_DIM + o0
                kv_tile = pl.reshape(kv_proj[:, src_token : src_token + 1, o0 : o0 + OUT_CHUNK], [B, OUT_CHUNK])
                score_tile = pl.reshape(score_proj[:, src_token : src_token + 1, o0 : o0 + OUT_CHUNK], [B, OUT_CHUNK])
                ape_tile = ape[s : s + 1, o0 : o0 + OUT_CHUNK]
                ape_base = pl.full([B, OUT_CHUNK], dtype=pl.FP32, value=0.0)
                score_tile = pl.add(score_tile, pl.col_expand(ape_base, ape_tile))
                kv_state_flat[:, state_col0 : state_col0 + OUT_CHUNK] = kv_tile
                score_state_flat[:, state_col0 : state_col0 + OUT_CHUNK] = score_tile

    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    cache_base = start_pos // COMPRESS_RATIO
    pooled_kv_all = pl.create_tensor([PREFILL_COMPRESSED_LEN, HEAD_DIM], dtype=pl.FP32)

    for c in pl.range(PREFILL_COMPRESSED_LEN):
        pooled_kv = pl.create_tensor([B, HEAD_DIM], dtype=pl.FP32)
        token0 = c * COMPRESS_RATIO

        for hb in pl.parallel(0, HEAD_BLOCKS, 1):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_softmax_pool"):
                h0 = hb * HEAD_CHUNK
                back_col0 = HEAD_DIM + h0
                init_token = token0 + COMPRESS_RATIO - 1
                mi = pl.reshape(
                    score_proj[:, init_token : init_token + 1, back_col0 : back_col0 + HEAD_CHUNK],
                    [B, HEAD_CHUNK],
                )
                oi = pl.reshape(
                    kv_proj[:, init_token : init_token + 1, back_col0 : back_col0 + HEAD_CHUNK],
                    [B, HEAD_CHUNK],
                )
                ape_tile = ape[COMPRESS_RATIO - 1 : COMPRESS_RATIO, back_col0 : back_col0 + HEAD_CHUNK]
                ape_base = pl.full([B, HEAD_CHUNK], dtype=pl.FP32, value=0.0)
                mi = pl.add(mi, pl.col_expand(ape_base, ape_tile))
                li = pl.exp(pl.sub(mi, mi))

                if c > 0:
                    prev_token0 = token0 - COMPRESS_RATIO
                    for s in pl.range(COMPRESS_RATIO):
                        front_token = prev_token0 + s
                        front_score = pl.reshape(
                            score_proj[:, front_token : front_token + 1, h0 : h0 + HEAD_CHUNK],
                            [B, HEAD_CHUNK],
                        )
                        front_kv = pl.reshape(
                            kv_proj[:, front_token : front_token + 1, h0 : h0 + HEAD_CHUNK],
                            [B, HEAD_CHUNK],
                        )
                        front_ape = ape[s : s + 1, h0 : h0 + HEAD_CHUNK]
                        front_score = pl.add(front_score, pl.col_expand(ape_base, front_ape))
                        mi_next_front = pl.maximum(mi, front_score)
                        alpha_front = pl.exp(pl.sub(mi, mi_next_front))
                        beta_front = pl.exp(pl.sub(front_score, mi_next_front))
                        li = pl.add(pl.mul(alpha_front, li), beta_front)
                        oi = pl.add(pl.mul(oi, alpha_front), pl.mul(front_kv, beta_front))
                        mi = mi_next_front

                for s in pl.range(0, COMPRESS_RATIO - 1):
                    back_token = token0 + s
                    back_score = pl.reshape(
                        score_proj[:, back_token : back_token + 1, back_col0 : back_col0 + HEAD_CHUNK],
                        [B, HEAD_CHUNK],
                    )
                    back_kv = pl.reshape(
                        kv_proj[:, back_token : back_token + 1, back_col0 : back_col0 + HEAD_CHUNK],
                        [B, HEAD_CHUNK],
                    )
                    back_ape = ape[s : s + 1, back_col0 : back_col0 + HEAD_CHUNK]
                    back_score = pl.add(back_score, pl.col_expand(ape_base, back_ape))
                    mi_next_back = pl.maximum(mi, back_score)
                    alpha_back = pl.exp(pl.sub(mi, mi_next_back))
                    beta_back = pl.exp(pl.sub(back_score, mi_next_back))
                    li = pl.add(pl.mul(alpha_back, li), beta_back)
                    oi = pl.add(pl.mul(oi, alpha_back), pl.mul(back_kv, beta_back))
                    mi = mi_next_back

                pooled_kv[:, h0 : h0 + HEAD_CHUNK] = pl.div(oi, li)

        pooled_kv_all = pl.assemble(pooled_kv_all, pooled_kv, [c, 0])

    normed_kv_all = pl.create_tensor([PREFILL_COMPRESSED_LEN, HEAD_DIM], dtype=pl.BF16)
    kv_final_all = pl.create_tensor([PREFILL_COMPRESSED_LEN, HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_rmsnorm"):
        partial_sq = pl.full([PREFILL_COMPRESSED_LEN, HEAD_CHUNK], dtype=pl.FP32, value=0.0)
        for k0 in pl.range(0, HEAD_DIM, HEAD_CHUNK):
            kv_rms_chunk = pl.cast(
                pl.cast(pooled_kv_all[:, k0 : k0 + HEAD_CHUNK], target_type=pl.BF16, mode="rint"),
                target_type=pl.FP32,
            )
            partial_sq = pl.add(partial_sq, pl.mul(kv_rms_chunk, kv_rms_chunk))

        reduce_ones = pl.full([HEAD_CHUNK, HEAD_CHUNK], dtype=pl.FP32, value=1.0)
        sq_sum = pl.matmul(partial_sq, reduce_ones, out_dtype=pl.FP32)
        inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, HEAD_DIM_INV), EPS)))
        for k0 in pl.range(0, HEAD_DIM, HEAD_CHUNK):
            kv_norm_chunk = pl.cast(
                pl.cast(pooled_kv_all[:, k0 : k0 + HEAD_CHUNK], target_type=pl.BF16, mode="rint"),
                target_type=pl.FP32,
            )
            gamma = norm_w_2d[:, k0 : k0 + HEAD_CHUNK]
            normed_chunk = pl.col_expand_mul(pl.mul(kv_norm_chunk, inv_rms), gamma)
            normed_kv_all[:, k0 : k0 + HEAD_CHUNK] = pl.cast(normed_chunk, target_type=pl.BF16, mode="rint")

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_rope_slice"):
        even_acc = pl.create_tensor([PREFILL_COMPRESSED_LEN, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
        odd_acc = pl.create_tensor([PREFILL_COMPRESSED_LEN, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
        for rb in pl.pipeline(0, ROPE_HEAD_DIM // ROPE_CHUCK, stage=2):
            r0 = rb * ROPE_CHUCK
            kv_rope_tile = normed_kv_all[:, NOPE_HEAD_DIM + r0 : NOPE_HEAD_DIM + r0 + ROPE_CHUCK]
            even_select_tile = even_select[r0 : r0 + ROPE_CHUCK, :]
            odd_select_tile = odd_select[r0 : r0 + ROPE_CHUCK, :]
            if r0 == 0:
                even_acc = pl.matmul(kv_rope_tile, even_select_tile, out_dtype=pl.FP32)
                odd_acc = pl.matmul(kv_rope_tile, odd_select_tile, out_dtype=pl.FP32)
            else:
                even_acc = pl.matmul_acc(even_acc, kv_rope_tile, even_select_tile)
                odd_acc = pl.matmul_acc(odd_acc, kv_rope_tile, odd_select_tile)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_rope_apply"):
        rope_even_acc = pl.cast(
            pl.sub(pl.mul(even_acc, cos[:, :]), pl.mul(odd_acc, sin[:, :])),
            target_type=pl.BF16,
            mode="rint",
        )
        rope_odd_acc = pl.cast(
            pl.add(pl.mul(even_acc, sin[:, :]), pl.mul(odd_acc, cos[:, :])),
            target_type=pl.BF16,
            mode="rint",
        )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_rope_assemble"):
        rope_acc = pl.create_tensor([PREFILL_COMPRESSED_LEN, ROPE_HEAD_DIM], dtype=pl.FP32)
        for ra_b in pl.pipeline(0, (ROPE_HEAD_DIM // 2) // ROPE_CHUCK, stage=2):
            ra_0 = ra_b * ROPE_CHUCK
            rope_even_tile = rope_even_acc[:, ra_0 : ra_0 + ROPE_CHUCK]
            rope_odd_tile = rope_odd_acc[:, ra_0 : ra_0 + ROPE_CHUCK]
            even_select_tile_t = even_select[:, ra_0 : ra_0 + ROPE_CHUCK]
            odd_select_tile_t = odd_select[:, ra_0 : ra_0 + ROPE_CHUCK]
            if ra_0 == 0:
                rope_acc = pl.matmul(rope_even_tile, even_select_tile_t, out_dtype=pl.FP32, b_trans=True)
            else:
                rope_acc = pl.matmul_acc(rope_acc, rope_even_tile, even_select_tile_t, b_trans=True)
            rope_acc = pl.matmul_acc(rope_acc, rope_odd_tile, odd_select_tile_t, b_trans=True)

        normed_kv_all[:, NOPE_HEAD_DIM : NOPE_HEAD_DIM + ROPE_HEAD_DIM] = pl.cast(
            rope_acc,
            target_type=pl.BF16,
            mode="rint",
        )

    if rotate:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_kv_hadamard"):
            kv_final_all[:, :] = pl.matmul(
                normed_kv_all[:, 0 : HEAD_DIM],
                hadamard[0 : HEAD_DIM, 0 : HEAD_DIM],
                out_dtype=pl.FP32,
            )
    else:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_kv_write"):
            kv_final_all[:, :] = pl.cast(normed_kv_all[:, 0 : HEAD_DIM], target_type=pl.FP32)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_kv_and_cache_write"):
        kv_flat[:, :] = kv_final_all
        kv_cache_flat[cache_base : cache_base + PREFILL_COMPRESSED_LEN, :] = pl.cast(
            kv_final_all,
            target_type=pl.BF16,
            mode="rint",
        )

    kv_state = pl.reshape(kv_state_flat, [B, STATE_LEN, OUT_DIM])
    score_state = pl.reshape(score_state_flat, [B, STATE_LEN, OUT_DIM])
    kv = pl.reshape(kv_flat, [B, PREFILL_COMPRESSED_LEN, HEAD_DIM])
    kv_cache = pl.reshape(kv_cache_flat, [B, IDX_KV_LEN, HEAD_DIM])
    return kv, kv_state, score_state, kv_cache


@pl.jit
def prefill_indexer_compressor_test(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Out[pl.Tensor[[B, PREFILL_COMPRESSED_LEN, HEAD_DIM], pl.FP32]],
    kv_state: pl.Out[pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32]],
    score_state: pl.Out[pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32]],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[PREFILL_COMPRESSED_LEN, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[PREFILL_COMPRESSED_LEN, ROPE_HEAD_DIM // 2], pl.FP32],
    even_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    odd_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    kv_cache: pl.Out[pl.Tensor[[B, IDX_KV_LEN, HEAD_DIM], pl.BF16]],
    start_pos: pl.Scalar[pl.INT32],
    rotate: pl.Scalar[pl.BOOL],
):
    kv, kv_state, score_state, kv_cache = prefill_indexer_compressor(
        x, kv, kv_state, score_state, wkv, wgate, ape, norm_w, cos, sin, even_select, odd_select, hadamard, kv_cache, start_pos, rotate
    )
    return kv, kv_state, score_state, kv_cache


def golden_prefill_indexer_compressor(tensors):
    """Torch reference for Compressor.forward prefill branch, ratio=4 overlap."""
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
    bsz, seqlen, _ = x.shape
    ratio, d, rd = COMPRESS_RATIO, HEAD_DIM, ROPE_HEAD_DIM

    if start_pos != 0:
        raise ValueError("golden_prefill_indexer_compressor expects start_pos == 0")

    tensors["kv"].zero_()
    kv_proj = x @ wkv
    score_proj = x @ wgate

    should_compress = seqlen >= ratio
    remainder = seqlen % ratio
    cutoff = seqlen - remainder
    n_comp = cutoff // ratio

    if cutoff >= ratio:
        kv_state[:bsz, :ratio] = kv_proj[:, cutoff - ratio : cutoff]
        score_state[:bsz, :ratio] = score_proj[:, cutoff - ratio : cutoff] + ape

    if remainder > 0:
        kv_state[:bsz, ratio : ratio + remainder] = kv_proj[:, cutoff:]
        score_state[:bsz, ratio : ratio + remainder] = score_proj[:, cutoff:] + ape[:remainder]

    tensors["kv_state"][:] = kv_state
    tensors["score_state"][:] = score_state

    if not should_compress:
        return

    kv_blocks = kv_proj[:, :cutoff].unflatten(1, (-1, ratio))
    score_blocks = score_proj[:, :cutoff].unflatten(1, (-1, ratio)) + ape

    kv_overlap = kv_blocks.new_zeros((bsz, n_comp, 2 * ratio, d))
    score_overlap = score_blocks.new_full((bsz, n_comp, 2 * ratio, d), FP32_NEG_INF)
    kv_overlap[:, :, ratio:] = kv_blocks[:, :, :, d:]
    score_overlap[:, :, ratio:] = score_blocks[:, :, :, d:]
    kv_overlap[:, 1:, :ratio] = kv_blocks[:, :-1, :, :d]
    score_overlap[:, 1:, :ratio] = score_blocks[:, :-1, :, :d]
    kv = (kv_overlap * score_overlap.softmax(dim=2)).sum(dim=2)

    def rmsnorm(value, weight):
        value = value.float()
        var = value.square().mean(-1, keepdim=True)
        value = value * torch.rsqrt(var + EPS)
        return weight * value

    kv = rmsnorm(kv, norm_w)

    x_pair = kv[..., -rd:].unflatten(-1, (-1, 2))
    x0, x1 = x_pair[..., 0], x_pair[..., 1]
    cos_v, sin_v = cos[:n_comp].view(1, n_comp, -1), sin[:n_comp].view(1, n_comp, -1)
    y0 = x0 * cos_v - x1 * sin_v
    y1 = x0 * sin_v + x1 * cos_v
    kv = torch.cat([kv[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1)

    if rotate:
        kv = kv.to(torch.bfloat16).float() @ hadamard

    tensors["kv"][:] = kv
    kv_cache[:bsz, :n_comp] = kv.to(kv_cache.dtype)
    tensors["kv_cache"][:] = kv_cache


def build_tensor_specs():
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
        return torch.rand(PREFILL_COMPRESSED_LEN, ROPE_HEAD_DIM // 2)
    def init_sin():
        return torch.rand(PREFILL_COMPRESSED_LEN, ROPE_HEAD_DIM // 2)
    def init_odd_select():
        matrix = torch.zeros((ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2))
        for i in range(ROPE_HEAD_DIM // 2):
            matrix[2 * i + 1, i] = 1
        return matrix
    def init_even_select():
        matrix = torch.zeros((ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2))
        for i in range(ROPE_HEAD_DIM // 2):
            matrix[2 * i, i] = 1
        return matrix
    def init_hadamard():
        return torch.rand(HEAD_DIM, HEAD_DIM) * (HEAD_DIM ** -0.5)
    def init_kv_cache():
        return torch.zeros(B, IDX_KV_LEN, HEAD_DIM)

    return [
        TensorSpec("x", [B, S, D], torch.bfloat16, init_value=init_x),
        TensorSpec("kv", [B, PREFILL_COMPRESSED_LEN, HEAD_DIM], torch.float32, is_output=True),
        TensorSpec("kv_state", [B, STATE_LEN, OUT_DIM], torch.float32, init_value=init_kv_state, is_output=True),
        TensorSpec("score_state", [B, STATE_LEN, OUT_DIM], torch.float32, init_value=init_score_state, is_output=True),
        TensorSpec("wkv", [D, OUT_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [D, OUT_DIM], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.float32, init_value=init_norm_w),
        TensorSpec("cos", [PREFILL_COMPRESSED_LEN, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_cos),
        TensorSpec("sin", [PREFILL_COMPRESSED_LEN, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_sin),
        TensorSpec("even_select", [ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], torch.bfloat16, init_value=init_even_select),
        TensorSpec("odd_select", [ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], torch.bfloat16, init_value=init_odd_select),
        TensorSpec("hadamard", [HEAD_DIM, HEAD_DIM], torch.bfloat16, init_value=init_hadamard),
        TensorSpec("kv_cache", [B, IDX_KV_LEN, HEAD_DIM], torch.bfloat16, init_value=init_kv_cache, is_output=True),
        ScalarSpec("start_pos", torch.int32, START_POS),
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
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=prefill_indexer_compressor_test,
        specs=build_tensor_specs(),
        golden_fn=golden_prefill_indexer_compressor,
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
            "kv_cache":    ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.005 / (IDX_CACHE_BLOCK_NUM * BLOCK_SIZE)),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
