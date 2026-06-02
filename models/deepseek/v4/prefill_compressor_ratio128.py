# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 prefill compressor, ratio=128 bring-up.

This standalone module mirrors the ratio-128 compressor used by
prefill_attention_hca.py: project KV/score, add APE, softmax-pool the
128-token prompt chunk, then apply the compressor RMSNorm/RoPE before
publishing one compressed KV row.
"""

import pypto.language as pl

from config import FLASH as M, PREFILL_BATCH, PREFILL_SEQ


B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
EPS = M.rms_norm_eps
D = M.hidden_size
HEAD_DIM = M.head_dim
HEAD_DIM_INV = 1.0 / HEAD_DIM
ROPE_HEAD_DIM = M.qk_rope_head_dim
ROPE_HALF = ROPE_HEAD_DIM // 2
NOPE_HEAD_DIM = HEAD_DIM - ROPE_HEAD_DIM
MAX_SEQ_LEN = M.max_position_embeddings

COMPRESS_RATIO = 128
OUT_DIM = HEAD_DIM
STATE_LEN = COMPRESS_RATIO
IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
PREFILL_COMPRESSED_LEN = S // COMPRESS_RATIO
START_POS = 0
ROTATE = False

K_CHUNK = 512
OUT_CHUNK = 32
HEAD_CHUNK = 64
RMS_TILE = 16
RMS_BLOCKS = (B + RMS_TILE - 1) // RMS_TILE
RMS_PAD_ROWS = RMS_BLOCKS * RMS_TILE
T_CHUNK = S
K_BLOCKS = D // K_CHUNK
OUT_BLOCKS = OUT_DIM // OUT_CHUNK
HEAD_BLOCKS = HEAD_DIM // HEAD_CHUNK
PROJ_BLOCKS = B * OUT_BLOCKS
POOL_BLOCKS = B * HEAD_BLOCKS

assert S == COMPRESS_RATIO, "ratio128 prefill compressor bring-up expects one full compression chunk"
assert PREFILL_COMPRESSED_LEN == 1


@pl.jit.inline
def prefill_compressor_ratio128(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Tensor[[B, PREFILL_COMPRESSED_LEN, HEAD_DIM], pl.FP32],
    kv_state: pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32],
    score_state: pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[PREFILL_COMPRESSED_LEN, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[PREFILL_COMPRESSED_LEN, ROPE_HALF], pl.FP32],
    even_idx: pl.Tensor[[1, ROPE_HALF], pl.INT32],
    odd_idx: pl.Tensor[[1, ROPE_HALF], pl.INT32],
    kv_cache: pl.Tensor[[B, IDX_KV_LEN, HEAD_DIM], pl.BF16],
    start_pos: pl.Scalar[pl.INT32],
):
    x_flat = pl.reshape(x, [B * S, D])
    kv_proj = pl.create_tensor([B * S, OUT_DIM], dtype=pl.FP32)
    score_proj = pl.create_tensor([B * S, OUT_DIM], dtype=pl.FP32)
    kv_flat = pl.reshape(kv, [B * PREFILL_COMPRESSED_LEN, HEAD_DIM])
    kv_cache_flat = pl.reshape(kv_cache, [B * IDX_KV_LEN, HEAD_DIM])
    kv_state_flat = pl.reshape(kv_state, [B * STATE_LEN, OUT_DIM])
    score_state_flat = pl.reshape(score_state, [B * STATE_LEN, OUT_DIM])
    pooled_kv_pad = pl.create_tensor([RMS_PAD_ROWS, HEAD_DIM], dtype=pl.FP32)
    normed_kv_pad = pl.create_tensor([RMS_PAD_ROWS, HEAD_DIM], dtype=pl.FP32)

    for pad_block in pl.spmd(RMS_BLOCKS, name_hint="prefill_c128_norm_pad_init"):
        pad_row = pad_block * RMS_TILE
        for hb in pl.pipeline(HEAD_BLOCKS, stage=2):
            init_h0 = hb * HEAD_CHUNK
            pooled_kv_pad[pad_row : pad_row + RMS_TILE, init_h0 : init_h0 + HEAD_CHUNK] = pl.full(
                [RMS_TILE, HEAD_CHUNK],
                dtype=pl.FP32,
                value=0.0,
            )
            normed_kv_pad[pad_row : pad_row + RMS_TILE, init_h0 : init_h0 + HEAD_CHUNK] = pl.full(
                [RMS_TILE, HEAD_CHUNK],
                dtype=pl.FP32,
                value=0.0,
            )

    for proj_idx in pl.spmd(PROJ_BLOCKS, name_hint="prefill_c128_proj"):
        batch_idx = proj_idx // OUT_BLOCKS
        o0 = (proj_idx - batch_idx * OUT_BLOCKS) * OUT_CHUNK
        t0 = batch_idx * S
        kv_acc = pl.create_tensor([T_CHUNK, OUT_CHUNK], dtype=pl.FP32)
        score_acc = pl.create_tensor([T_CHUNK, OUT_CHUNK], dtype=pl.FP32)
        for kb in pl.pipeline(0, K_BLOCKS, stage=2):
            k0 = kb * K_CHUNK
            x_tile = x_flat[t0 : t0 + T_CHUNK, k0 : k0 + K_CHUNK]
            wkv_tile = wkv[k0 : k0 + K_CHUNK, o0 : o0 + OUT_CHUNK]
            wgate_tile = wgate[k0 : k0 + K_CHUNK, o0 : o0 + OUT_CHUNK]
            if k0 == 0:
                kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32)
                score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32)
            else:
                kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile)
                score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile)
        score_acc = pl.add(score_acc, ape[0:T_CHUNK, o0 : o0 + OUT_CHUNK])
        kv_proj[t0 : t0 + T_CHUNK, o0 : o0 + OUT_CHUNK] = kv_acc
        score_proj[t0 : t0 + T_CHUNK, o0 : o0 + OUT_CHUNK] = score_acc
        kv_state_flat[t0 : t0 + T_CHUNK, o0 : o0 + OUT_CHUNK] = kv_acc
        score_state_flat[t0 : t0 + T_CHUNK, o0 : o0 + OUT_CHUNK] = score_acc

    for pool_idx in pl.spmd(POOL_BLOCKS, name_hint="prefill_c128_softmax_pool"):
        pool_b = pool_idx // HEAD_BLOCKS
        hb = pool_idx - pool_b * HEAD_BLOCKS
        pool_h0 = hb * HEAD_CHUNK
        t0 = pool_b * S
        score_tile = score_proj[t0 : t0 + S, pool_h0 : pool_h0 + HEAD_CHUNK]
        kv_tile = kv_proj[t0 : t0 + S, pool_h0 : pool_h0 + HEAD_CHUNK]
        score_t = pl.transpose(score_tile, axis1=0, axis2=1)
        kv_t = pl.transpose(kv_tile, axis1=0, axis2=1)
        score_max = pl.row_max(score_t)
        score_exp = pl.exp(pl.row_expand_sub(score_t, score_max))
        score_sum = pl.row_sum(score_exp)
        score_prob = pl.row_expand_div(score_exp, score_sum)
        pooled_t = pl.row_sum(pl.mul(kv_t, score_prob))
        pooled_chunk = pl.reshape(pooled_t, [1, HEAD_CHUNK])
        pooled_bf16 = pl.cast(pooled_chunk, target_type=pl.BF16, mode="rint")
        kv_row = pool_b * PREFILL_COMPRESSED_LEN
        kv_flat[kv_row : kv_row + 1, pool_h0 : pool_h0 + HEAD_CHUNK] = pl.cast(pooled_bf16, target_type=pl.FP32)
        pooled_kv_pad[pool_b : pool_b + 1, pool_h0 : pool_h0 + HEAD_CHUNK] = pl.cast(pooled_bf16, target_type=pl.FP32)

    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    for norm_block in pl.spmd(RMS_BLOCKS, name_hint="prefill_c128_norm_rope"):
        batch_base = norm_block * RMS_TILE
        rope_row_ones = pl.full([RMS_TILE, ROPE_HALF], dtype=pl.FP32, value=1.0)
        cos_b = pl.col_expand_mul(rope_row_ones, cos[0:1, 0:ROPE_HALF])
        sin_b = pl.col_expand_mul(rope_row_ones, sin[0:1, 0:ROPE_HALF])
        partial_sq = pl.full([1, RMS_TILE], dtype=pl.FP32, value=0.0)
        for rms_kb in pl.pipeline(HEAD_BLOCKS, stage=2):
            rms_h0 = rms_kb * HEAD_CHUNK
            kv_rms_chunk = pooled_kv_pad[batch_base : batch_base + RMS_TILE, rms_h0 : rms_h0 + HEAD_CHUNK]
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            kv_rms_rowsum = pl.reshape(pl.row_sum(kv_rms_sq), [1, RMS_TILE])
            partial_sq = pl.add(partial_sq, kv_rms_rowsum)

        variance = pl.reshape(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS), [RMS_TILE, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        for rms_kb in pl.pipeline(NOPE_HEAD_DIM // HEAD_CHUNK, stage=2):
            norm_h0 = rms_kb * HEAD_CHUNK
            kv_norm_chunk = pooled_kv_pad[batch_base : batch_base + RMS_TILE, norm_h0 : norm_h0 + HEAD_CHUNK]
            gamma = norm_w_2d[:, norm_h0 : norm_h0 + HEAD_CHUNK]
            normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
            normed_chunk_bf16 = pl.cast(
                normed_chunk,
                target_type=pl.BF16,
                mode="rint",
            )
            normed_kv_pad[batch_base : batch_base + RMS_TILE, norm_h0 : norm_h0 + HEAD_CHUNK] = pl.cast(
                normed_chunk_bf16,
                target_type=pl.FP32,
            )

        kv_rope_norm = pooled_kv_pad[batch_base : batch_base + RMS_TILE, NOPE_HEAD_DIM:HEAD_DIM]
        gamma_rope = norm_w_2d[:, NOPE_HEAD_DIM:HEAD_DIM]
        rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
        rope_normed_bf16 = pl.cast(rope_normed, target_type=pl.BF16, mode="rint")
        rope_normed_fp32 = pl.cast(rope_normed_bf16, target_type=pl.FP32)
        rope_even = pl.gather(rope_normed_fp32, mask_pattern=pl.tile.MaskPattern.P0101)
        rope_odd = pl.gather(rope_normed_fp32, mask_pattern=pl.tile.MaskPattern.P1010)
        rope_rot_even = pl.sub(pl.mul(rope_even, cos_b), pl.mul(rope_odd, sin_b))
        rope_rot_odd = pl.add(pl.mul(rope_even, sin_b), pl.mul(rope_odd, cos_b))
        rope_even_bf16 = pl.cast(rope_rot_even, target_type=pl.BF16, mode="rint")
        rope_odd_bf16 = pl.cast(rope_rot_odd, target_type=pl.BF16, mode="rint")
        idx_target = pl.full([RMS_TILE, ROPE_HALF], dtype=pl.INT32, value=0)
        even_idx_full = pl.col_expand(idx_target, even_idx)
        odd_idx_full = pl.col_expand(idx_target, odd_idx)
        rope_buf = pl.full([RMS_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        rope_buf = pl.tensor.scatter(rope_buf, dim=-1, index=even_idx_full, src=pl.cast(rope_even_bf16, target_type=pl.FP32))
        rope_buf = pl.tensor.scatter(rope_buf, dim=-1, index=odd_idx_full, src=pl.cast(rope_odd_bf16, target_type=pl.FP32))
        normed_kv_pad[batch_base : batch_base + RMS_TILE, NOPE_HEAD_DIM:HEAD_DIM] = rope_buf

    for final_b in pl.parallel(0, B, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_c128_finalize"):
            kv_row = final_b * PREFILL_COMPRESSED_LEN
            cache_row = final_b * IDX_KV_LEN + pl.cast(start_pos // COMPRESS_RATIO, pl.INDEX)
            for hb in pl.range(HEAD_BLOCKS):
                final_h0 = hb * HEAD_CHUNK
                final_chunk = normed_kv_pad[final_b : final_b + 1, final_h0 : final_h0 + HEAD_CHUNK]
                final_bf16 = pl.cast(final_chunk, target_type=pl.BF16, mode="rint")
                kv_flat[kv_row : kv_row + 1, final_h0 : final_h0 + HEAD_CHUNK] = pl.cast(final_bf16, target_type=pl.FP32)
                kv_cache_flat[cache_row : cache_row + 1, final_h0 : final_h0 + HEAD_CHUNK] = final_bf16

    kv = pl.reshape(kv_flat, [B, PREFILL_COMPRESSED_LEN, HEAD_DIM])
    kv_cache = pl.reshape(kv_cache_flat, [B, IDX_KV_LEN, HEAD_DIM])
    kv_state = pl.reshape(kv_state_flat, [B, STATE_LEN, OUT_DIM])
    score_state = pl.reshape(score_state_flat, [B, STATE_LEN, OUT_DIM])
    return kv, kv_state, score_state, kv_cache


@pl.jit
def prefill_compressor_ratio128_test(
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
    even_idx: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.INT32],
    odd_idx: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.INT32],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    kv_cache: pl.Out[pl.Tensor[[B, IDX_KV_LEN, HEAD_DIM], pl.BF16]],
    start_pos: pl.Scalar[pl.INT32],
    rotate: pl.Scalar[pl.BOOL],
):
    return prefill_compressor_ratio128(
        x,
        kv,
        kv_state,
        score_state,
        wkv,
        wgate,
        ape,
        norm_w,
        cos,
        sin,
        even_idx,
        odd_idx,
        kv_cache,
        start_pos,
    )


def golden_prefill_compressor_ratio128(tensors):
    import torch

    start_pos = int(tensors["start_pos"])
    if start_pos % COMPRESS_RATIO != 0:
        raise ValueError("prefill_compressor_ratio128 expects start_pos aligned to COMPRESS_RATIO")
    cache_slot = start_pos // COMPRESS_RATIO
    if cache_slot >= IDX_KV_LEN:
        raise ValueError("prefill_compressor_ratio128 start_pos exceeds kv_cache length")

    kv_proj = tensors["x"].float() @ tensors["wkv"].float()
    score_proj = tensors["x"].float() @ tensors["wgate"].float()
    score_proj = score_proj + tensors["ape"][:S].view(1, S, OUT_DIM)

    tensors["kv_state"][:, :S, :] = kv_proj
    tensors["score_state"][:, :S, :] = score_proj

    pooled = (kv_proj * score_proj.softmax(dim=1)).sum(dim=1, keepdim=True).to(torch.bfloat16).float()
    norm_w = tensors["norm_w"].float()
    inv = torch.rsqrt(pooled.square().mean(dim=-1, keepdim=True) + EPS)
    kv_norm = (pooled * inv * norm_w.view(1, 1, HEAD_DIM)).to(torch.bfloat16)
    rope = kv_norm[..., NOPE_HEAD_DIM:].float()
    rope_pair = rope.unflatten(-1, (-1, 2))
    rope_even = rope_pair[..., 0]
    rope_odd = rope_pair[..., 1]
    cos = tensors["cos"].float().view(1, PREFILL_COMPRESSED_LEN, ROPE_HALF)
    sin = tensors["sin"].float().view(1, PREFILL_COMPRESSED_LEN, ROPE_HALF)
    rot_even = (rope_even * cos - rope_odd * sin).to(torch.bfloat16)
    rot_odd = (rope_even * sin + rope_odd * cos).to(torch.bfloat16)
    rope_full = torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)
    kv_bf16 = torch.cat([kv_norm[..., :NOPE_HEAD_DIM], rope_full], dim=-1).to(torch.bfloat16)
    tensors["kv"][:, 0:1, :] = kv_bf16.float()
    tensors["kv_cache"][:, cache_slot : cache_slot + 1, :] = kv_bf16


def build_tensor_specs(start_pos: int = START_POS):
    import torch
    from golden import ScalarSpec, TensorSpec

    def seeded_uniform(shape, seed, scale=1.0):
        generator = torch.Generator()
        generator.manual_seed(seed)
        return (torch.rand(*shape, generator=generator) - 0.5) * scale

    def init_x():
        return seeded_uniform((B, S, D), 1, 0.1)
    def init_kv():
        return torch.zeros(B, PREFILL_COMPRESSED_LEN, HEAD_DIM)
    def init_kv_state():
        return torch.zeros(B, STATE_LEN, OUT_DIM)
    def init_wkv():
        return seeded_uniform((D, OUT_DIM), 2, D ** -0.5)
    def init_wgate():
        return seeded_uniform((D, OUT_DIM), 3, D ** -0.5)
    def init_ape():
        return seeded_uniform((COMPRESS_RATIO, OUT_DIM), 4, 0.1)
    def init_norm_w():
        return torch.ones(HEAD_DIM)
    def init_cos():
        return torch.cos(torch.arange(PREFILL_COMPRESSED_LEN * ROPE_HALF).reshape(PREFILL_COMPRESSED_LEN, ROPE_HALF) * 1e-3)
    def init_sin():
        return torch.sin(torch.arange(PREFILL_COMPRESSED_LEN * ROPE_HALF).reshape(PREFILL_COMPRESSED_LEN, ROPE_HALF) * 1e-3)
    def init_even_idx():
        return torch.arange(0, ROPE_HEAD_DIM, 2, dtype=torch.int32).unsqueeze(0)
    def init_odd_idx():
        return torch.arange(1, ROPE_HEAD_DIM, 2, dtype=torch.int32).unsqueeze(0)
    def init_hadamard():
        return torch.zeros(HEAD_DIM, HEAD_DIM)
    def init_kv_cache():
        return torch.zeros(B, IDX_KV_LEN, HEAD_DIM)

    return [
        TensorSpec("x", [B, S, D], torch.bfloat16, init_value=init_x),
        TensorSpec("kv", [B, PREFILL_COMPRESSED_LEN, HEAD_DIM], torch.float32, init_value=init_kv, is_output=True),
        TensorSpec("kv_state", [B, STATE_LEN, OUT_DIM], torch.float32, init_value=init_kv_state, is_output=True),
        TensorSpec("score_state", [B, STATE_LEN, OUT_DIM], torch.float32, init_value=init_kv_state, is_output=True),
        TensorSpec("wkv", [D, OUT_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [D, OUT_DIM], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.float32, init_value=init_norm_w),
        TensorSpec("cos", [PREFILL_COMPRESSED_LEN, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_cos),
        TensorSpec("sin", [PREFILL_COMPRESSED_LEN, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_sin),
        TensorSpec("even_idx", [1, ROPE_HEAD_DIM // 2], torch.int32, init_value=init_even_idx),
        TensorSpec("odd_idx", [1, ROPE_HEAD_DIM // 2], torch.int32, init_value=init_odd_idx),
        TensorSpec("hadamard", [HEAD_DIM, HEAD_DIM], torch.bfloat16, init_value=init_hadamard),
        TensorSpec("kv_cache", [B, IDX_KV_LEN, HEAD_DIM], torch.bfloat16, init_value=init_kv_cache, is_output=True),
        ScalarSpec("start_pos", torch.int32, start_pos),
        ScalarSpec("rotate", torch.bool, ROTATE),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--start-pos", type=int, default=START_POS)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=prefill_compressor_ratio128_test,
        specs=build_tensor_specs(args.start_pos),
        golden_fn=golden_prefill_compressor_ratio128,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
            "kv_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "score_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
