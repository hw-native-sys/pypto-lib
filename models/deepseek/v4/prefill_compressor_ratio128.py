# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 prefill compressor, ratio=128 bring-up.

This standalone module mirrors the temporary ratio-128 compressor used by
prefill_attention_hca.py: project KV/score, add APE, softmax-pool the
128-token prompt chunk, and publish one compressed KV row. It intentionally
keeps the current HCA golden contract and does not add unverified compressor
RMS/RoPE semantics yet.
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
        h0 = hb * HEAD_CHUNK
        t0 = pool_b * S
        score_tile = score_proj[t0 : t0 + S, h0 : h0 + HEAD_CHUNK]
        kv_tile = kv_proj[t0 : t0 + S, h0 : h0 + HEAD_CHUNK]
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
        cache_row = pool_b * IDX_KV_LEN + pl.cast(start_pos // COMPRESS_RATIO, pl.INDEX)
        kv_flat[kv_row : kv_row + 1, h0 : h0 + HEAD_CHUNK] = pl.cast(pooled_bf16, target_type=pl.FP32)
        kv_cache_flat[cache_row : cache_row + 1, h0 : h0 + HEAD_CHUNK] = pooled_bf16

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
    even_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    odd_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
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

    pooled = (kv_proj * score_proj.softmax(dim=1)).sum(dim=1, keepdim=True)
    kv_bf16 = pooled.to(torch.bfloat16)
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
        return torch.zeros(PREFILL_COMPRESSED_LEN, ROPE_HEAD_DIM // 2)
    def init_sin():
        return torch.zeros(PREFILL_COMPRESSED_LEN, ROPE_HEAD_DIM // 2)
    def init_even_select():
        matrix = torch.zeros((ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2))
        for i in range(ROPE_HEAD_DIM // 2):
            matrix[2 * i, i] = 1
        return matrix
    def init_odd_select():
        matrix = torch.zeros((ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2))
        for i in range(ROPE_HEAD_DIM // 2):
            matrix[2 * i + 1, i] = 1
        return matrix
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
        TensorSpec("even_select", [ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], torch.bfloat16, init_value=init_even_select),
        TensorSpec("odd_select", [ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], torch.bfloat16, init_value=init_odd_select),
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
