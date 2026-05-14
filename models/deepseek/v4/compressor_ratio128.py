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

from config import DEMO as M, DECODE_BATCH, DECODE_SEQ

# model config
B = DECODE_BATCH
S = DECODE_SEQ
EPS = M.rms_norm_eps
D = M.hidden_size
HEAD_DIM = M.head_dim
HEAD_DIM_INV = 1.0 / HEAD_DIM
ROPE_HEAD_DIM = M.qk_rope_head_dim
NOPE_HEAD_DIM = M.nope_head_dim

# kernel-local (ratio-128 non-overlap compressor)
COMPRESS_RATIO = 128
MAX_SEQ_LEN = 4096
IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
FP32_NEG_INF = -3.4028234663852886e38

COFF = 1
OUT_DIM = COFF * HEAD_DIM          # 512
STATE_LEN = COFF * COMPRESS_RATIO  # 128
START_POS = 127      # ScalarSpec default; (START_POS+1)%COMPRESS_RATIO==0

# tiling
ROPE_CHUNK = 32
K_CHUNK = 512
OUT_CHUNK = 128

HEAD_CHUNK = 128
K_BLOCKS = D // K_CHUNK            # 8
OUT_BLOCKS = OUT_DIM // OUT_CHUNK  # 4
HEAD_BLOCKS = HEAD_DIM // HEAD_CHUNK  # 4


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
    kv_proj = pl.create_tensor([B * S, OUT_DIM], dtype=pl.FP32)
    score_proj = pl.create_tensor([B * S, OUT_DIM], dtype=pl.FP32)
    ape_row = pl.cast(start_pos % COMPRESS_RATIO, target_type=pl.INDEX)
    compress_rem = (start_pos + 1) % COMPRESS_RATIO
    score_ape = pl.create_tensor([B * S, OUT_DIM], dtype=pl.FP32)
    # Non-overlap: scatter into slot = ape_row
    state_col0 = ape_row * OUT_DIM
    kv_state_flat = pl.reshape(kv_state, [B, STATE_LEN * OUT_DIM])
    score_state_flat = pl.reshape(score_state, [B, STATE_LEN * OUT_DIM])
    for o0 in pl.parallel(0, OUT_DIM, OUT_CHUNK):
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

            kv_proj = pl.assemble(kv_proj, kv_acc, [0, o0])
            score_proj = pl.assemble(score_proj, score_acc, [0, o0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="score_ape"):
            score_tile = score_proj[:, o0 : o0 + OUT_CHUNK]
            ape_tile = ape[ape_row : ape_row + 1, o0 : o0 + OUT_CHUNK]
            ape_base = pl.full([B * S, OUT_CHUNK], dtype=pl.FP32, value=0.0)
            score_tile = pl.add(score_tile, pl.col_expand(ape_base, ape_tile))
            score_ape = pl.assemble(score_ape, score_tile, [0, o0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_scatter"):
            kv_tile = kv_proj[:, o0 : o0 + OUT_CHUNK]
            score_tile = score_ape[:, o0 : o0 + OUT_CHUNK]
            kv_state_flat = pl.assemble(kv_state_flat, kv_tile, [0, state_col0 + o0])
            score_state_flat = pl.assemble(score_state_flat, score_tile, [0, state_col0 + o0])

    pooled_kv = pl.create_tensor([B * S, HEAD_DIM], dtype=pl.FP32)
    normed_kv = pl.create_tensor([B * S, HEAD_DIM], dtype=pl.BF16)
    kv_flat = pl.reshape(kv, [B * S, HEAD_DIM])

    if compress_rem == 0:
        for hb in pl.parallel(0, HEAD_BLOCKS, 1):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="softmax_pool"):
                h0 = hb * HEAD_CHUNK
                # Initialize m/l/o from last slot
                last_col0 = (STATE_LEN - 1) * OUT_DIM + h0
                mi = score_state_flat[:, last_col0 : last_col0 + HEAD_CHUNK]
                li = pl.exp(pl.sub(mi, mi))
                oi = kv_state_flat[:, last_col0 : last_col0 + HEAD_CHUNK]

                # Online softmax over all remaining slots
                for s in pl.range(0, STATE_LEN - 1):
                    col0 = s * OUT_DIM + h0
                    slot_score = score_state_flat[:, col0 : col0 + HEAD_CHUNK]
                    slot_kv = kv_state_flat[:, col0 : col0 + HEAD_CHUNK]
                    mi_next = pl.maximum(mi, slot_score)
                    alpha = pl.exp(pl.sub(mi, mi_next))
                    beta = pl.exp(pl.sub(slot_score, mi_next))
                    li = pl.add(pl.mul(alpha, li), beta)
                    oi = pl.add(pl.mul(oi, alpha), pl.mul(slot_kv, beta))
                    mi = mi_next

                pooled_chunk = pl.div(oi, li)
                pooled_kv = pl.assemble(pooled_kv, pooled_chunk, [0, h0])

        # No state shift for non-overlap

        # RMSNorm with BF16 intermediate
        norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm"):
            partial_sq = pl.full([1, B * S], dtype=pl.FP32, value=0.0)
            for k0 in pl.range(0, HEAD_DIM, HEAD_CHUNK):
                kv_rms_chunk = pl.cast(
                    pl.cast(pooled_kv[:, k0 : k0 + HEAD_CHUNK], target_type=pl.BF16),
                    target_type=pl.FP32,
                )
                partial_sq = pl.add(
                    partial_sq,
                    pl.reshape(pl.row_sum(pl.mul(kv_rms_chunk, kv_rms_chunk)), [1, B * S]),
                )

            variance = pl.reshape(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS), [B * S, 1])
            inv_rms = pl.recip(pl.sqrt(variance))
            for k0 in pl.range(0, HEAD_DIM, HEAD_CHUNK):
                kv_norm_chunk = pl.cast(
                    pl.cast(pooled_kv[:, k0 : k0 + HEAD_CHUNK], target_type=pl.BF16),
                    target_type=pl.FP32,
                )
                gamma = norm_w_2d[:, k0 : k0 + HEAD_CHUNK]
                normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
                normed_kv = pl.assemble(normed_kv, pl.cast(normed_chunk, target_type=pl.BF16), [0, k0])

        # Selector-based RoPE
        kv_rope = pl.create_tensor([B * S, ROPE_HEAD_DIM], dtype=pl.BF16)
        kv_proj_even = pl.create_tensor([B * S, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
        kv_proj_odd = pl.create_tensor([B * S, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
        rope_even = pl.create_tensor([B * S, ROPE_HEAD_DIM // 2], dtype=pl.BF16)
        rope_odd = pl.create_tensor([B * S, ROPE_HEAD_DIM // 2], dtype=pl.BF16)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_rope_slice"):
            kv_rope = pl.assemble(kv_rope, normed_kv[:, NOPE_HEAD_DIM : HEAD_DIM], [0, 0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_slice"):
            kv_rope_tile = kv_rope[:, 0 : ROPE_CHUNK]
            even_select_tile = even_select[0 : ROPE_CHUNK, :]
            odd_select_tile = odd_select[0 : ROPE_CHUNK, :]
            even_acc = pl.matmul(kv_rope_tile, even_select_tile, out_dtype=pl.FP32)
            odd_acc = pl.matmul(kv_rope_tile, odd_select_tile, out_dtype=pl.FP32)

            for r0 in pl.range(ROPE_CHUNK, ROPE_HEAD_DIM, ROPE_CHUNK):
                kv_rope_tile = kv_rope[:, r0 : r0 + ROPE_CHUNK]
                even_select_tile = even_select[r0 : r0 + ROPE_CHUNK, :]
                odd_select_tile = odd_select[r0 : r0 + ROPE_CHUNK, :]
                even_acc = pl.matmul_acc(even_acc, kv_rope_tile, even_select_tile)
                odd_acc = pl.matmul_acc(odd_acc, kv_rope_tile, odd_select_tile)
            kv_proj_even = pl.assemble(kv_proj_even, even_acc, [0, 0])
            kv_proj_odd = pl.assemble(kv_proj_odd, odd_acc, [0, 0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_apply"):
            even_tile = kv_proj_even[:, :]
            odd_tile = kv_proj_odd[:, :]
            rope_even_acc = pl.cast(pl.sub(pl.col_expand_mul(even_tile, cos), pl.col_expand_mul(odd_tile, sin)), target_type=pl.BF16)
            rope_odd_acc = pl.cast(pl.add(pl.col_expand_mul(even_tile, sin), pl.col_expand_mul(odd_tile, cos)), target_type=pl.BF16)
            rope_even = pl.assemble(rope_even, rope_even_acc, [0, 0])
            rope_odd = pl.assemble(rope_odd, rope_odd_acc, [0, 0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_assemble"):
            rope_even_tile = rope_even[:, 0 : ROPE_CHUNK]
            rope_odd_tile = rope_odd[:, 0 : ROPE_CHUNK]
            even_select_tile_t = even_select[:, 0 : ROPE_CHUNK]
            odd_select_tile_t = odd_select[:, 0 : ROPE_CHUNK]
            rope_acc = pl.matmul(rope_even_tile, even_select_tile_t, out_dtype=pl.FP32, b_trans=True)
            rope_acc = pl.matmul_acc(rope_acc, rope_odd_tile, odd_select_tile_t, b_trans=True)

            for r0 in pl.range(ROPE_CHUNK, ROPE_HEAD_DIM // 2, ROPE_CHUNK):
                rope_even_tile = rope_even[:, r0 : r0 + ROPE_CHUNK]
                rope_odd_tile = rope_odd[:, r0 : r0 + ROPE_CHUNK]
                even_select_tile_t = even_select[:, r0 : r0 + ROPE_CHUNK]
                odd_select_tile_t = odd_select[:, r0 : r0 + ROPE_CHUNK]
                rope_acc = pl.matmul_acc(rope_acc, rope_even_tile, even_select_tile_t, b_trans=True)
                rope_acc = pl.matmul_acc(rope_acc, rope_odd_tile, odd_select_tile_t, b_trans=True)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_write"):
            normed_kv = pl.assemble(normed_kv, pl.cast(rope_acc, target_type=pl.BF16), [0, NOPE_HEAD_DIM])

        if rotate:
            for o0 in pl.range(0, HEAD_DIM, OUT_CHUNK):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_hadamard"):
                    kv_proj_tile = normed_kv[:, 0 : HEAD_DIM]
                    hadamard_tile = hadamard[0 : HEAD_DIM, o0 : o0 + OUT_CHUNK]
                    kv_hadamard_acc = pl.matmul(kv_proj_tile, hadamard_tile, out_dtype=pl.FP32)
                    kv_flat = pl.assemble(kv_flat, kv_hadamard_acc, [0, o0])
        else:
            for o0 in pl.parallel(0, HEAD_DIM, OUT_CHUNK):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_write"):
                    kv_out_tile = normed_kv[:, o0 : o0 + OUT_CHUNK]
                    kv_flat = pl.assemble(kv_flat, pl.cast(kv_out_tile, target_type=pl.FP32), [0, o0])

        kv_cache_flat = pl.reshape(kv_cache, [B * IDX_KV_LEN, HEAD_DIM])
        cache_col = start_pos // COMPRESS_RATIO
        for b_idx in pl.parallel(B):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_cache_write"):
                cache_row = b_idx * IDX_KV_LEN + cache_col
                cache_kv_row = kv_flat[b_idx : b_idx + 1, 0 : HEAD_DIM]
                kv_cache_flat = pl.assemble(kv_cache_flat, pl.cast(cache_kv_row, target_type=pl.BF16), [cache_row, 0])
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

    kv = x @ wkv
    score = x @ wgate

    should_compress = (start_pos + 1) % ratio == 0
    score = score + ape[start_pos % ratio]

    # Non-overlap: scatter into slot start_pos % ratio
    kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
    score_state[:bsz, start_pos % ratio] = score.squeeze(1)
    if should_compress:
        kv = (kv_state[:bsz] * score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        # No shift for non-overlap

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
    tensors["kv"][:] = kv

    kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)

    tensors["kv_cache"][:] = kv_cache


def build_tensor_specs():
    import torch  # type: ignore[import]
    from golden import ScalarSpec, TensorSpec

    def init_x():
        return torch.randn(B, S, D) * 0.1
    def init_kv_state():
        return torch.zeros(B, STATE_LEN, OUT_DIM)
    def init_score_state():
        return torch.full((B, STATE_LEN, OUT_DIM), FP32_NEG_INF)
    def init_wkv():
        return torch.randn(D, OUT_DIM) / D ** 0.5
    def init_wgate():
        return torch.randn(D, OUT_DIM) / D ** 0.5
    def init_ape():
        return torch.randn(COMPRESS_RATIO, OUT_DIM) * 0.1
    def init_norm_w():
        return torch.ones(HEAD_DIM)
    def init_cos():
        return torch.cos(torch.arange(ROPE_HEAD_DIM // 2).reshape(1, ROPE_HEAD_DIM // 2) * 1e-3)
    def init_sin():
        return torch.sin(torch.arange(ROPE_HEAD_DIM // 2).reshape(1, ROPE_HEAD_DIM // 2) * 1e-3)
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
        H = torch.ones((1, 1))
        while H.shape[0] < HEAD_DIM:
            H = torch.cat([
                torch.cat([H,  H], dim=1),
                torch.cat([H, -H], dim=1),
            ], dim=0)
        return H / (HEAD_DIM ** 0.5)
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
        ScalarSpec("start_pos", torch.int32, START_POS),
        ScalarSpec("rotate", torch.bool, True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import RunConfig, bf16_allclose_or_ulp, ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=compressor_test,
        specs=build_tensor_specs(),
        golden_fn=golden_compressor,
        config=RunConfig(
            rtol=1e-3,
            atol=1e-3,
            compile=dict(dump_passes=True),
            runtime=dict(
                platform=args.platform,
                device_id=args.device,
                enable_l2_swimlane=args.enable_l2_swimlane,
            ),
            compare_fn={
                "kv":          ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
                "kv_state":    ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
                "score_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
                "kv_cache":    bf16_allclose_or_ulp(),
            },
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
