# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 KV Compressor (decode incremental, ratio=128 non-overlap).

Uses non-overlapping state layout with 128 slots. Loop-based softmax+pool
over all slots. No state shift needed."""

import pypto.language as pl

B = 16
S = 1
EPS = 1e-6

COMPRESS_RATIO = 128
HEAD_DIM = 512
ROTATE = False

D = 4096
ROPE_HEAD_DIM = 64
NOPE_HEAD_DIM = HEAD_DIM - ROPE_HEAD_DIM
OVERLAP = False
COFF = 1

OUT_DIM = COFF * HEAD_DIM   # 512
STATE_LEN = COFF * COMPRESS_RATIO  # 128

START_POS = 127
SHOULD_COMPRESS = COMPRESS_RATIO != 0 and ((START_POS + 1) % COMPRESS_RATIO) == 0
APE_ROW = START_POS % COMPRESS_RATIO  # 127
SCATTER_SLOT = APE_ROW  # 127 (no overlap)

HEAD_DIM_INV = 1.0 / HEAD_DIM

K_CHUNK = 512
OUT_CHUNK = 64
K_BLOCKS = D // K_CHUNK  # 8
OUT_BLOCKS = OUT_DIM // OUT_CHUNK  # 8

HEAD_CHUNK = 128
HEAD_BLOCKS = HEAD_DIM // HEAD_CHUNK  # 4


@pl.jit.inline
def compressor(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv_state: pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32],
    score_state: pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.BF16],
    sin: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.BF16],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    start_pos: pl.Scalar[pl.INT32],
    out: pl.Tensor[[B, HEAD_DIM], pl.BF16],
):
    x_flat = pl.reshape(x, [B, D])
    kv_state_flat = pl.reshape(kv_state, [B, STATE_LEN * OUT_DIM])
    score_state_flat = pl.reshape(score_state, [B, STATE_LEN * OUT_DIM])

    kv_fp32 = pl.create_tensor([B, OUT_DIM], dtype=pl.FP32)
    score_fp32 = pl.create_tensor([B, OUT_DIM], dtype=pl.FP32)
    slot_off = SCATTER_SLOT * OUT_DIM

    for ob in pl.parallel(0, OUT_BLOCKS, 1):
        oc0 = ob * OUT_CHUNK
        # Block 1a (Cube): kv = x @ wkv.T
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_proj"):
            a0 = pl.slice(x_flat, [B, K_CHUNK], [0, 0])
            b0 = pl.slice(wkv, [OUT_CHUNK, K_CHUNK], [oc0, 0])
            kv_acc = pl.matmul(a0, b0, out_dtype=pl.FP32, b_trans=True)
            for kb in pl.range(1, K_BLOCKS):
                a_i = pl.slice(x_flat, [B, K_CHUNK], [0, kb * K_CHUNK])
                b_i = pl.slice(wkv, [OUT_CHUNK, K_CHUNK], [oc0, kb * K_CHUNK])
                kv_acc = pl.matmul_acc(kv_acc, a_i, b_i, b_trans=True)
            kv_fp32 = pl.assemble(kv_fp32, kv_acc, [0, oc0])

        # Block 1b (Cube): score = x @ wgate.T
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="score_proj"):
            a0g = pl.slice(x_flat, [B, K_CHUNK], [0, 0])
            b0g = pl.slice(wgate, [OUT_CHUNK, K_CHUNK], [oc0, 0])
            sc_acc = pl.matmul(a0g, b0g, out_dtype=pl.FP32, b_trans=True)
            for kb in pl.range(1, K_BLOCKS):
                a_ig = pl.slice(x_flat, [B, K_CHUNK], [0, kb * K_CHUNK])
                b_ig = pl.slice(wgate, [OUT_CHUNK, K_CHUNK], [oc0, kb * K_CHUNK])
                sc_acc = pl.matmul_acc(sc_acc, a_ig, b_ig, b_trans=True)
            score_fp32 = pl.assemble(score_fp32, sc_acc, [0, oc0])

        # Block 2 (Vector): score += ape[APE_ROW]
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="ape_add"):
            sc = pl.slice(score_fp32, [B, OUT_CHUNK], [0, oc0])
            ape_row = pl.slice(ape, [1, OUT_CHUNK], [APE_ROW, oc0])
            ones_b = pl.full([B, OUT_CHUNK], dtype=pl.FP32, value=1.0)
            ape_broadcast = pl.col_expand_mul(ones_b, ape_row)
            sc = pl.add(sc, ape_broadcast)
            score_fp32 = pl.assemble(score_fp32, sc, [0, oc0])

        # Block 3 (Vector): scatter current kv/score into state
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_scatter"):
            kv_chunk = pl.slice(kv_fp32, [B, OUT_CHUNK], [0, oc0])
            kv_state_flat = pl.assemble(kv_state_flat, kv_chunk, [0, slot_off + oc0])
            sc_chunk = pl.slice(score_fp32, [B, OUT_CHUNK], [0, oc0])
            score_state_flat = pl.assemble(score_state_flat, sc_chunk, [0, slot_off + oc0])

    # Reshape state to per-state-row 2D views
    kv_state_per_row = pl.reshape(kv_state_flat, [B * STATE_LEN, OUT_DIM])
    score_state_per_row = pl.reshape(score_state_flat, [B * STATE_LEN, OUT_DIM])

    # Block 5+6 (Vector): softmax+pool over 128 slots via loop.
    pooled = pl.create_tensor([B, HEAD_DIM], dtype=pl.FP32)
    for b_idx in pl.parallel(0, B, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="softmax_pool_loop"):
            row_b = b_idx * STATE_LEN
            for hb in pl.range(HEAD_BLOCKS):
                h0 = hb * HEAD_CHUNK
                # Max via loop over all 128 slots.
                s_max = pl.slice(score_state_per_row, [1, HEAD_CHUNK], [row_b, h0])
                for s in pl.range(1, STATE_LEN):
                    sr_max = pl.slice(score_state_per_row, [1, HEAD_CHUNK], [row_b + s, h0])
                    s_max = pl.maximum(s_max, sr_max)

                # Exp sum and weighted sum via loop.
                e_sum = pl.full([1, HEAD_CHUNK], dtype=pl.FP32, value=0.0)
                weighted = pl.full([1, HEAD_CHUNK], dtype=pl.FP32, value=0.0)
                for s in pl.range(STATE_LEN):
                    sr_exp = pl.slice(score_state_per_row, [1, HEAD_CHUNK], [row_b + s, h0])
                    kv_row = pl.slice(kv_state_per_row, [1, HEAD_CHUNK], [row_b + s, h0])
                    e_row = pl.exp(pl.sub(sr_exp, s_max))
                    e_sum = pl.add(e_sum, e_row)
                    weighted = pl.add(weighted, pl.mul(e_row, kv_row))

                pooled_chunk = pl.div(weighted, e_sum)
                pooled = pl.assemble(pooled, pooled_chunk, [b_idx, h0])

    # No block 7 (no shift in non-overlap mode).

    # Reshape state back to 3D
    kv_state = pl.reshape(kv_state_per_row, [B, STATE_LEN, OUT_DIM])
    score_state = pl.reshape(score_state_per_row, [B, STATE_LEN, OUT_DIM])

    # Block 8 (Vector): RMSNorm pooled with norm_w over HEAD_DIM.
    normed_pooled = pl.create_tensor([B, HEAD_DIM], dtype=pl.FP32)
    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm"):
        partial_sq = pl.full([1, B], dtype=pl.FP32, value=0.0)
        for hb in pl.range(HEAD_BLOCKS):
            h0 = hb * HEAD_CHUNK
            pc = pl.slice(pooled, [B, HEAD_CHUNK], [0, h0])
            partial_sq = pl.add(
                partial_sq,
                pl.reshape(pl.row_sum(pl.mul(pc, pc)), [1, B]),
            )
        inv_rms = pl.reshape(
            pl.recip(pl.sqrt(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS))),
            [B, 1],
        )
        for hb in pl.range(HEAD_BLOCKS):
            h0 = hb * HEAD_CHUNK
            nc = pl.slice(pooled, [B, HEAD_CHUNK], [0, h0])
            nw_chunk = pl.cast(
                pl.slice(norm_w_2d, [1, HEAD_CHUNK], [0, h0]),
                target_type=pl.FP32,
            )
            normed = pl.col_expand_mul(pl.row_expand_mul(nc, inv_rms), nw_chunk)
            normed_pooled = pl.assemble(normed_pooled, normed, [0, h0])

    # Block 11a (Vector): cast non-rope range to BF16 and store to out.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="store_nope"):
        nope_chunk = pl.slice(normed_pooled, [B, NOPE_HEAD_DIM], [0, 0])
        out = pl.assemble(out, pl.cast(nope_chunk, target_type=pl.BF16), [0, 0])

    # Block 9 + 11b (Vector): half-vector RoPE on the last ROPE_HEAD_DIM cols, then store.
    HALF_RD = ROPE_HEAD_DIM // 2
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_store"):
        x_lo = pl.slice(normed_pooled, [B, HALF_RD], [0, NOPE_HEAD_DIM])
        x_hi = pl.slice(normed_pooled, [B, HALF_RD], [0, NOPE_HEAD_DIM + HALF_RD])
        cos_fp32 = pl.cast(cos, target_type=pl.FP32)
        sin_fp32 = pl.cast(sin, target_type=pl.FP32)
        y_lo = pl.sub(pl.col_expand_mul(x_lo, cos_fp32), pl.col_expand_mul(x_hi, sin_fp32))
        y_hi = pl.add(pl.col_expand_mul(x_lo, sin_fp32), pl.col_expand_mul(x_hi, cos_fp32))
        out = pl.assemble(out, pl.cast(y_lo, target_type=pl.BF16), [0, NOPE_HEAD_DIM])
        out = pl.assemble(out, pl.cast(y_hi, target_type=pl.BF16), [0, NOPE_HEAD_DIM + HALF_RD])

    return out


@pl.jit
def compressor_test(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv_state: pl.Out[pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32]],
    score_state: pl.Out[pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32]],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.BF16],
    sin: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.BF16],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    start_pos: pl.Scalar[pl.INT32],
    out: pl.Out[pl.Tensor[[B, HEAD_DIM], pl.BF16]],
):
    out = compressor(
        x, kv_state, score_state, wkv, wgate, ape, norm_w, cos, sin, hadamard, start_pos, out,
    )
    return out


def golden_compressor(tensors):
    """Torch reference for Compressor.forward (decode branch, ratio=128)."""
    import torch

    x = tensors["x"]
    kv_state = tensors["kv_state"]
    score_state = tensors["score_state"]
    wkv = tensors["wkv"].float()
    wgate = tensors["wgate"].float()
    ape = tensors["ape"].float()
    norm_w = tensors["norm_w"].float()
    cos = tensors["cos"].float()
    sin = tensors["sin"].float()
    hadamard = tensors["hadamard"].float()

    bsz, _, _ = x.shape
    ratio, overlap, rotate, d, rd = COMPRESS_RATIO, OVERLAP, ROTATE, HEAD_DIM, ROPE_HEAD_DIM
    dtype = x.dtype
    x = x.float()
    kv = x.view(bsz, -1) @ wkv.T
    score = x.view(bsz, -1) @ wgate.T

    should_compress = (START_POS + 1) % ratio == 0
    score = score + ape[START_POS % ratio]
    # Non-overlap path
    kv_state[:bsz, START_POS % ratio] = kv
    score_state[:bsz, START_POS % ratio] = score
    if should_compress:
        kv = (kv_state[:bsz] * score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)

    if not should_compress:
        tensors["out"][:] = torch.zeros(B, HEAD_DIM, dtype=torch.bfloat16)
        return

    kv_c = kv.squeeze(1)
    kv_c = kv_c * torch.rsqrt(kv_c.square().mean(-1, keepdim=True) + EPS) * norm_w

    half_rd = rd // 2
    x_lo = kv_c[..., -rd:-half_rd]
    x_hi = kv_c[..., -half_rd:]
    cos_v, sin_v = cos.view(-1), sin.view(-1)
    y_lo = x_lo * cos_v - x_hi * sin_v
    y_hi = x_lo * sin_v + x_hi * cos_v
    kv_c = torch.cat([kv_c[..., :-rd], y_lo, y_hi], dim=-1)

    if rotate:
        kv_c = (kv_c @ hadamard).to(torch.bfloat16).float()
    else:
        pass

    tensors["out"][:] = kv_c.to(torch.bfloat16)


def build_tensor_specs():
    import torch  # type: ignore[import]
    from golden import ScalarSpec, TensorSpec

    torch.manual_seed(42)

    def init_x():
        return torch.randn(B, S, D) - 0.5
    def init_kv_state():
        return torch.zeros(B, STATE_LEN, OUT_DIM)
    def init_score_state():
        return torch.full((B, STATE_LEN, OUT_DIM), float("-inf"))
    def init_wkv():
        return (torch.randn(OUT_DIM, D) - 0.5) / (D ** 0.5)
    def init_wgate():
        return (torch.randn(OUT_DIM, D) - 0.5) / (D ** 0.5)
    def init_ape():
        return torch.randn(COMPRESS_RATIO, OUT_DIM) * 0.01
    def init_norm_w():
        return torch.ones(HEAD_DIM)
    def init_cos():
        return torch.cos(torch.arange(ROPE_HEAD_DIM // 2).reshape(1, ROPE_HEAD_DIM // 2) * 1e-3)
    def init_sin():
        return torch.sin(torch.arange(ROPE_HEAD_DIM // 2).reshape(1, ROPE_HEAD_DIM // 2) * 1e-3)
    def init_hadamard():
        return torch.eye(HEAD_DIM)
    return [
        TensorSpec("x", [B, S, D], torch.bfloat16, init_value=init_x),
        TensorSpec("kv_state", [B, STATE_LEN, OUT_DIM], torch.float32, init_value=init_kv_state, is_output=True),
        TensorSpec("score_state", [B, STATE_LEN, OUT_DIM], torch.float32, init_value=init_score_state, is_output=True),
        TensorSpec("wkv", [OUT_DIM, D], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [OUT_DIM, D], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("cos", [1, ROPE_HEAD_DIM // 2], torch.bfloat16, init_value=init_cos),
        TensorSpec("sin", [1, ROPE_HEAD_DIM // 2], torch.bfloat16, init_value=init_sin),
        TensorSpec("hadamard", [HEAD_DIM, HEAD_DIM], torch.bfloat16, init_value=init_hadamard),
        ScalarSpec("start_pos", torch.int32, START_POS),
        TensorSpec("out", [B, HEAD_DIM], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import RunConfig, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=compressor_test,
        specs=build_tensor_specs(),
        golden_fn=golden_compressor,
        config=RunConfig(
            rtol=2e-3,
            atol=2e-3,
            compile=dict(dump_passes=True),
            runtime=dict(
                platform=args.platform,
                device_id=args.device,
                runtime_profiling=args.runtime_profiling,
            ),
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
