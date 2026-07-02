# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F401,F403,F405,F821
"""MTP logits smoke.

This validates the first MTP tail contract in the same style as the other
model smokes in this directory:

    mtp_projection.py output -> local vocab-shard logits

The input is the already-projected MTP hidden state. ``mtp_projection.py``
covers ``e_proj(enorm(hidden_states)) + h_proj(hnorm(prev_hidden_states))``.
This smoke keeps a local 512-column shard so both MTP candidate positions are
validated without introducing distributed serving or TP routing dependencies.
Full TP vocabulary routing remains covered by ``lm_head.py``.
"""

import pypto.language as pl

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ


B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
EPS = M.rms_norm_eps
D_INV = 1.0 / D

T_TILE = 8
MATMUL_T_TILE = 16
VOCAB_SHARD = 512
LM_HEAD_K_CHUNK = 128
LM_HEAD_K_BLOCKS = D // LM_HEAD_K_CHUNK
RMS_K_CHUNK = 128
RMS_K_BLOCKS = D // RMS_K_CHUNK

assert D % LM_HEAD_K_CHUNK == 0
assert D % RMS_K_CHUNK == 0
assert T % T_TILE == 0
assert T <= MATMUL_T_TILE


@pl.jit.inline
def mtp_local_logits(
    mtp_hidden: pl.Tensor[[B, S, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_SHARD, D], pl.BF16],
    candidate_logits: pl.Out[pl.Tensor[[T, VOCAB_SHARD], pl.FP32]],
) -> pl.Tensor[[T, VOCAB_SHARD], pl.FP32]:
    hidden_flat = pl.reshape(mtp_hidden, [T, D])
    hidden_pad = pl.create_tensor([MATMUL_T_TILE, D], dtype=pl.BF16)
    logits_pad = pl.create_tensor([MATMUL_T_TILE, VOCAB_SHARD], dtype=pl.FP32)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_logits_hidden_pad"):
        for k0 in pl.pipeline(0, D, LM_HEAD_K_CHUNK, stage=2):
            for t0 in pl.range(0, T, T_TILE):
                hidden_pad[t0 : t0 + T_TILE, k0 : k0 + LM_HEAD_K_CHUNK] = (
                    hidden_flat[t0 : t0 + T_TILE, k0 : k0 + LM_HEAD_K_CHUNK]
                )
            if MATMUL_T_TILE > T:
                hidden_pad[T:MATMUL_T_TILE, k0 : k0 + LM_HEAD_K_CHUNK] = pl.full(
                    [MATMUL_T_TILE - T, LM_HEAD_K_CHUNK],
                    dtype=pl.BF16,
                    value=0.0,
                )

    for t0 in pl.parallel(0, MATMUL_T_TILE, MATMUL_T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_logits_lm_head_shard"):
            for kb in pl.pipeline(LM_HEAD_K_BLOCKS, stage=2):
                k0 = kb * LM_HEAD_K_CHUNK
                hidden_chunk = hidden_pad[t0 : t0 + MATMUL_T_TILE, k0 : k0 + LM_HEAD_K_CHUNK]
                weight_chunk = lm_head_weight[:, k0 : k0 + LM_HEAD_K_CHUNK]
                if kb == 0:
                    acc = pl.matmul(hidden_chunk, weight_chunk, b_trans=True, out_dtype=pl.FP32)
                else:
                    acc = pl.matmul_acc(acc, hidden_chunk, weight_chunk, b_trans=True)
            logits_pad[t0 : t0 + MATMUL_T_TILE, 0:VOCAB_SHARD] = acc

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_logits_output"):
        for t0 in pl.range(0, T, T_TILE):
            candidate_logits[t0 : t0 + T_TILE, 0:VOCAB_SHARD] = (
                logits_pad[t0 : t0 + T_TILE, 0:VOCAB_SHARD]
            )

    return candidate_logits


@pl.jit.inline
def mtp_shared_head_norm(
    mtp_hidden: pl.Tensor[[B, S, D], pl.BF16],
    shared_head_norm_w: pl.Tensor[[D], pl.FP32],
    normed_hidden: pl.Out[pl.Tensor[[B, S, D], pl.BF16]],
) -> pl.Tensor[[B, S, D], pl.BF16]:
    hidden_flat = pl.reshape(mtp_hidden, [T, D])
    normed_flat = pl.reshape(normed_hidden, [T, D])

    for t0 in pl.parallel(0, T, T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_shared_head_norm_rms"):
            sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.pipeline(RMS_K_BLOCKS, stage=2):
                k0 = kb * RMS_K_CHUNK
                hidden_chunk = pl.cast(
                    hidden_flat[t0 : t0 + T_TILE, k0 : k0 + RMS_K_CHUNK],
                    target_type=pl.FP32,
                )
                sq_sum = pl.add(
                    sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(hidden_chunk, hidden_chunk)), [1, T_TILE]),
                )
            inv = pl.reshape(pl.rsqrt(pl.add(pl.mul(sq_sum, D_INV), EPS)), [T_TILE, 1])
            for kb in pl.range(RMS_K_BLOCKS):
                k0 = kb * RMS_K_CHUNK
                hidden_chunk = pl.cast(
                    hidden_flat[t0 : t0 + T_TILE, k0 : k0 + RMS_K_CHUNK],
                    target_type=pl.FP32,
                )
                weight = pl.reshape(shared_head_norm_w[k0 : k0 + RMS_K_CHUNK], [1, RMS_K_CHUNK])
                normed = pl.col_expand_mul(pl.row_expand_mul(hidden_chunk, inv), weight)
                normed_flat[t0 : t0 + T_TILE, k0 : k0 + RMS_K_CHUNK] = pl.cast(
                    normed,
                    target_type=pl.BF16,
                    mode="rint",
                )

    normed_hidden = pl.reshape(normed_flat, [B, S, D])
    return normed_hidden


@pl.jit
def mtp_logits(
    mtp_hidden: pl.Tensor[[B, S, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_SHARD, D], pl.BF16],
    candidate_logits: pl.Out[pl.Tensor[[T, VOCAB_SHARD], pl.FP32]],
):
    candidate_logits = mtp_local_logits(mtp_hidden, lm_head_weight, candidate_logits)
    return candidate_logits


@pl.jit
def mtp_shared_head_logits(
    mtp_hidden: pl.Tensor[[B, S, D], pl.BF16],
    shared_head_norm_w: pl.Tensor[[D], pl.FP32],
    lm_head_weight: pl.Tensor[[VOCAB_SHARD, D], pl.BF16],
    candidate_logits: pl.Out[pl.Tensor[[T, VOCAB_SHARD], pl.FP32]],
):
    normed_hidden = pl.create_tensor([B, S, D], dtype=pl.BF16)
    normed_hidden = mtp_shared_head_norm(mtp_hidden, shared_head_norm_w, normed_hidden)
    candidate_logits = mtp_local_logits(normed_hidden, lm_head_weight, candidate_logits)
    return candidate_logits


def _rms_norm(x, weight):
    import torch

    shape = x.shape
    x_2d = x.reshape(T, D).float()
    sq_sum = torch.zeros(T, 1, dtype=torch.float32)
    for k0 in range(0, D, RMS_K_CHUNK):
        x_chunk = x_2d[:, k0:k0 + RMS_K_CHUNK]
        sq_sum += (x_chunk * x_chunk).sum(dim=1, keepdim=True)
    inv = torch.rsqrt(sq_sum * D_INV + EPS)
    return (x_2d * inv * weight.float().view(1, D)).reshape(shape)


def _local_logits(mtp_hidden, lm_head_weight):
    return mtp_hidden.reshape(T, D).float().matmul(lm_head_weight.float().t())


def golden_mtp_local_logits(tensors):
    tensors["candidate_logits"][:] = _local_logits(
        tensors["mtp_hidden"],
        tensors["lm_head_weight"],
    )
    _check_logits_contract(tensors)


def golden_mtp_shared_head_logits(tensors):
    mtp_hidden = _rms_norm(
        tensors["mtp_hidden"],
        tensors["shared_head_norm_w"],
    ).to(tensors["mtp_hidden"].dtype)
    tensors["candidate_logits"][:] = _local_logits(
        mtp_hidden,
        tensors["lm_head_weight"],
    )
    _check_logits_contract(tensors)


def _candidate_position_rows():
    return [b * S + s for b in range(B) for s in range(S)]


def _check_logits_contract(tensors):
    logits = tensors["candidate_logits"]
    assert tuple(logits.shape) == (T, VOCAB_SHARD), tuple(logits.shape)

    rows = _candidate_position_rows()
    view = logits.reshape(B, S, VOCAB_SHARD)
    for b in range(B):
        for s in range(S):
            row = rows[b * S + s]
            assert bool((view[b, s] == logits[row]).all()), (b, s, row)

    values, indices = logits.float().max(dim=-1)
    assert tuple(values.shape) == (T,)
    assert tuple(indices.shape) == (T,)
    assert bool((indices >= 0).all())
    assert bool((indices < VOCAB_SHARD).all())


def build_tensor_specs(include_shared_head_norm=False):
    import torch
    from golden import TensorSpec

    specs = [
        TensorSpec(
            "mtp_hidden",
            [B, S, D],
            torch.bfloat16,
            init_value=lambda: torch.randn(B, S, D),
        ),
    ]
    if include_shared_head_norm:
        specs.append(
            TensorSpec(
                "shared_head_norm_w",
                [D],
                torch.float32,
                init_value=lambda: torch.ones(D),
            )
        )
    specs.extend([
        TensorSpec(
            "lm_head_weight",
            [VOCAB_SHARD, D],
            torch.bfloat16,
            init_value=lambda: (torch.randn(VOCAB_SHARD, D) / D ** 0.5).to(torch.bfloat16),
        ),
        TensorSpec("candidate_logits", [T, VOCAB_SHARD], torch.float32, is_output=True),
    ])
    return specs


CASES = {
    "local-logits": (
        mtp_logits,
        False,
        golden_mtp_local_logits,
    ),
    "shared-head-logits": (
        mtp_shared_head_logits,
        True,
        golden_mtp_shared_head_logits,
    ),
}


if __name__ == "__main__":
    import argparse
    import torch
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--case", type=str, default="local-logits", choices=sorted(CASES))
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    fn, include_shared_head_norm, golden_fn = CASES[args.case]

    result = run_jit(
        fn=fn,
        specs=build_tensor_specs(include_shared_head_norm=include_shared_head_norm),
        golden_fn=golden_fn,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            # Candidate logits are FP32 accumulations over a BF16 MTP hidden
            # state. The output shape [T, VOCAB_SHARD] keeps both MTP
            # candidate positions visible in flattened decode-token order.
            "candidate_logits": ratio_allclose(atol=1e-2, rtol=1e-2, max_error_ratio=0.02),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
