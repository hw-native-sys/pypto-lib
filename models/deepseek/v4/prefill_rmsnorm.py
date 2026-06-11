# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 prefill attention RMSNorm.

This module materializes the model.py attention norm for prefill paths. QKV
projection kernels consume the normalized activation directly.
"""

import pypto.language as pl

from config import FLASH as M, PREFILL_BATCH, PREFILL_SEQ


B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
D = M.hidden_size
EPS = M.rms_norm_eps

ATTN_NORM_GROUP = 4
ATTN_RMS_PARTIALS = 2
D_CHUNK = 128 if T >= 128 else (256 if T >= 64 else 512)
D_BLOCKS = D // D_CHUNK
PREFILL_RMSNORM_TOKEN_CHUNK = min(64, T)
PREFILL_RMSNORM_CHUNKS = T // PREFILL_RMSNORM_TOKEN_CHUNK
PREFILL_ATTN_NORM_T_TILE = 8

assert D % D_CHUNK == 0, "D must be divisible by D_CHUNK"
assert D_BLOCKS % ATTN_NORM_GROUP == 0, "D_BLOCKS must be divisible by ATTN_NORM_GROUP"
assert D_BLOCKS % ATTN_RMS_PARTIALS == 0, "D_BLOCKS must be divisible by ATTN_RMS_PARTIALS"
assert T % PREFILL_RMSNORM_TOKEN_CHUNK == 0, "prefill token count must be divisible by the RMSNorm token chunk"


@pl.jit.inline
def prefill_attn_norm(
    x: pl.Tensor[[T, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.FP32],
    x_normed: pl.Tensor[[T, D], pl.BF16],
):
    for chunk_idx in pl.range(PREFILL_RMSNORM_CHUNKS):
        t0 = chunk_idx * PREFILL_RMSNORM_TOKEN_CHUNK
        x_tile = pl.slice(x, [PREFILL_RMSNORM_TOKEN_CHUNK, D], [t0, 0])

        D_BLOCKS_PER_PARTIAL = D_BLOCKS // ATTN_RMS_PARTIALS
        x_sq_partial = pl.create_tensor([ATTN_RMS_PARTIALS, PREFILL_RMSNORM_TOKEN_CHUNK], dtype=pl.FP32)
        for wg in pl.parallel(0, ATTN_RMS_PARTIALS, 1):
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="prefill_attn_norm_rms_partial"):
                rms_d_base = wg * D_BLOCKS_PER_PARTIAL * D_CHUNK
                local_sum = pl.full([1, PREFILL_RMSNORM_TOKEN_CHUNK], dtype=pl.FP32, value=0.0)
                for rms_db in pl.range(D_BLOCKS_PER_PARTIAL):
                    rms_d0 = rms_d_base + rms_db * D_CHUNK
                    rms_x_chunk = pl.cast(x_tile[:, rms_d0 : rms_d0 + D_CHUNK], target_type=pl.FP32)
                    local_sum = pl.add(
                        local_sum,
                        pl.reshape(pl.row_sum(pl.mul(rms_x_chunk, rms_x_chunk)), [1, PREFILL_RMSNORM_TOKEN_CHUNK]),
                    )
                x_sq_partial[wg : wg + 1, :] = local_sum

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_attn_norm_rms_final"):
            x_sq_sum = pl.full([1, PREFILL_RMSNORM_TOKEN_CHUNK], dtype=pl.FP32, value=0.0)
            for w in pl.range(ATTN_RMS_PARTIALS):
                x_sq_sum = pl.add(x_sq_sum, x_sq_partial[w : w + 1, :])
            x_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(x_sq_sum, 1.0 / D), EPS)))

        x_inv_rms_t = pl.reshape(x_inv_rms, [PREFILL_RMSNORM_TOKEN_CHUNK, 1])
        for dbg in pl.parallel(0, D_BLOCKS, ATTN_NORM_GROUP):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_attn_norm_apply"):
                for d_inner in pl.range(ATTN_NORM_GROUP):
                    apply_d0 = (dbg + d_inner) * D_CHUNK
                    apply_x_chunk = pl.cast(x_tile[:, apply_d0 : apply_d0 + D_CHUNK], target_type=pl.FP32)
                    norm_w_chunk = pl.reshape(norm_w[apply_d0 : apply_d0 + D_CHUNK], [1, D_CHUNK])
                    x_normed_chunk = pl.col_expand_mul(pl.row_expand_mul(apply_x_chunk, x_inv_rms_t), norm_w_chunk)
                    x_normed[t0 : t0 + PREFILL_RMSNORM_TOKEN_CHUNK, apply_d0 : apply_d0 + D_CHUNK] = pl.cast(
                        x_normed_chunk,
                        target_type=pl.BF16,
                        mode="rint",
                    )

    return x_normed


@pl.jit
def prefill_rmsnorm(
    x: pl.Tensor[[T, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.FP32],
    x_normed: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    x_normed = prefill_attn_norm(x, norm_w, x_normed)
    return x_normed


def golden_prefill_attn_norm(x, norm_w):
    import torch

    x = x.float()
    norm_w = norm_w.float()
    inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    return (x * inv * norm_w).to(torch.bfloat16)


def golden_prefill_rmsnorm(tensors):
    tensors["x_normed"][:] = golden_prefill_attn_norm(tensors["x"], tensors["norm_w"])


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_x():
        return torch.randn(T, D) - 0.5

    def init_norm_w():
        return torch.randn(D) * 0.1 + 1.0

    return [
        TensorSpec("x", [T, D], torch.bfloat16, init_value=init_x),
        TensorSpec("norm_w", [D], torch.float32, init_value=init_norm_w),
        TensorSpec("x_normed", [T, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 prefill attention RMSNorm validation.")
    parser.add_argument(
        "-p",
        "--platform",
        type=str,
        default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
        help="PyPTO compile/runtime backend for this standalone validation. Default: %(default)s.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=int,
        default=0,
        help="NPU device id passed to runtime_cfg.device_id. Under task-submit, '{}' is usually substituted here.",
    )
    parser.add_argument(
        "--enable-l2-swimlane",
        action="store_true",
        default=False,
        help="Enable L2 swimlane profiling/report generation in runtime_cfg for this validation run.",
    )
    args = parser.parse_args()

    result = run_jit(
        fn=prefill_rmsnorm,
        specs=build_tensor_specs(),
        golden_fn=golden_prefill_rmsnorm,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=5e-3,
        atol=5e-3,
        compare_fn={
            "x_normed": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
