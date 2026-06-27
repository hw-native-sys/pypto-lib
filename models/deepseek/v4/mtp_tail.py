# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F401,F403,F405,F821
"""MTP projection-to-logits smoke.

This file composes the two independently validated pieces in this directory:

    hidden_states + prev_hidden_states
        -> mtp_projection_impl
        -> mtp_local_logits
        -> candidate_logits

It is intentionally still a single-card local-vocab-shard smoke. Distributed
TP routing remains covered by ``lm_head.py`` and serving acceptance/state
updates are left to the serving integration stage.
"""

import pypto.language as pl

from mtp_logits import (
    VOCAB_SHARD,
    _check_logits_contract,
    _local_logits,
    mtp_local_logits,
)
from mtp_projection import (
    B,
    D,
    S,
    T,
    build_tensor_specs as build_projection_tensor_specs,
    golden_mtp_projection,
    mtp_projection_impl,
)


@pl.jit
def mtp_tail(
    hidden_states: pl.Tensor[[B, S, D], pl.BF16],
    prev_hidden_states: pl.Tensor[[B, S, D], pl.BF16],
    enorm_w: pl.Tensor[[D], pl.FP32],
    hnorm_w: pl.Tensor[[D], pl.FP32],
    e_proj_w: pl.Tensor[[D, D], pl.INT8],
    e_proj_w_scale: pl.Tensor[[D], pl.FP32],
    e_proj_smooth: pl.Tensor[[D], pl.FP32],
    h_proj_w: pl.Tensor[[D, D], pl.INT8],
    h_proj_w_scale: pl.Tensor[[D], pl.FP32],
    h_proj_smooth: pl.Tensor[[D], pl.FP32],
    lm_head_weight: pl.Tensor[[VOCAB_SHARD, D], pl.BF16],
    candidate_logits: pl.Out[pl.Tensor[[T, VOCAB_SHARD], pl.FP32]],
):
    mtp_hidden = pl.create_tensor([B, S, D], dtype=pl.BF16)
    mtp_hidden = mtp_projection_impl(
        hidden_states,
        prev_hidden_states,
        enorm_w,
        hnorm_w,
        e_proj_w,
        e_proj_w_scale,
        e_proj_smooth,
        h_proj_w,
        h_proj_w_scale,
        h_proj_smooth,
        mtp_hidden,
    )
    candidate_logits = mtp_local_logits(mtp_hidden, lm_head_weight, candidate_logits)
    return candidate_logits


def golden_mtp_tail(tensors):
    import torch

    tensors["hidden_states_out"] = torch.empty(B, S, D, dtype=torch.bfloat16)
    golden_mtp_projection(tensors)
    tensors["candidate_logits"][:] = _local_logits(
        tensors["hidden_states_out"],
        tensors["lm_head_weight"],
    )
    _check_logits_contract(tensors)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    specs = [
        spec
        for spec in build_projection_tensor_specs()
        if spec.name != "hidden_states_out"
    ]
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
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    result = run_jit(
        fn=mtp_tail,
        specs=build_tensor_specs(),
        golden_fn=golden_mtp_tail,
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
            "candidate_logits": ratio_allclose(atol=1e-2, rtol=1e-2, max_error_ratio=0.02),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
