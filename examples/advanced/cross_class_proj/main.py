# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Cross-class / cross-file ``@pl.jit.inline`` example.

    output = (x @ w) + hidden_states

The two stages live on two different classes in two different files:

* ``proj_lib.Projections.linear``   — tiled matmul (BF16 x BF16 -> FP32)
* ``eltwise_lib.Elementwise.residual_add`` — FP32 tensor + cast(BF16 -> FP32)

Both are decorated with ``@pl.jit.inline``, so the ``InlineFunctions`` IR
pass splices their bodies into this entry function during compilation —
producing the same lowered IR as a hand-fused single-function kernel, but
with the source split across files for reuse.

Run with::

    python examples/advanced/cross_class_proj/main.py -p a2a3sim
"""

import pypto.language as pl

from config import BATCH, HIDDEN
from eltwise_lib import Elementwise
from proj_lib import Projections

linear = Projections.linear
residual_add = Elementwise.residual_add

class ProjResidual:
    @pl.jit
    def proj_residual(
        x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        w: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
        hidden_states: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.FP32]],
    ):
        # Stage 0: linear projection from another class in another file.
        proj_out = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
        proj_out = linear(x, w, proj_out)

        # Stage 1: residual add from yet another class in another file.
        out = residual_add(proj_out, hidden_states, out)
        return out


def build_tensor_specs():
    import torch

    from golden import TensorSpec

    scale = HIDDEN ** 0.5

    def init_x():
        return torch.rand(BATCH, HIDDEN) - 0.5

    def init_w():
        return (torch.rand(HIDDEN, HIDDEN) - 0.5) / scale

    def init_h():
        return torch.rand(BATCH, HIDDEN) - 0.5

    return [
        TensorSpec("x",             [BATCH, HIDDEN],  torch.bfloat16, init_value=init_x),
        TensorSpec("w",             [HIDDEN, HIDDEN], torch.bfloat16, init_value=init_w),
        TensorSpec("hidden_states", [BATCH, HIDDEN],  torch.bfloat16, init_value=init_h),
        TensorSpec("out",           [BATCH, HIDDEN],  torch.float32,  is_output=True),
    ]


def golden_proj_residual(tensors):
    x_f32 = tensors["x"].float()
    w_f32 = tensors["w"].float()
    h_f32 = tensors["hidden_states"].float()
    tensors["out"][:] = x_f32 @ w_f32 + h_f32


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
        fn=ProjResidual.proj_residual,
        specs=build_tensor_specs(),
        golden_fn=golden_proj_residual,
        config=RunConfig(
            rtol=1e-3,
            atol=1e-3,
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
