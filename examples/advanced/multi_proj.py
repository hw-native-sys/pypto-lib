# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Multi-call projection: ``qkv_proj`` calls a shared ``proj`` three times for Q/K/V.

``proj`` parallelises over the N dimension and pipelines the K reduction.
"""

import pypto.language as pl

BATCH = 16
HIDDEN = 8192
N_OUT_CHUNK = 256       # N tile per parallel core-group
K_PROJ_CHUNK = 128      # K reduction tile inside each scope


@pl.jit.inline
def proj(
    x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    w: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
    y: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.FP32]],
):
    for n0 in pl.parallel(0, HIDDEN, N_OUT_CHUNK):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj"):
            acc = pl.create_tensor([BATCH, N_OUT_CHUNK], dtype=pl.FP32)
            for kb in pl.pipeline(0, HIDDEN // K_PROJ_CHUNK, stage=2):
                k0 = kb * K_PROJ_CHUNK
                tile_x = x[:, k0 : k0 + K_PROJ_CHUNK]
                tile_w = w[k0 : k0 + K_PROJ_CHUNK, n0 : n0 + N_OUT_CHUNK]
                if k0 == 0:
                    acc = pl.matmul(tile_x, tile_w, out_dtype=pl.FP32)
                else:
                    acc = pl.matmul_acc(acc, tile_x, tile_w)
            y = pl.assemble(y, acc, [0, n0])
    return y


@pl.jit
def qkv_proj(
    x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    wq: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
    wk: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
    wv: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
    q_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.FP32]],
    k_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.FP32]],
    v_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.FP32]],
):
    qy = proj(x, wq, q_out)
    ky = proj(x, wk, k_out)
    vy = proj(x, wv, v_out)
    return qy, ky, vy


def build_tensor_specs():
    import torch

    from golden import TensorSpec

    scale = HIDDEN ** 0.5

    def init_x():
        return torch.rand(BATCH, HIDDEN) - 0.5

    def init_w():
        return (torch.rand(HIDDEN, HIDDEN) - 0.5) / scale

    return [
        TensorSpec("x",     [BATCH, HIDDEN],  torch.bfloat16, init_value=init_x),
        TensorSpec("wq",    [HIDDEN, HIDDEN], torch.bfloat16, init_value=init_w),
        TensorSpec("wk",    [HIDDEN, HIDDEN], torch.bfloat16, init_value=init_w),
        TensorSpec("wv",    [HIDDEN, HIDDEN], torch.bfloat16, init_value=init_w),
        TensorSpec("q_out", [BATCH, HIDDEN],  torch.float32,  is_output=True),
        TensorSpec("k_out", [BATCH, HIDDEN],  torch.float32,  is_output=True),
        TensorSpec("v_out", [BATCH, HIDDEN],  torch.float32,  is_output=True),
    ]


def golden_qkv_proj(tensors):
    x_f32 = tensors["x"].float()
    tensors["q_out"][:] = x_f32 @ tensors["wq"].float()
    tensors["k_out"][:] = x_f32 @ tensors["wk"].float()
    tensors["v_out"][:] = x_f32 @ tensors["wv"].float()


if __name__ == "__main__":
    import argparse

    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=qkv_proj,
        specs=build_tensor_specs(),
        golden_fn=golden_qkv_proj,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
