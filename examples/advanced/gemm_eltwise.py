# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GEMM + Elementwise — matmul fused with a residual add in one InCore scope.

    output = matmul(attn_out, wo) + hidden_states

Each N tile runs a tiled K-reduction matmul then adds the residual inside a
single pl.at block, keeping the matmul output on chip. Inputs BF16; output FP32.
"""
import pypto.language as pl

BATCH = 16
HIDDEN = 8192
K_TILE = 128         # K dimension tile size for matmul
N_TILE = 64          # N dimension tile size for matmul output


@pl.jit
def gemm_eltwise(
    attn_out: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    hidden_states: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    wo: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
    resid: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.FP32]],
):
    for n0 in pl.parallel(0, HIDDEN, N_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="gemm_eltwise_tile"):
            # tiled K-reduction matmul: attn_out @ wo[:, n-tile]
            acc = pl.create_tensor([BATCH, N_TILE], dtype=pl.FP32)
            for kb in pl.range(HIDDEN // K_TILE):
                k0 = kb * K_TILE
                tile_a = attn_out[:, k0 : k0 + K_TILE]
                tile_w = wo[k0 : k0 + K_TILE, n0 : n0 + N_TILE]
                if kb == 0:
                    acc = pl.matmul(tile_a, tile_w, out_dtype=pl.FP32)
                else:
                    acc = pl.matmul_acc(acc, tile_a, tile_w)

            # fuse the residual add while the matmul output is still on chip
            hidden_tile = pl.cast(hidden_states[:, n0 : n0 + N_TILE], target_type=pl.FP32)
            resid[:, n0 : n0 + N_TILE] = pl.add(acc, hidden_tile)
    return resid


def build_tensor_specs(
    batch: int = BATCH,
    hidden: int = HIDDEN,
):
    import torch
    from golden import TensorSpec

    return [
        TensorSpec("attn_out", [batch, hidden], torch.bfloat16, init_value=torch.randn),
        TensorSpec("hidden_states", [batch, hidden], torch.bfloat16, init_value=torch.randn),
        TensorSpec("wo", [hidden, hidden], torch.bfloat16, init_value=torch.randn),
        TensorSpec("resid", [batch, hidden], torch.float32, is_output=True),
    ]


def golden_gemm_eltwise(tensors):
    import torch

    o_proj = torch.matmul(tensors["attn_out"].float(), tensors["wo"].float())
    tensors["resid"][:] = o_proj + tensors["hidden_states"].float()


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
        fn=gemm_eltwise,
        specs=build_tensor_specs(),
        golden_fn=golden_gemm_eltwise,
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
