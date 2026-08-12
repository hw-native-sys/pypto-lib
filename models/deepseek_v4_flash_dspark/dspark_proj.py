# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 DSpark main-hidden projection and RMSNorm.

Collapses the three target-layer hidden states into one drafter hidden row.
``main_proj`` stays BF16: the W8A8 checkpoint quantizes it only under an FP8
quant method, so the ascend serving path runs it as a plain replicated linear.
"""

import pypto.language as pl

from config import DECODE_BATCH, DECODE_SEQ, FLASH as M, PREFILL_BATCH, PREFILL_SEQ, TP
from rmsnorm import golden_rms_norm, rms_norm


# Dynamic shape variables.
T_DYN = pl.dynamic("DSPARK_PROJ_T_DYN")

# model config
D = M.hidden_size
TARGET_LAYERS = 3                        # dspark_target_layer_ids
MAIN_HIDDEN_DIM = TARGET_LAYERS * D

# tiling
T_TILE = 16                              # cube M-tile; matmul rows must be a multiple of 16
N_TILE = 128                             # 32 output blocks over D
K_TILE = 512                             # 24 reduction steps over MAIN_HIDDEN_DIM


@pl.jit.inline
def dspark_proj(
    main_hidden: pl.Tensor[[T_DYN, MAIN_HIDDEN_DIM], pl.BF16],
    main_proj_w: pl.Tensor[[D, MAIN_HIDDEN_DIM], pl.BF16],
    main_norm_w: pl.Tensor[[D], pl.BF16],
    main_x: pl.Tensor[[T_DYN, D], pl.BF16],
):
    t_dim = pl.tensor.dim(main_hidden, 0)
    t_matmul = ((t_dim + T_TILE - 1) // T_TILE) * T_TILE

    projected = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    for proj_idx in pl.spmd(
        (t_matmul // T_TILE) * (D // N_TILE), name_hint="dspark_main_proj", allow_early_resolve=True
    ):
        t0 = (proj_idx // (D // N_TILE)) * T_TILE
        n0 = (proj_idx % (D // N_TILE)) * N_TILE
        proj_rows = pl.min(T_TILE, t_dim - t0)
        hidden_k0 = pl.slice(main_hidden, [T_TILE, K_TILE], [t0, 0], valid_shape=[proj_rows, K_TILE])
        weight_k0 = main_proj_w[n0 : n0 + N_TILE, 0:K_TILE]
        proj_acc = pl.matmul(hidden_k0, weight_k0, b_trans=True, out_dtype=pl.FP32)
        for k0 in pl.pipeline(K_TILE, MAIN_HIDDEN_DIM, K_TILE, stage=2):
            hidden_k = pl.slice(main_hidden, [T_TILE, K_TILE], [t0, k0], valid_shape=[proj_rows, K_TILE])
            weight_k = main_proj_w[n0 : n0 + N_TILE, k0 : k0 + K_TILE]
            proj_acc = pl.matmul_acc(proj_acc, hidden_k, weight_k, b_trans=True)
        proj_bf16 = pl.cast(proj_acc, target_type=pl.BF16, mode="rint")
        proj_valid = pl.set_validshape(proj_bf16, proj_rows, N_TILE)
        projected[t0 : t0 + T_TILE, n0 : n0 + N_TILE] = proj_valid

    rms_norm(projected, main_norm_w, main_x)
    return main_x


@pl.jit
def dspark_proj_test(
    main_hidden: pl.Tensor[[T_DYN, MAIN_HIDDEN_DIM], pl.BF16],
    main_proj_w: pl.Tensor[[D, MAIN_HIDDEN_DIM], pl.BF16],
    main_norm_w: pl.Tensor[[D], pl.BF16],
    main_x: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    main_hidden.bind_dynamic(0, T_DYN)
    main_x.bind_dynamic(0, T_DYN)
    return dspark_proj(main_hidden, main_proj_w, main_norm_w, main_x)


def golden_dspark_proj(tensors):
    import torch

    main_hidden = tensors["main_hidden"].to(torch.bfloat16).float()
    main_proj_w = tensors["main_proj_w"].to(torch.bfloat16).float()
    projected = torch.matmul(main_hidden, main_proj_w.t()).to(torch.bfloat16)
    tensors["main_x"][:] = golden_rms_norm(projected, tensors["main_norm_w"])


def build_tensor_specs(batch=DECODE_BATCH // TP, seq=DECODE_SEQ):
    import torch
    from golden import TensorSpec

    t = batch * seq

    def init_main_hidden():
        return torch.randn(t, MAIN_HIDDEN_DIM, dtype=torch.bfloat16)

    def init_main_proj_w():
        return (torch.randn(D, MAIN_HIDDEN_DIM) / MAIN_HIDDEN_DIM ** 0.5).to(torch.bfloat16)

    return [
        TensorSpec("main_hidden", [t, MAIN_HIDDEN_DIM], torch.bfloat16, init_value=init_main_hidden),
        TensorSpec("main_proj_w", [D, MAIN_HIDDEN_DIM], torch.bfloat16, init_value=init_main_proj_w),
        TensorSpec("main_norm_w", [D], torch.bfloat16, init_value=lambda: torch.randn(D) * 0.1 + 1.0),
        TensorSpec("main_x", [t, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="DeepSeek-V4 DSpark main-hidden projection validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["decode", "prefill", "all"], default="all")
    parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2, 4))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    modes = {
        "decode": (DECODE_BATCH // TP, DECODE_SEQ),
        "prefill": (PREFILL_BATCH, PREFILL_SEQ),
    }
    for mode in (modes if args.mode == "all" else [args.mode]):
        batch, seq = modes[mode]
        print(f"--- dspark_proj_test {mode}: T={batch * seq} ---")
        result = run_jit(
            fn=dspark_proj_test,
            specs=build_tensor_specs(batch, seq),
            golden_fn=golden_dspark_proj,
            compile_cfg=dict(dump_passes=args.dump_passes),
            runtime_cfg=dict(
                platform=args.platform,
                device_id=args.device,
                enable_l2_swimlane=args.enable_l2_swimlane,
            ),
            rtol=5e-3,
            atol=5e-3,
            compare_fn={
                "main_x": ratio_allclose(atol=5e-3, rtol=5e-3, max_error_ratio=0.01),
            },
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
