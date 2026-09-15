# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""`in_proj_qkv` in A8W8: the block's widest projection, hidden -> q, k and v.

    qkv = x @ W^T          [T, 5120] -> [T, 10240], INT8 both sides, bf16 out

The arithmetic, the golden and the CLI are `a8w8_linear`; this file is the
shape, the tiles, and where the weight and the references come from. The
output is the short convolution's input, so it leaves as bf16 rather than
staying quantized.
"""
import a8w8_linear

from config import QWEN3_8_27B

# model shape
K = QWEN3_8_27B.hidden_size                 # contraction: the block's input width
N = QWEN3_8_27B.qkv_width                   # q and k at Hg heads, v at H

MODULE = "in_proj_qkv"

# case shape
T = 8192                # tokens (single sequence, B = 1)

# tiling
M_TILE = 128
N_TILE = 128
K_TILE = 512
M_GROUP = 4
N_GROUP = 2

OUTPUTS = ("y",)                    # bench.py reads it


def build_kernel(t: int = T, k: int = K, n: int = N, inline: bool = False,
                 m_tile: int = M_TILE, n_tile: int = N_TILE, k_tile: int = K_TILE,
                 m_group: int = M_GROUP, n_group: int = N_GROUP, row_off: int = 0):
    return a8w8_linear.build_kernel(t, k, n, name=MODULE, inline=inline, m_tile=m_tile,
                                    n_tile=n_tile, k_tile=k_tile, m_group=m_group,
                                    n_group=n_group, row_off=row_off)


def build_tensor_specs(t: int = T, k: int = K, n: int = N, weights: str | None = None,
                       **_):
    act = a8w8_linear.hidden_activation(QWEN3_8_27B, t, weights)
    w_q, w_scale = a8w8_linear.quantized_weight(QWEN3_8_27B, MODULE, weights)
    return a8w8_linear.build_specs(t, k, n, lambda: act()["q"], lambda: act()["scale"],
                                   w_q, w_scale)


def compare_y(t: int, weights: str | None):
    act = a8w8_linear.hidden_activation(QWEN3_8_27B, t, weights)
    truth, fakequant = a8w8_linear.references(QWEN3_8_27B, MODULE, act, weights)
    return a8w8_linear.compare_y(truth, fakequant, "qkv")


if __name__ == "__main__":
    import argparse
    import os

    from golden import run

    args = a8w8_linear.add_args(
        argparse.ArgumentParser(description=__doc__.splitlines()[0]),
        M_TILE, N_TILE, K_TILE, T, M_GROUP, N_GROUP).parse_args()
    if args.rounds:
        os.environ["PYPTO_BENCH"] = "1"
        os.environ["PYPTO_BENCH_ROUNDS"] = str(args.rounds)
        os.environ["PYPTO_BENCH_WARMUP"] = "5"

    result = run(
        fn=build_kernel(t=args.seq_len, m_tile=args.m_tile, n_tile=args.n_tile,
                        k_tile=args.k_tile, m_group=args.m_group, n_group=args.n_group),
        specs=build_tensor_specs(t=args.seq_len, weights=args.weights),
        golden_fn=a8w8_linear.golden_fn,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        config=dict(platform=args.platform, device_id=args.device),
        rtol=1e-3,
        atol=1e-5,
        compare_fn={"y": compare_y(args.seq_len, args.weights)},
        compile_only=args.compile_only,
    )
    a8w8_linear.report("in_proj_qkv", args, result, K, N)
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
