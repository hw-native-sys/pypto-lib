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

# model config
K = QWEN3_8_27B.hidden_size                 # contraction: the block's input width
N = QWEN3_8_27B.qkv_width                   # q and k at Hg heads, v at H

MODULE = "in_proj_qkv"
# validation
# one block of M_GROUP * M_TILE token rows, then two. The token count must be a multiple of 512.
DEFAULT_CASES = (512, 1024)
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len

# tiling
M_TILE = 128
N_TILE = 128
K_TILE = 512
K_STAGGER = 1               # rotate this block's K start; see `a8w8_linear`
RING_SLOTS = 2              # GM transport depth; the on-chip reserve stays at 1
M_GROUP = 4
N_GROUP = 2



def build_kernel(t: int = BENCH_SEQ, k: int = K, n: int = N, inline: bool = False,
                 m_tile: int = M_TILE, n_tile: int = N_TILE, k_tile: int = K_TILE,
                 m_group: int = M_GROUP, n_group: int = N_GROUP, row_off: int = 0):
    return a8w8_linear.build_kernel(t, k, n, name=MODULE, inline=inline, m_tile=m_tile,
                                    n_tile=n_tile, k_tile=k_tile, m_group=m_group,
                                    n_group=n_group, row_off=row_off,
                                    k_stagger=K_STAGGER, ring_slots=RING_SLOTS)


def build_tensor_specs(t: int = BENCH_SEQ, k: int = K, n: int = N, weights: str | None = None,
                       **_):
    from weights import quantized_weight

    act = a8w8_linear.hidden_activation(QWEN3_8_27B, t, weights)
    w_q, w_scale = quantized_weight(QWEN3_8_27B, MODULE, weights)
    return a8w8_linear.build_specs(t, k, n, lambda: act()["q"], lambda: act()["scale"],
                                   w_q, w_scale)


def compare_y(t: int, weights: str | None):
    act = a8w8_linear.hidden_activation(QWEN3_8_27B, t, weights)
    truth, fakequant = a8w8_linear.references(QWEN3_8_27B, MODULE, act, weights)
    return a8w8_linear.compare_y(truth, fakequant, "qkv")


def validate() -> None:
    import argparse
    import os

    from golden import run

    parser = a8w8_linear.add_args(
        argparse.ArgumentParser(description=__doc__.splitlines()[0]),
        M_TILE, N_TILE, K_TILE, M_GROUP, N_GROUP)
    args = parser.parse_args()
    cases = [args.seq_len] if args.seq_len is not None else list(DEFAULT_CASES)
    if args.m_tile <= 0 or args.m_group <= 0 or any(t <= 0 or t % (args.m_tile * args.m_group) for t in cases):
        parser.error("--seq-len must be a positive multiple of --m-tile times --m-group")
    if len(cases) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one shape; name it with --seq-len")
    if args.rounds:
        os.environ["PYPTO_BENCH"] = "1"
        os.environ["PYPTO_BENCH_ROUNDS"] = str(args.rounds)
        os.environ["PYPTO_BENCH_WARMUP"] = "5"

    for t in cases:
        print(f"[case] in_proj_qkv T={t}", flush=True)
        result = run(
            fn=build_kernel(t=t, m_tile=args.m_tile, n_tile=args.n_tile,
                            k_tile=args.k_tile, m_group=args.m_group, n_group=args.n_group),
            specs=build_tensor_specs(t=t, weights=args.weights),
            golden_fn=a8w8_linear.golden_fn,
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            rtol=1e-3,
            atol=1e-5,
            compare_fn={"y": compare_y(t, args.weights)},
            compile_only=args.compile_only,
        )
        a8w8_linear.report("in_proj_qkv", t, args, result, K, N)
        if not result.passed:
            if result.error:
                print(result.error)
            # Stop at the first failure: one device failure poisons the chip run
            # lane, so every later case reports a failure it did not cause.
            raise SystemExit(f"FAILED at T={t}")


if __name__ == "__main__":
    validate()
