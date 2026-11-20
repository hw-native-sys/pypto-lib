# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""`out_proj` in A8W8: the block's last step, the normed output back to hidden.

    out = y @ W^T          [T, 6144] -> [T, 5120], INT8 both sides, bf16 out

The arithmetic, the golden and the CLI are `a8w8_linear`; this file is the
shape, the tiles, and where the weight and the references come from.

Its activation is not quantized here: `gated_rmsnorm` writes `y_q` and
`y_scale` in its own epilogue, so the normed values never reach GM in bf16.
That is why there is no `quant_y` kernel beside `quant_x`.
"""
import a8w8_linear

from config import QWEN3_8_27B

# model shape
K = QWEN3_8_27B.value_width                 # contraction: H value heads of D
N = QWEN3_8_27B.hidden_size                 # back to the residual stream

MODULE = "out_proj"

# validation
DEFAULT_TEST_SEQ_LEN = 512

# tiling
M_TILE = 128
N_TILE = 128
K_TILE = 512
M_GROUP = 8
N_GROUP = 2

OUTPUTS = ("y",)                    # bench.py reads it


def build_kernel(t: int = DEFAULT_TEST_SEQ_LEN, k: int = K, n: int = N, inline: bool = False,
                 m_tile: int = M_TILE, n_tile: int = N_TILE, k_tile: int = K_TILE,
                 m_group: int | None = None, n_group: int = N_GROUP):
    if m_group is None:
        m_group = a8w8_linear.default_m_group(t, m_tile, M_GROUP)
    return a8w8_linear.build_kernel(t, k, n, name=MODULE, inline=inline, m_tile=m_tile,
                                    n_tile=n_tile, k_tile=k_tile, m_group=m_group,
                                    n_group=n_group)


def build_tensor_specs(t: int = DEFAULT_TEST_SEQ_LEN, k: int = K, n: int = N, weights: str | None = None, **_):
    from functools import cache

    import torch

    from reference import quantize_rows
    from weights import quantized_weight

    @cache
    def activation():
        generator = torch.Generator().manual_seed(42)
        x = torch.randn(t, k, generator=generator)
        q, scale = quantize_rows(x)
        return q, scale.reshape(1, t)

    w_q, w_scale = quantized_weight(QWEN3_8_27B, MODULE, weights)
    return a8w8_linear.build_specs(t, k, n, lambda: activation()[0], lambda: activation()[1], w_q, w_scale)


def validate() -> None:
    import argparse
    import os

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args = a8w8_linear.parse_args(parser, M_TILE, N_TILE, K_TILE, DEFAULT_TEST_SEQ_LEN, M_GROUP, N_GROUP)
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
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        rtol=1e-3,
        atol=1e-5,
        compare_fn={"y": ratio_allclose(atol=3e-3, rtol=3e-3, max_error_ratio=0.02)},
        compile_only=args.compile_only,
    )
    a8w8_linear.report("out_proj", args, result, K, N)
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


if __name__ == "__main__":
    validate()
