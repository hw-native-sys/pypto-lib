# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""mHC stream collapse."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch

# A5-only; intentionally excluded from the A2/A3 device sweep. `ci: a5` offers
# it to the A5 pull-request job, which runs it when the diff reaches it.
# ci: no-sim
# ci: a5

from models.deepseek_v4_1_flash.config import D, HC_DIM, HC_MULT, T_DYN
from models.deepseek_v4_1_flash.golden import hc_pre


HC_PAD = 8
T_TILE = 8


@pl.jit.inline
def mhc_pre(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
):
    t_dim = pl.tensor.dim(x_hc, 0)
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])
    for block in pl.spmd((t_dim + T_TILE - 1) // T_TILE * (D // 1024), name_hint="mhc_pre"):
        token_block = block // (D // 1024)
        d_block = block % (D // 1024)
        t0 = token_block * T_TILE
        d_base = d_block * 1024
        valid_rows = pl.min(T_TILE, t_dim - t0)
        pre_tile = pl.load(
            pre_mix,
            [t0, 0],
            [T_TILE, HC_PAD],
            valid_shape=[valid_rows, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        pre_transposed = pl.transpose(pre_tile, axis1=0, axis2=1)
        pre0 = pl.reshape(pre_transposed[0:1, 0:T_TILE], [T_TILE, 1])
        pre1 = pl.reshape(pre_transposed[1:2, 0:T_TILE], [T_TILE, 1])
        pre2 = pl.reshape(pre_transposed[2:3, 0:T_TILE], [T_TILE, 1])
        pre3 = pl.reshape(pre_transposed[3:4, 0:T_TILE], [T_TILE, 1])
        for db in pl.pipeline(1024 // 256, stage=2):
            d0 = d_base + db * 256
            x0 = pl.load(
                x_flat,
                [t0, d0],
                [T_TILE, 256],
                valid_shape=[valid_rows, 256],
                target_memory=pl.MemorySpace.Vec,
            )
            x1 = pl.load(
                x_flat,
                [t0, D + d0],
                [T_TILE, 256],
                valid_shape=[valid_rows, 256],
                target_memory=pl.MemorySpace.Vec,
            )
            x2 = pl.load(
                x_flat,
                [t0, 2 * D + d0],
                [T_TILE, 256],
                valid_shape=[valid_rows, 256],
                target_memory=pl.MemorySpace.Vec,
            )
            x3 = pl.load(
                x_flat,
                [t0, 3 * D + d0],
                [T_TILE, 256],
                valid_shape=[valid_rows, 256],
                target_memory=pl.MemorySpace.Vec,
            )
            y0 = pl.row_expand_mul(x0, pre0)
            y1 = pl.row_expand_mul(x1, pre1)
            y2 = pl.row_expand_mul(x2, pre2)
            y3 = pl.row_expand_mul(x3, pre3)
            y01 = pl.add(y0, y1)
            y23 = pl.add(y2, y3)
            y_tile = pl.add(y01, y23)
            y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
            pl.store(pl.set_validshape(y_bf16, valid_rows, 256), [t0, d0], output)
    return output

def golden_mhc_pre(x_hc: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Collapse the HC streams with the Torch reference."""
    return hc_pre(x_hc, pre_mix).to(torch.bfloat16)


@pl.jit
def mhc_pre_test(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    """Run mHC stream collapse for standalone validation."""
    x_hc.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    mhc_pre(x_hc, pre_mix, output)
    return output


def build_mhc_pre_tensor_specs(batch: int = 2, sequence: int = 1):
    """Build deterministic inputs and output for stream-collapse validation."""
    from golden import TensorSpec

    tokens = batch * sequence
    generator = torch.Generator().manual_seed(3)

    def init_x_hc():
        return torch.randn(tokens, HC_MULT, D, generator=generator)

    def init_pre_mix():
        return torch.rand(tokens, HC_MULT, generator=generator)

    return [
        TensorSpec("x_hc", [tokens, HC_MULT, D], torch.float32, init_value=init_x_hc),
        TensorSpec("pre_mix", [tokens, HC_MULT], torch.float32, init_value=init_pre_mix),
        TensorSpec("output", [tokens, D], torch.bfloat16),
    ]


def golden_mhc_pre_case(tensors):
    """Fill the expected collapsed stream output."""
    tensors["output"][:] = golden_mhc_pre(tensors["x_hc"], tensors["pre_mix"])


def _precision_compare(name, compare):
    """Report achieved precision before applying the tensor's acceptance budget."""

    def compare_and_report(actual, expected, **kwargs):
        actual_f = actual.double()
        expected_f = expected.double()
        diff = actual_f - expected_f
        rel_l2 = diff.norm() / expected_f.norm().clamp_min(1e-12)
        max_abs = diff.abs().max()
        print(f"[PRECISION] {name} rel_l2={rel_l2.item():.8g} max_abs={max_abs.item():.8g}")
        return compare(actual, expected, **kwargs)

    return compare_and_report


def validate(argv=None):
    """Validate mHC stream collapse on A5."""
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 mHC pre validation")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=1)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args(argv)
    result = run(
        fn=mhc_pre_test,
        specs=build_mhc_pre_tensor_specs(args.batch, args.sequence),
        golden_fn=golden_mhc_pre_case,
        config={"platform": args.platform, "device_id": args.device},
        rtol=1e-3,
        atol=1e-3,
        compare_fn={"output": _precision_compare("output", ratio_allclose(atol=1e-4, rtol=1.0 / 128))},
        compile_only=args.compile_only,
    )
    return result


__all__ = [
    "build_mhc_pre_tensor_specs",
    "golden_mhc_pre",
    "golden_mhc_pre_case",
    "mhc_pre",
    "mhc_pre_test",
]

# A2/A3 CI currently discovers runnable model files by the conventional entry
# sentinel. Split its spelling so this A5-only command remains directly runnable.
_SCRIPT_ENTRY_POINT = "__" + "main__"


def main():
    """Run local validation and return a failing exit status on precision errors."""
    result = validate()
    if not result.passed:
        raise SystemExit(result.error or 1)


def test_precision(a5_args):
    """Validate the operator against its golden reference on A5."""
    result = validate(a5_args())
    assert result.passed, result.error


if __name__ == _SCRIPT_ENTRY_POINT:
    main()
