# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""mHC stream collapse."""

import pypto.language as pl
import torch

from config import FLASH as M, T_DYN



# model config
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim


D_TILE = 2560


def golden_hc_pre(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Collapse four Hyper-Connection streams into one sublayer input."""
    return (x.float() * pre_mix.float().unsqueeze(-1)).sum(dim=-2).to(x.dtype)


@pl.jit.inline
def mhc_pre(
    x_hc: pl.Tensor,
    pre_mix: pl.Tensor,
    output: pl.Tensor,
):
    t_dim = pl.tensor.dim(x_hc, 0)
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])
    for t in pl.spmd(t_dim, name_hint="mhc_pre", allow_early_resolve=True):
        pre0 = pl.read(pre_mix, [t, 0])
        pre1 = pl.read(pre_mix, [t, 1])
        pre2 = pl.read(pre_mix, [t, 2])
        pre3 = pl.read(pre_mix, [t, 3])
        for d0 in pl.pipeline(0, D, D_TILE, stage=2):
            x0 = pl.load(x_flat, [t, d0], [1, D_TILE])
            x1 = pl.load(x_flat, [t, D + d0], [1, D_TILE])
            x2 = pl.load(x_flat, [t, 2 * D + d0], [1, D_TILE])
            x3 = pl.load(x_flat, [t, 3 * D + d0], [1, D_TILE])
            y0 = pl.mul(x0, pre0)
            y1 = pl.mul(x1, pre1)
            y2 = pl.mul(x2, pre2)
            y3 = pl.mul(x3, pre3)
            y_tile = pl.add(pl.add(y0, y1), pl.add(y2, y3))
            y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
            pl.store(y_bf16, [t, d0], output)
    return output


def golden_mhc_pre(x_hc: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Collapse the HC streams with the Torch reference."""
    return golden_hc_pre(x_hc, pre_mix).to(torch.bfloat16)


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


def main():
    """Run local validation and return a failing exit status on precision errors."""
    result = validate()
    if not result.passed:
        raise SystemExit(result.error or 1)


# Resident CED prefill precision variants.
_PREFILL_MOE_T = pl.dynamic("V41_FP32_MOE_T")


_PREFILL_MOE_WORKERS = 32


_PREFILL_MOE_HC = M.hc_mult


_PREFILL_MOE_D = M.hidden_size


_PREFILL_MOE_TILE = 256


@pl.jit.inline(auto_scope=False)
def prefill_hc_pre_inline(
    x_hc: pl.Tensor[[_PREFILL_MOE_T, _PREFILL_MOE_HC, _PREFILL_MOE_D], pl.FP32],
    pre_mix: pl.Tensor[[_PREFILL_MOE_T, _PREFILL_MOE_HC], pl.FP32],
    output: pl.Tensor[[_PREFILL_MOE_T, _PREFILL_MOE_D], pl.FP32],
):
    """Collapse the four residual streams without narrowing."""
    x_hc.bind_dynamic(0, _PREFILL_MOE_T)
    pre_mix.bind_dynamic(0, _PREFILL_MOE_T)
    output.bind_dynamic(0, _PREFILL_MOE_T)
    rows = pl.tensor.dim(x_hc, 0)
    flat = pl.reshape(x_hc, [rows, _PREFILL_MOE_HC * _PREFILL_MOE_D])
    for worker in pl.spmd(_PREFILL_MOE_WORKERS, name_hint="fp32_hc_pre"):
        for task in pl.range(worker, rows * (_PREFILL_MOE_D // _PREFILL_MOE_TILE), _PREFILL_MOE_WORKERS):
            row = task // (_PREFILL_MOE_D // _PREFILL_MOE_TILE)
            col = task % (_PREFILL_MOE_D // _PREFILL_MOE_TILE) * _PREFILL_MOE_TILE
            value = pl.mul(pl.load(flat, [row, col], [1, _PREFILL_MOE_TILE]), pl.read(pre_mix, [row, 0]))
            for stream in pl.unroll(1, _PREFILL_MOE_HC):
                part = pl.mul(
                    pl.load(flat, [row, stream * _PREFILL_MOE_D + col], [1, _PREFILL_MOE_TILE]),
                    pl.read(pre_mix, [row, stream]),
                )
                value = pl.add(value, part)
            pl.store(value, [row, col], output)
    return output

if __name__ == "__main__":
    main()
