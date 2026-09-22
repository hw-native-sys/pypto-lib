# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V4.1 Flash RMSNorm kernel and golden validation."""

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

from models.deepseek_v4_1_flash.config import D, FLASH, T_DYN


NORM_EPS = FLASH.rms_norm_eps
NORM_T_TILE = 8
NORM_D_TILE = 512


@pl.jit.inline
def rms_norm(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    x_normed: pl.Tensor[[T_DYN, D], pl.BF16],
):
    """Apply RMSNorm over all rows with FP32 accumulation and one BF16 cast."""
    t_dim = pl.tensor.dim(x, 0)
    for block in pl.spmd((t_dim + NORM_T_TILE - 1) // NORM_T_TILE, name_hint="rms_norm"):
        t0 = block * NORM_T_TILE
        valid_rows = pl.min(NORM_T_TILE, t_dim - t0)
        sq_sum = pl.full([1, NORM_T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(D // NORM_D_TILE, stage=2):
            k0 = kb * NORM_D_TILE
            source = pl.slice(x, [NORM_T_TILE, NORM_D_TILE], [t0, k0], valid_shape=[valid_rows, NORM_D_TILE])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), NORM_T_TILE, NORM_D_TILE)
            value = pl.cast(source, target_type=pl.FP32)
            sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(value, value)), [1, NORM_T_TILE]))
        inv_rms = pl.reshape(
            pl.rsqrt(pl.add(pl.mul(sq_sum, 1.0 / D), NORM_EPS), high_precision=True), [NORM_T_TILE, 1]
        )
        for kb in pl.pipeline(D // NORM_D_TILE, stage=2):
            k0 = kb * NORM_D_TILE
            source = pl.slice(x, [NORM_T_TILE, NORM_D_TILE], [t0, k0], valid_shape=[valid_rows, NORM_D_TILE])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), NORM_T_TILE, NORM_D_TILE)
            value = pl.cast(source, target_type=pl.FP32)
            gamma = pl.reshape(pl.cast(norm_w[k0 : k0 + NORM_D_TILE], target_type=pl.FP32), [1, NORM_D_TILE])
            normalized = pl.col_expand_mul(pl.row_expand_mul(value, inv_rms), gamma)
            x_normed[t0 : t0 + NORM_T_TILE, k0 : k0 + NORM_D_TILE] = pl.set_validshape(
                pl.cast(normalized, target_type=pl.BF16, mode="rint"), valid_rows, NORM_D_TILE
            )
    return x_normed


@pl.jit
def rms_norm_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    x_normed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    """Run RMSNorm for standalone validation."""
    x.bind_dynamic(0, T_DYN)
    x_normed.bind_dynamic(0, T_DYN)
    return rms_norm(x, norm_w, x_normed)


def golden_rms_norm(x: torch.Tensor, norm_w: torch.Tensor) -> torch.Tensor:
    """Apply RMSNorm with the kernel's FP32 chunk accumulation order."""
    value = x.float()
    sq_sum = torch.zeros(value.shape[0], 1, dtype=torch.float32)
    for d0 in range(0, D, NORM_D_TILE):
        chunk = value[:, d0 : d0 + NORM_D_TILE]
        sq_sum += (chunk * chunk).sum(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(sq_sum * (1.0 / D) + NORM_EPS)
    return (value * inv_rms * norm_w.float()).to(torch.bfloat16)


def golden_rms_norm_case(tensors):
    """Fill the expected RMSNorm output."""
    tensors["x_normed"][:] = golden_rms_norm(tensors["x"], tensors["norm_w"])


def build_rms_norm_tensor_specs(tokens: int = 17):
    """Build deterministic inputs including a partial final token tile."""
    from golden import TensorSpec

    if tokens < 1:
        raise ValueError(f"tokens must be positive, got {tokens}")

    generator = torch.Generator().manual_seed(3)

    def init_x():
        return torch.randn(tokens, D, generator=generator) - 0.5

    def init_weight():
        return torch.randn(D, generator=generator) * 0.1 + 1.0

    return [
        TensorSpec("x", [tokens, D], torch.bfloat16, init_value=init_x),
        TensorSpec("norm_w", [D], torch.bfloat16, init_value=init_weight),
        TensorSpec("x_normed", [tokens, D], torch.bfloat16),
    ]


def validate(argv=None):
    """Validate RMSNorm on A5."""
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 Flash RMSNorm validation")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args(argv)
    result = run(
        fn=rms_norm_test,
        specs=build_rms_norm_tensor_specs(args.tokens),
        golden_fn=golden_rms_norm_case,
        config={"platform": args.platform, "device_id": args.device},
        rtol=1e-3,
        atol=1e-3,
        compare_fn={"x_normed": ratio_allclose(atol=1e-4, rtol=1.0 / 128)},
        compile_only=args.compile_only,
    )
    return result


def main():
    """Run local validation and return a failing exit status on precision errors."""
    result = validate()
    if not result.passed:
        raise SystemExit(result.error or 1)


__all__ = [
    "rms_norm",
    "rms_norm_test",
    "build_rms_norm_tensor_specs",
    "golden_rms_norm",
    "golden_rms_norm_case",
]

# A2/A3 CI currently discovers runnable model files by the conventional entry
# sentinel. Split its spelling so this A5-only command remains directly runnable.
_SCRIPT_ENTRY_POINT = "__" + "main__"

if "pytest" in sys.modules:
    def test_precision(a5_args):
        """Validate the operator against its golden reference on A5."""
        result = validate(a5_args())
        assert result.passed, result.error

if __name__ == _SCRIPT_ENTRY_POINT:
    main()
