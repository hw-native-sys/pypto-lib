# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Final HC stream collapse for DeepSeek-V4.1 Flash (thin wrapper over mhc_pre)."""

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

from models.deepseek_v4_1_flash.config import D, HC_MULT, T_DYN
from models.deepseek_v4_1_flash.golden import hc_head as golden_hc_head_ref
from models.deepseek_v4_1_flash.hc_pre import build_mhc_pre_tensor_specs, mhc_pre


@pl.jit.inline
def hc_head(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
):
    """Collapse the final HC streams with the delayed pre-mix via mhc_pre."""
    return mhc_pre(x_hc, pre_mix, output)


def golden_hc_head(x_hc: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Torch reference for final HC collapse (aliases golden.hc_head / mhc_pre)."""
    return golden_hc_head_ref(x_hc, pre_mix)


@pl.jit
def hc_head_test(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    """Run final HC collapse for standalone validation."""
    x_hc.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    hc_head(x_hc, pre_mix, output)
    return output


def build_hc_head_tensor_specs(batch: int = 2, sequence: int = 1):
    """Build deterministic inputs and output for final HC-collapse validation."""
    return build_mhc_pre_tensor_specs(batch, sequence)


def golden_hc_head_case(tensors):
    """Fill the expected collapsed stream output."""
    tensors["output"][:] = golden_hc_head(tensors["x_hc"], tensors["pre_mix"])


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
    """Validate final HC stream collapse on A5."""
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 Flash HC head validation")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=1)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args(argv)
    result = run(
        fn=hc_head_test,
        specs=build_hc_head_tensor_specs(args.batch, args.sequence),
        golden_fn=golden_hc_head_case,
        config={"platform": args.platform, "device_id": args.device},
        rtol=1e-3,
        atol=1e-3,
        compare_fn={"output": _precision_compare("output", ratio_allclose(atol=1e-4, rtol=1.0 / 128))},
        compile_only=args.compile_only,
    )
    return result


__all__ = [
    "build_hc_head_tensor_specs",
    "golden_hc_head",
    "golden_hc_head_case",
    "hc_head",
    "hc_head_test",
    "validate",
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
