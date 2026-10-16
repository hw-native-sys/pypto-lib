# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Final HC stream collapse for DeepSeek-V4.1 Flash (thin wrapper over mhc_pre)."""

import pypto.language as pl
import torch

from config import FLASH as M, T_DYN


from hc_pre import build_mhc_pre_tensor_specs, mhc_pre
from hc_pre import golden_hc_pre


# model config
D = M.hidden_size
HC_MULT = M.hc_mult


def golden_hc_head(
    x_hc: torch.Tensor,
    pre_mix: torch.Tensor,
) -> torch.Tensor:
    """Collapse the final HC streams with the last layer's delayed pre-mix."""
    return golden_hc_pre(x_hc, pre_mix).to(torch.bfloat16)


@pl.jit.inline
def hc_head(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
):
    """Collapse the final HC streams with the delayed pre-mix via mhc_pre."""
    return mhc_pre(x_hc, pre_mix, output)


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


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 Flash HC head validation")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=1)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    result = run(
        fn=hc_head_test,
        specs=build_hc_head_tensor_specs(args.batch, args.sequence),
        golden_fn=golden_hc_head_case,
        config={"platform": args.platform, "device_id": args.device},
        rtol=1e-3,
        atol=1e-3,
        compare_fn={"output": ratio_allclose(atol=1e-4, rtol=1.0 / 128)},
        compile_only=args.compile_only,
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
