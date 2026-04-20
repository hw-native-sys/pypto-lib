# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Golden output validation."""

import torch


def validate_golden(
    outputs: dict[str, torch.Tensor],
    golden: dict[str, torch.Tensor],
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    """Compare actual outputs against golden reference using ``torch.allclose``.

    Reports the per-tensor result for every output (pass or fail), then raises
    if any tensor failed. This makes mismatches in later tensors visible even
    when an earlier one already failed.

    Raises:
        AssertionError: If any output tensor does not match within tolerances.
    """
    failures: dict[str, str] = {}
    for name, actual_tensor in outputs.items():
        actual = actual_tensor.cpu()
        expected = golden[name].cpu()

        ok = torch.allclose(actual, expected, rtol=rtol, atol=atol)
        if ok:
            print(f"  [{name}] PASS  shape={tuple(actual.shape)} dtype={actual.dtype}")
            continue

        close_mask = torch.isclose(actual, expected, rtol=rtol, atol=atol)
        mismatch_indices = torch.where(~close_mask.flatten())[0]
        flat_actual = actual.flatten()
        flat_expected = expected.flatten()
        n_show = min(20, mismatch_indices.numel())
        idx = mismatch_indices[:n_show]
        lines = [
            f"    [{i.item()}] actual={flat_actual[i].item()}, expected={flat_expected[i].item()}"
            for i in idx
        ]
        msg = (
            f"  [{name}] FAIL  shape={tuple(actual.shape)} dtype={actual.dtype}\n"
            f"    Mismatched elements: {mismatch_indices.numel()}/{actual.numel()}  rtol={rtol} atol={atol}\n"
            f"    first {n_show} mismatches:\n" + "\n".join(lines)
        )
        print(msg)
        failures[name] = msg

    if failures:
        detail = "\n".join(failures.values())
        raise AssertionError(
            f"Output(s) does not match golden: {list(failures)}\n{detail}"
        )
