# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Comparators this model's validation entries share. Host only; no kernels here."""


def rel_frobenius(threshold: float, label: str):
    """Relative Frobenius norm against the reference, with the peak reported.

    `torch.allclose` on a large FP32 state reports the first element that fails and not
    how far off the tensor is, which is the number a precision gate needs.
    """
    def compare(actual, expected, **_kwargs):
        num = (actual.double() - expected.double()).norm()
        den = expected.double().norm()
        rel = float(num / den) if float(den) > 0 else float(num)
        peak = float((actual.double() - expected.double()).abs().max())
        detail = f"{label} rel {rel:.3e} (<= {threshold:.0e}), peak abs {peak:.3e}"
        ok = rel <= threshold
        return ok, "" if ok else detail

    return compare


def int8_step_and_exact(min_exact: float, label: str = "act_q"):
    """Two gates on an INT8 output: every element within one step, and most exact.

    A step bound alone accepts a rounding-mode change -- truncation instead of
    round-to-nearest moves a large fraction of the output by exactly one step and a step
    bound accepts all of it. The exact-match rate is what separates the two, so it is a
    gate and not a statistic. The step bound itself delegates to `golden.ratio_allclose`.
    """
    import torch
    from golden import ratio_allclose

    step_gate = ratio_allclose(atol=1, rtol=0, max_error_ratio=0.0)

    def compare(actual, expected, actual_outputs=None, **kw):
        ok, detail = step_gate(actual, expected, actual_outputs=actual_outputs, **kw)
        step = (actual.to(torch.int32) - expected.to(torch.int32)).abs()
        exact = float((step == 0).float().mean())
        line = (f"[stats] {label}: {exact * 100:.4f}% exact (gate {min_exact * 100:.0f}%), "
                f"max {int(step.max())} step(s) off")
        if exact < min_exact:
            why = f"only {exact * 100:.2f}% exact, below the {min_exact * 100:.0f}% gate"
            return False, f"{line}\n  {why}"
        return ok, (detail + "\n" + line) if not ok else ""

    return compare


__all__ = ["int8_step_and_exact", "rel_frobenius"]
