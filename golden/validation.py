# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Golden output validation."""

from collections.abc import Callable

import torch


def validate_golden(
    outputs: dict[str, torch.Tensor],
    golden: dict[str, torch.Tensor],
    rtol: float = 1e-5,
    atol: float = 1e-5,
    compare_fn: dict[str, Callable] | None = None,
    inputs: dict[str, torch.Tensor] | None = None,
) -> None:
    """Compare actual outputs against golden reference.

    By default uses ``torch.allclose``. ``compare_fn`` overrides the default
    for specific output names — useful for tensors where exact equality is
    not the right notion of correctness (e.g. top-k index outputs where
    near-tie scores can produce legal index swaps).

    Each callable in ``compare_fn`` receives:

        cmp(actual, expected, *,
            actual_outputs, expected_outputs, inputs, rtol, atol)
            -> tuple[bool, str]

    where the second tuple element is a diagnostic message used on failure.

    Args:
        outputs: Kernel output tensors keyed by name.
        golden: Golden reference tensors keyed by name.
        rtol: Default relative tolerance.
        atol: Default absolute tolerance.
        compare_fn: Per-name custom comparators, applied instead of allclose.
        inputs: Input tensors of the run, exposed to custom comparators.

    Raises:
        AssertionError: If any output tensor does not match.
    """
    compare_fn = compare_fn or {}
    inputs = inputs or {}
    failures: dict[str, str] = {}
    for name, actual_tensor in outputs.items():
        actual = actual_tensor.cpu()
        expected = golden[name].cpu()

        if name in compare_fn:
            fn = compare_fn[name]
            label = getattr(fn, "__name__", "custom")
            ok, detail = fn(
                actual,
                expected,
                actual_outputs=outputs,
                expected_outputs=golden,
                inputs=inputs,
                rtol=rtol,
                atol=atol,
            )
            if ok:
                print(f"[RUN]   '{name}' PASS  shape={tuple(actual.shape)} dtype={actual.dtype} ({label})")
                continue
            msg = (
                f"  '{name}' FAIL ({label})  shape={tuple(actual.shape)} dtype={actual.dtype}\n"
                f"{detail}"
            )
            print(f"[RUN]   '{name}' FAIL  shape={tuple(actual.shape)} dtype={actual.dtype} ({label})")
            failures[name] = msg
            continue

        ok = torch.allclose(actual, expected, rtol=rtol, atol=atol)
        if ok:
            print(f"[RUN]   '{name}' PASS  shape={tuple(actual.shape)} dtype={actual.dtype}")
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
            f"  '{name}' FAIL  shape={tuple(actual.shape)} dtype={actual.dtype}\n"
            f"    Mismatched elements: {mismatch_indices.numel()}/{actual.numel()}  rtol={rtol} atol={atol}\n"
            f"    first {n_show} mismatches:\n" + "\n".join(lines)
        )
        print(f"[RUN]   '{name}' FAIL  shape={tuple(actual.shape)} dtype={actual.dtype}")
        failures[name] = msg

    if failures:
        detail = "\n".join(failures.values())
        raise AssertionError(
            f"Output(s) does not match golden: {list(failures)}\n{detail}"
        )


def bf16_allclose_or_ulp(max_ulp: int = 1) -> Callable:
    """Return a BF16 comparator that allows a bounded ULP difference.

    The comparator first applies the normal ``torch.isclose`` tolerance. Values
    outside that tolerance are accepted only when their raw BF16 encodings differ
    by at most ``max_ulp``.
    """
    if max_ulp < 0:
        raise ValueError(f"max_ulp must be non-negative, got {max_ulp}")

    def cmp(
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        actual_outputs: dict[str, torch.Tensor],
        expected_outputs: dict[str, torch.Tensor],
        inputs: dict[str, torch.Tensor],
        rtol: float,
        atol: float,
    ) -> tuple[bool, str]:
        if actual.dtype != torch.bfloat16 or expected.dtype != torch.bfloat16:
            return False, (
                f"    bf16_allclose_or_ulp requires BF16 tensors, "
                f"got actual={actual.dtype} expected={expected.dtype}"
            )

        actual_f = actual.cpu().to(torch.float32)
        expected_f = expected.cpu().to(torch.float32)
        close_mask = torch.isclose(actual_f, expected_f, rtol=rtol, atol=atol)
        finite_mask = torch.isfinite(actual_f) & torch.isfinite(expected_f)

        actual_bits = actual.cpu().contiguous().view(torch.int16).to(torch.int32)
        expected_bits = expected.cpu().contiguous().view(torch.int16).to(torch.int32)
        ulp_diff = torch.abs(actual_bits - expected_bits)
        ok_mask = close_mask | (finite_mask & (ulp_diff <= max_ulp))
        if bool(ok_mask.all()):
            return True, ""

        mismatch_indices = torch.where(~ok_mask.flatten())[0]
        flat_actual = actual_f.flatten()
        flat_expected = expected_f.flatten()
        flat_ulp = ulp_diff.flatten()
        n_show = min(20, mismatch_indices.numel())
        idx = mismatch_indices[:n_show]
        lines = [
            (
                f"    [{i.item()}] actual={flat_actual[i].item()}, "
                f"expected={flat_expected[i].item()}, ulp_diff={flat_ulp[i].item()}"
            )
            for i in idx
        ]
        detail = (
            f"    Mismatched elements after {max_ulp}-ULP allowance: "
            f"{mismatch_indices.numel()}/{actual.numel()}  rtol={rtol} atol={atol}\n"
            f"    first {n_show} mismatches:\n" + "\n".join(lines)
        )
        return False, detail

    cmp.__name__ = f"bf16_allclose_or_ulp(max_ulp={max_ulp})"
    return cmp


def topk_pair_compare(vals_name: str) -> Callable:
    """Return a comparator for top-k outputs that is robust to score ties.

    For a top-k operation that emits both an index tensor and a paired value
    tensor, kernel-vs-golden index mismatches are legal whenever the picked
    score sets are equivalent — e.g. when INT8 quantization collapses several
    candidates onto the same score.

    The returned comparator looks up the paired value tensors (kernel and
    golden) by ``vals_name`` and checks that, per row, the values are equal
    after sorting. This passes legal tie-break swaps and fails real misses
    (where one side picked a strictly lower-scoring candidate).

    Use it for the index tensor; the value tensor itself can stay on the
    default ``allclose`` path because top-k outputs are conventionally
    emitted in descending score order, so equivalent score sets line up
    positionally.

        compare_fn = {
            "topk_idx_out": topk_pair_compare("topk_vals_out"),
        }
    """
    def cmp(
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        actual_outputs: dict[str, torch.Tensor],
        expected_outputs: dict[str, torch.Tensor],
        inputs: dict[str, torch.Tensor],
        rtol: float,
        atol: float,
    ) -> tuple[bool, str]:
        if vals_name not in actual_outputs or vals_name not in expected_outputs:
            return False, (
                f"    compare_fn misconfigured: vals_name='{vals_name}' not found "
                f"(outputs={list(actual_outputs)}, golden={list(expected_outputs)})"
            )
        a_vals = actual_outputs[vals_name].cpu().to(torch.float32)
        e_vals = expected_outputs[vals_name].cpu().to(torch.float32)
        if a_vals.shape != e_vals.shape:
            return False, f"    vals shape mismatch: {tuple(a_vals.shape)} vs {tuple(e_vals.shape)}"
        a_sorted = torch.sort(a_vals, dim=-1, descending=True).values
        e_sorted = torch.sort(e_vals, dim=-1, descending=True).values
        ok = torch.allclose(a_sorted, e_sorted, rtol=rtol, atol=atol)
        if ok:
            return True, ""
        diff = (a_sorted - e_sorted).abs()
        flat_diff = diff.reshape(-1, diff.shape[-1])
        b_worst = int(flat_diff.amax(dim=-1).argmax().item())
        a_row = a_sorted.reshape(-1, a_sorted.shape[-1])[b_worst]
        e_row = e_sorted.reshape(-1, e_sorted.shape[-1])[b_worst]
        worst_diff = float((a_row - e_row).abs().max().item())
        return False, (
            f"    top-k pair mismatch via '{vals_name}' "
            f"(rtol={rtol} atol={atol}): worst row={b_worst} max_diff={worst_diff:.6g}\n"
            f"      actual_sorted  = {a_row.tolist()}\n"
            f"      expected_sorted= {e_row.tolist()}"
        )
    cmp.__name__ = "topk_pair_compare"
    return cmp


def ratio_allclose(
    atol: float | None = None,
    rtol: float | None = None,
    max_error_ratio: float = 0.005,
    max_show: int = 10,
) -> Callable:
    """Return an allclose-style comparator that tolerates a bounded outlier ratio.

    Mirrors ``torch.allclose``'s per-point tolerance rule but, instead of
    requiring every point to pass, allows up to ``max_error_ratio`` of points
    to exceed tolerance:

        tolerance = atol + rtol * |expected|
        pass iff (count of points where |actual - expected| > tolerance) / numel
                 <= max_error_ratio

    Useful for quantized kernels where a small fraction of points may diverge
    from the FP reference due to INT8 round-off, while the bulk of the output
    stays within a tight per-point tolerance.

    NaN / Inf in ``actual`` always fail (hard check, independent of the ratio).

    Args:
        atol: Absolute tolerance. If ``None``, falls back to ``validate_golden``'s atol.
        rtol: Relative tolerance. If ``None``, falls back to ``validate_golden``'s rtol.
        max_error_ratio: Fraction of points permitted to exceed tolerance
            (default 0.5%). Set to 0.0 for strict allclose semantics.
        max_show: Maximum number of mismatched points printed on failure.

    Example — attention output with INT8 activation quant::

        compare_fn = {
            "attn_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        }
    """
    if max_error_ratio < 0.0 or max_error_ratio > 1.0:
        raise ValueError(f"max_error_ratio must be in [0, 1], got {max_error_ratio}")

    def cmp(
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        actual_outputs: dict[str, torch.Tensor],
        expected_outputs: dict[str, torch.Tensor],
        inputs: dict[str, torch.Tensor],
        rtol: float,
        atol: float,
    ) -> tuple[bool, str]:
        eff_atol = atol if (cmp.atol_override is None) else cmp.atol_override
        eff_rtol = rtol if (cmp.rtol_override is None) else cmp.rtol_override

        actual_f = actual.cpu().to(torch.float32)
        expected_f = expected.cpu().to(torch.float32)

        nan_count = int(torch.isnan(actual_f).sum().item())
        inf_count = int(torch.isinf(actual_f).sum().item())
        if nan_count or inf_count:
            return False, (
                f"    illegal values in actual: NaN={nan_count} Inf={inf_count}"
            )

        diff_abs = (actual_f - expected_f).abs()
        tolerance = eff_atol + eff_rtol * expected_f.abs()
        bad_mask = diff_abs > tolerance
        error_count = int(bad_mask.sum().item())
        numel = actual_f.numel()
        threshold = round(max_error_ratio * numel)

        max_diff, flat_max_pos = torch.max(diff_abs.flatten(), dim=0)
        max_pos = torch.unravel_index(flat_max_pos, actual_f.shape)
        max_pos = tuple(int(i.item()) for i in max_pos)
        max_tol = float(tolerance[max_pos].item())

        if error_count <= threshold:
            return True, ""

        bad_indices = torch.where(bad_mask.flatten())[0]
        flat_actual = actual_f.flatten()
        flat_expected = expected_f.flatten()
        flat_tol = tolerance.flatten()
        flat_diff = diff_abs.flatten()
        n_show = min(max_show, bad_indices.numel())
        idx = bad_indices[:n_show]
        lines = [
            (
                f"    [{i.item()}] actual={flat_actual[i].item():.8g}, "
                f"expected={flat_expected[i].item():.8g}, "
                f"diff={flat_diff[i].item():.4g}, tol={flat_tol[i].item():.4g}"
            )
            for i in idx
        ]
        return False, (
            f"    ratio_allclose fail: error_count={error_count}/{numel} "
            f"(ratio={error_count / numel:.4%}, allowed<={max_error_ratio:.4%}, "
            f"threshold={threshold} pts)\n"
            f"    atol={eff_atol} rtol={eff_rtol}\n"
            f"    max abs diff={max_diff.item():.6g} at {max_pos} (tol={max_tol:.6g})\n"
            f"    first {n_show} mismatches:\n" + "\n".join(lines)
        )

    cmp.atol_override = atol
    cmp.rtol_override = rtol
    cmp.__name__ = (
        f"ratio_allclose(atol={atol}, rtol={rtol}, "
        f"max_error_ratio={max_error_ratio})"
    )
    return cmp
