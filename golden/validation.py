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


def topk_pair_compare(
    vals_name: str,
    *,
    dim: int = -1,
    descending: bool = True,
    max_show: int = 10,
) -> Callable:
    """Return a comparator for top-k idx outputs that tolerates score-tie swaps.

    For a top-k operation that emits both an index tensor and a paired value
    tensor, kernel-vs-golden index mismatches are legal whenever the picked
    candidate's score is tied with its neighbors — e.g. when INT8 quantization
    collapses several candidates onto the same score.

    The returned comparator first does a position-wise idx compare. For each
    position ``i`` where ``actual_idx[i] != expected_idx[i]``, it verifies
    that ``actual_vals`` is still monotonically ordered across ``i`` along
    ``dim`` (descending if ``descending=True``, otherwise ascending) within
    tolerance. A legal tie-swap preserves that order; a real miss — kernel
    picked a strictly worse-scoring candidate at position ``i`` — breaks it.

    The paired ``vals`` output stays on the default ``allclose`` path and is
    what catches "kernel reported a worse score than golden"; this comparator
    only adjudicates idx differences and intentionally does not consult
    ``expected_vals``.

    Parameters
    ----------
    vals_name : name of the paired score tensor in the outputs dict.
    dim : axis along which the top-k is sorted (default ``-1``).
    descending : whether ``actual_vals`` is expected to be in descending order
        along ``dim`` (default ``True``).
    max_show : maximum number of per-position diagnostics to print on failure.

    On failure, up to ``max_show`` per-position diagnostics are printed:
    tensor coordinate, actual_idx, expected_idx, the actual score, and the
    surrounding a_vals window along ``dim``.

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
        if vals_name not in actual_outputs:
            return False, (
                f"    compare_fn misconfigured: vals_name='{vals_name}' not found "
                f"in actual outputs={list(actual_outputs)}"
            )
        a_idx = actual.cpu()
        e_idx = expected.cpu()
        a_vals = actual_outputs[vals_name].cpu().to(torch.float32)
        if a_idx.shape != e_idx.shape:
            return False, f"    idx shape mismatch: {tuple(a_idx.shape)} vs {tuple(e_idx.shape)}"
        if a_idx.shape != a_vals.shape:
            return False, (
                f"    idx/vals shape mismatch: idx={tuple(a_idx.shape)} "
                f"vs vals={tuple(a_vals.shape)}"
            )
        ndim = a_idx.dim()
        dim_pos = dim if dim >= 0 else dim + ndim
        if not 0 <= dim_pos < ndim:
            return False, f"    dim={dim} out of range for shape {tuple(a_idx.shape)}"
        a_idx_m = a_idx.movedim(dim_pos, -1)
        e_idx_m = e_idx.movedim(dim_pos, -1)
        a_vals_m = a_vals.movedim(dim_pos, -1)
        orig_shape = tuple(a_idx.shape)
        leading_axes = [d for d in range(ndim) if d != dim_pos]
        leading_shape = tuple(orig_shape[d] for d in leading_axes)
        a_idx_2d = a_idx_m.reshape(-1, a_idx_m.shape[-1])
        e_idx_2d = e_idx_m.reshape(-1, e_idx_m.shape[-1])
        a_vals_2d = a_vals_m.reshape(-1, a_vals_m.shape[-1])
        n_rows, k = a_idx_2d.shape

        def _coord(r: int, pos: int) -> str:
            coords_leading: list[int] = []
            rem = r
            for sz in reversed(leading_shape):
                coords_leading.append(rem % sz)
                rem //= sz
            coords_leading.reverse()
            full = [0] * ndim
            for idx_pos, axis in enumerate(leading_axes):
                full[axis] = coords_leading[idx_pos]
            full[dim_pos] = pos
            return "[" + ",".join(str(c) for c in full) + "]"

        mismatch_mask = a_idx_2d != e_idx_2d
        if not mismatch_mask.any().item():
            return True, ""

        if k >= 2:
            left_slc = a_vals_2d[:, :-1]
            right_slc = a_vals_2d[:, 1:]
            pair_ok = (left_slc >= right_slc) if descending else (left_slc <= right_slc)
            left_ok = torch.ones_like(mismatch_mask)
            left_ok[:, 1:] = pair_ok  # position i: pair (i-1, i)
            right_ok = torch.ones_like(mismatch_mask)
            right_ok[:, :-1] = pair_ok  # position i: pair (i, i+1)
            pos_ok = left_ok & right_ok
        else:
            pos_ok = torch.ones_like(mismatch_mask)
        fail_mask = mismatch_mask & ~pos_ok
        if not fail_mask.any().item():
            return True, ""

        fail_rc = fail_mask.nonzero(as_tuple=False)
        n_fail = fail_rc.shape[0]
        order_word = "descending" if descending else "ascending"
        lines = [
            f"    top-k idx mismatch via '{vals_name}' "
            f"(dim={dim} order={order_word}): "
            f"{n_fail} position(s) where a_vals breaks {order_word} order at the mismatch"
        ]
        for i in range(min(n_fail, max_show)):
            r = int(fail_rc[i, 0].item())
            pos = int(fail_rc[i, 1].item())
            lo = max(0, pos - 1)
            hi = min(k, pos + 2)
            local = a_vals_2d[r, lo:hi].tolist()
            local_str = ", ".join(f"{v:.6g}" for v in local)
            lines.append(
                f"      {_coord(r, pos)} "
                f"actual_idx={int(a_idx_2d[r, pos].item())} "
                f"expected_idx={int(e_idx_2d[r, pos].item())} "
                f"actual_score={float(a_vals_2d[r, pos].item()):.6g} "
                f"actual_vals[{lo}:{hi}]=[{local_str}]"
            )
        if n_fail > max_show:
            lines.append(f"      ... and {n_fail - max_show} more")
        return False, "\n".join(lines)
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

    Upstream reference: ``compare()`` in cann-recipes-infer ``ops/pypto_python/example/compare.py``.

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


def ratio_reldiff(
    diff_thd: float = 0.01,
    pct_thd: float = 0.05,
    max_diff_hd: float = float("inf"),
    max_show: int = 10,
) -> Callable:
    """Relative-diff comparator with bad-point ratio and single-point cap.

    Algorithm::

        a = |actual - expected|
        b = max(|actual|, |expected|, (1 / 2^14) / diff_thd) + 1e-9
        rdiff = a if a < diff_thd else a / b
        error_count = count(rdiff > diff_thd)
        pass iff error_count / numel <= pct_thd
                 AND max(rdiff over bad points) < max_diff_hd

    The denominator floor ``(1 / 2^14) / diff_thd`` keeps rdiff well-defined
    for near-zero values (capped via the ``a < diff_thd`` early-return).
    NaN / Inf in ``actual`` always fail.

    Upstream reference: ``data_compare()`` in cann-recipes-infer.

    Args:
        diff_thd: Per-point relative-difference threshold.
        pct_thd: Allowed fraction of points exceeding ``diff_thd``.
        max_diff_hd: Hard cap on worst per-point rdiff. Defaults to ``+inf``
            (no cap); pass an explicit value for a single-point catastrophic
            failure check.
        max_show: Maximum mismatched points to print on failure.
    """
    if not 0.0 < diff_thd:
        raise ValueError(f"diff_thd must be > 0, got {diff_thd}")
    if not 0.0 <= pct_thd <= 1.0:
        raise ValueError(f"pct_thd must be in [0, 1], got {pct_thd}")
    if not 0.0 < max_diff_hd:
        raise ValueError(f"max_diff_hd must be > 0, got {max_diff_hd}")

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
        actual_f = actual.cpu().to(torch.float32)
        expected_f = expected.cpu().to(torch.float32)

        nan_count = int(torch.isnan(actual_f).sum().item())
        inf_count = int(torch.isinf(actual_f).sum().item())
        if nan_count or inf_count:
            return False, (
                f"    illegal values in actual: NaN={nan_count} Inf={inf_count}"
            )

        diff_abs = (actual_f - expected_f).abs()
        small_value_floor = (1.0 / (1 << 14)) / diff_thd
        denom = torch.maximum(
            torch.maximum(actual_f.abs(), expected_f.abs()),
            torch.full_like(actual_f, small_value_floor),
        ) + 1e-9
        rdiff = torch.where(diff_abs < diff_thd, diff_abs, diff_abs / denom)

        bad_mask = rdiff > diff_thd
        error_count = int(bad_mask.sum().item())
        numel = actual_f.numel()
        pct_threshold = round(pct_thd * numel)

        # Worst single-point rdiff among bad points (0 if no bad points).
        if error_count > 0:
            worst_rdiff = float(rdiff[bad_mask].max().item())
        else:
            worst_rdiff = 0.0

        passed = (error_count <= pct_threshold) and (worst_rdiff < max_diff_hd)
        if passed:
            return True, ""

        bad_indices = torch.where(bad_mask.flatten())[0]
        flat_actual = actual_f.flatten()
        flat_expected = expected_f.flatten()
        flat_abs = diff_abs.flatten()
        flat_rdiff = rdiff.flatten()
        n_show = min(max_show, bad_indices.numel())
        idx = bad_indices[:n_show]
        lines = [
            (
                f"    [{i.item()}] actual={flat_actual[i].item():.8g}, "
                f"expected={flat_expected[i].item():.8g}, "
                f"abs_diff={flat_abs[i].item():.4g}, "
                f"rdiff={flat_rdiff[i].item():.4g}"
            )
            for i in idx
        ]
        reasons = []
        if error_count > pct_threshold:
            reasons.append(
                f"error_count={error_count}/{numel} "
                f"(ratio={error_count / numel:.4%}, allowed<={pct_thd:.4%}, "
                f"threshold={pct_threshold} pts)"
            )
        if worst_rdiff >= max_diff_hd:
            reasons.append(
                f"worst rdiff={worst_rdiff:.4g} >= max_diff_hd={max_diff_hd:.4g}"
            )
        return False, (
            f"    ratio_reldiff fail: {' AND '.join(reasons)}\n"
            f"    diff_thd={diff_thd} pct_thd={pct_thd} max_diff_hd={max_diff_hd}\n"
            f"    first {n_show} mismatches:\n" + "\n".join(lines)
        )

    cmp.__name__ = (
        f"ratio_reldiff(diff_thd={diff_thd}, pct_thd={pct_thd}, "
        f"max_diff_hd={max_diff_hd})"
    )
    return cmp


def error_distribution(
    diff_thds: tuple[float, ...] = (1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 5e-2),
    quantiles: tuple[float, ...] = (0.5, 0.9, 0.99, 0.999, 0.9999, 1.0),
    always_pass: bool = True,
) -> Callable:
    """Diagnostic comparator that prints an error-distribution report.

    This is a *measurement* comparator, not a pass/fail gate: by default it
    always returns ``True`` so a run never aborts on it, and the report is
    printed to stdout. Use it to characterize where a kernel's error lives
    before picking a real tolerance (``ratio_allclose`` / ``ratio_reldiff``).

    For the named output it prints:

    - overall rel-L2 (``||a - e|| / ||e||``) and cosine similarity — the right
      whole-tensor metrics for quantized / low-magnitude outputs, where
      per-element relative diff explodes on near-zero entries;
    - a ``frac>thd`` table over ``diff_thds`` using the same floored
      relative-diff rule as ``ratio_reldiff`` — read it as "what tolerance
      level does this output actually need", i.e. the threshold at which the
      bad-point fraction drops to your budget;
    - percentiles of the plain per-element relative diff, the absolute diff,
      and the golden magnitude — the magnitude row tells you whether a large
      relative diff is just an output "pressed low" near zero.

    Args:
        diff_thds: Threshold levels for the ``frac>thd`` sweep.
        quantiles: Quantile points for the percentile rows.
        always_pass: When ``True`` (default) the comparator never fails the
            run; set ``False`` to additionally hard-fail on NaN / Inf.

    Example — measure a layer output's error shape, gate the cache strictly::

        compare_fn = {
            "x_next": error_distribution(),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        }
    """
    qs = torch.tensor(list(quantiles))

    def cmp(actual: torch.Tensor, expected: torch.Tensor, **_kw) -> tuple[bool, str]:
        a = actual.cpu().to(torch.float32)
        e = expected.cpu().to(torch.float32)

        nan_count = int(torch.isnan(a).sum().item())
        inf_count = int(torch.isinf(a).sum().item())
        if nan_count or inf_count:
            msg = f"    illegal values in actual: NaN={nan_count} Inf={inf_count}"
            print(msg)
            if not always_pass:
                return False, msg

        diff = (a - e).abs()
        rel = (diff.norm() / e.norm().clamp_min(1e-12)).item()
        cos = torch.nn.functional.cosine_similarity(
            a.flatten(), e.flatten(), dim=0
        ).item()
        print(f"rel-L2 = {rel:.4%}  cosine = {cos:.7f}  numel={a.numel()}")

        # frac>thd: floored relative diff, same rule as ratio_reldiff.
        for thd in diff_thds:
            floor = (1.0 / (1 << 14)) / thd
            denom = torch.maximum(a.abs(), e.abs()).clamp_min(floor) + 1e-9
            rdiff = torch.where(diff < thd, diff, diff / denom)
            bad = rdiff > thd
            ec = int(bad.sum().item())
            worst = float(rdiff[bad].max().item()) if ec else 0.0
            print(
                f"  diff_thd={thd:.0e}  frac>thd={ec / a.numel():.4%}  worst={worst:.3g}"
            )

        def _pct(label: str, flat: torch.Tensor) -> None:
            flat = flat[torch.isfinite(flat)]
            if flat.numel() == 0:
                print(f"  {label}: (no finite values)")
                return
            pv = torch.quantile(flat, qs)
            print(
                f"  {label}: "
                + "  ".join(
                    f"p{q * 100:g}={v:.3g}"
                    for q, v in zip(quantiles, pv.tolist())
                )
            )

        denom = torch.maximum(a.abs(), e.abs()).clamp_min(1e-6)
        _pct("rel-diff percentiles", (diff / denom).flatten())
        _pct("abs-diff percentiles", diff.flatten())
        em = e.abs().flatten()
        print(f"  |golden| mean={em.mean():.4g}")
        _pct("|golden| percentiles", em)
        return True, ""

    cmp.__name__ = f"error_distribution(diff_thds={diff_thds})"
    return cmp
