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


def _valid_prefix(
    actual: torch.Tensor,
    expected: torch.Tensor,
    valid_rows: int | None,
    valid_axis: int,
    zero_tail: bool,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Restrict a comparison to the leading ``valid_rows`` along ``valid_axis``.

    Packed kernels carry a padded token buffer where only the leading rows are
    active; the inactive tail would otherwise dilute a ratio-based verdict.
    Returns ``(actual, expected, error)``; ``error`` is non-empty only when
    ``zero_tail`` is set and the dropped tail is not all zeros.
    """
    if valid_rows is None:
        return actual, expected, ""
    total = actual.shape[valid_axis]
    if not 0 <= valid_rows <= total:
        return actual, expected, (
            f"    valid_rows={valid_rows} out of range for axis {valid_axis} of length {total}"
        )
    if zero_tail:
        tail = actual.narrow(valid_axis, valid_rows, total - valid_rows)
        tail_nonzero = int(tail.count_nonzero().item())
        if tail_nonzero:
            return actual, expected, (
                f"    inactive tail contains {tail_nonzero} nonzero values"
            )
    return (
        actual.narrow(valid_axis, 0, valid_rows),
        expected.narrow(valid_axis, 0, valid_rows),
        "",
    )


def _nonfinite_error(actual: torch.Tensor, expected: torch.Tensor) -> str:
    """Describe non-finite values on either side of a comparison."""
    actual_nan_count = int(torch.isnan(actual).sum().item())
    actual_inf_count = int(torch.isinf(actual).sum().item())
    expected_nan_count = int(torch.isnan(expected).sum().item())
    expected_inf_count = int(torch.isinf(expected).sum().item())
    if not (
        actual_nan_count
        or actual_inf_count
        or expected_nan_count
        or expected_inf_count
    ):
        return ""
    return (
        "    illegal values in comparison: "
        f"actual: NaN={actual_nan_count} Inf={actual_inf_count}; "
        f"expected: NaN={expected_nan_count} Inf={expected_inf_count}"
    )


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
    valid_rows: int | None = None,
    valid_axis: int = 0,
    zero_tail: bool = False,
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

    NaN / Inf in ``actual`` or ``expected`` always fail (hard check,
    independent of the ratio).

    Upstream reference: ``compare()`` in cann-recipes-infer ``ops/pypto_python/example/compare.py``.

    Args:
        atol: Absolute tolerance. If ``None``, falls back to ``validate_golden``'s atol.
        rtol: Relative tolerance. If ``None``, falls back to ``validate_golden``'s rtol.
        max_error_ratio: Fraction of points permitted to exceed tolerance
            (default 0.5%). Set to 0.0 for strict allclose semantics.
        max_show: Maximum number of mismatched points printed on failure.
        valid_rows: Compare only the leading ``valid_rows`` entries along
            ``valid_axis``. ``None`` (default) compares the whole tensor, ``0``
            compares nothing and passes. Use it for packed buffers whose
            inactive tail would otherwise dilute the error ratio.
        valid_axis: Axis ``valid_rows`` slices (default 0). Pass 1 when a
            leading rank axis precedes the token axis.
        zero_tail: Additionally require the dropped tail to be all zeros. Only
            meaningful with ``valid_rows``; catches a kernel writing past the
            active token count.

    Example — attention output with INT8 activation quant::

        compare_fn = {
            "attn_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        }

    Example — packed prefill output, active prefix only::

        compare_fn = {
            "x_out": ratio_allclose(atol=1e-4, rtol=1e-2, valid_rows=num_tokens, zero_tail=True),
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

        actual, expected, prefix_error = _valid_prefix(actual, expected, valid_rows, valid_axis, zero_tail)
        if prefix_error:
            return False, prefix_error
        if actual.numel() == 0:
            return True, ""

        actual_f = actual.cpu().to(torch.float32)
        expected_f = expected.cpu().to(torch.float32)

        nonfinite_error = _nonfinite_error(actual_f, expected_f)
        if nonfinite_error:
            return False, nonfinite_error

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
        f"ratio_allclose(atol={atol}, rtol={rtol}, max_error_ratio={max_error_ratio}, "
        f"valid_rows={valid_rows}, valid_axis={valid_axis}, zero_tail={zero_tail})"
    )
    return cmp


def _mapped_pool_compare(
    mapping_name: str,
    *,
    mapping_shape: tuple[int, ...],
    block_size: int,
    leading_rank_axis: bool,
    pool_name: str,
    mapped_compare: Callable,
    comparator_name: str,
) -> Callable:
    """Apply ``mapped_compare`` to mapped rows and preserve every other row."""
    if not mapping_shape or any(dim <= 0 for dim in mapping_shape):
        raise ValueError(f"mapping_shape must contain positive dimensions, got {mapping_shape}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if leading_rank_axis and len(mapping_shape) < 2:
        raise ValueError(
            "leading_rank_axis requires mapping_shape to include a rank axis "
            f"and at least one mapped-item axis, got {mapping_shape}"
        )
    integer_dtypes = (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    )

    def compare(actual: torch.Tensor, expected: torch.Tensor, **kwargs) -> tuple[bool, str]:
        if actual.shape != expected.shape:
            return False, (
                f"    {pool_name} shape mismatch: actual={tuple(actual.shape)} "
                f"expected={tuple(expected.shape)}"
            )

        block_axis = 2 if leading_rank_axis else 1
        minimum_rank = block_axis + 1
        if actual.ndim < minimum_rank or actual.shape[block_axis] != block_size:
            layout = (
                "[ranks, blocks, block_size, ...]"
                if leading_rank_axis
                else "[blocks, block_size, ...]"
            )
            return False, (
                f"    expected block-major {pool_name} layout {layout} with "
                f"block_size={block_size}, got {tuple(actual.shape)}"
            )

        mapping = kwargs.get("inputs", {}).get(mapping_name)
        if mapping is None:
            return False, f"    compare_fn misconfigured: missing input '{mapping_name}'"
        if mapping.dtype not in integer_dtypes:
            return False, f"    '{mapping_name}' must have an integer dtype, got {mapping.dtype}"
        if tuple(mapping.shape) != mapping_shape:
            return False, (
                f"    '{mapping_name}' must have shape {mapping_shape}, "
                f"got {tuple(mapping.shape)}"
            )

        actual_cpu = actual.cpu()
        expected_cpu = expected.cpu()
        mapping = mapping.cpu().to(torch.int64)
        if leading_rank_axis:
            rank_count = mapping_shape[0]
            if actual.shape[0] != rank_count:
                return False, (
                    f"    leading rank count of {pool_name} must be {rank_count}, "
                    f"got {actual.shape[0]}"
                )
            row_count = actual.shape[1] * block_size
            actual_rows = actual_cpu.reshape(rank_count, row_count, -1)
            expected_rows = expected_cpu.reshape(rank_count, row_count, -1)
            mapping_rows = mapping.reshape(rank_count, -1)
        else:
            rank_count = 1
            row_count = actual.shape[0] * block_size
            actual_rows = actual_cpu.reshape(1, row_count, -1)
            expected_rows = expected_cpu.reshape(1, row_count, -1)
            mapping_rows = mapping.reshape(1, -1)

        invalid_negative = mapping_rows < -1
        if invalid_negative.any().item():
            first = invalid_negative.nonzero(as_tuple=False)[0]
            rank = int(first[0].item())
            item = int(first[1].item())
            value = int(mapping_rows[rank, item].item())
            location = f"[{rank}, {item}]" if leading_rank_axis else f"[{item}]"
            return False, (
                f"    '{mapping_name}'{location}={value} is invalid; "
                "only -1 is a negative sentinel"
            )

        valid = mapping_rows >= 0
        out_of_range = valid & (mapping_rows >= row_count)
        if out_of_range.any().item():
            first = out_of_range.nonzero(as_tuple=False)[0]
            rank = int(first[0].item())
            item = int(first[1].item())
            value = int(mapping_rows[rank, item].item())
            location = f"[{rank}, {item}]" if leading_rank_axis else f"[{item}]"
            return False, (
                f"    '{mapping_name}'{location}={value} is outside "
                f"physical row range [0, {row_count})"
            )

        written_rows = torch.zeros((rank_count, row_count), dtype=torch.bool)
        for rank in range(rank_count):
            rank_mapping = mapping_rows[rank, valid[rank]]
            if rank_mapping.numel() > 1:
                unique_rows, counts = torch.unique(rank_mapping, return_counts=True)
                duplicates = counts > 1
                if duplicates.any().item():
                    duplicate_row = int(unique_rows[duplicates][0].item())
                    if leading_rank_axis:
                        return False, (
                            f"    '{mapping_name}' contains duplicate physical row "
                            f"{duplicate_row} on rank {rank}"
                        )
                    return False, (
                        f"    '{mapping_name}' contains duplicate physical row {duplicate_row}"
                    )
            written_rows[rank, rank_mapping] = True

        equal_rows = (actual_rows == expected_rows).all(dim=-1)
        stray_rows = ~written_rows & ~equal_rows
        if stray_rows.any().item():
            first = stray_rows.nonzero(as_tuple=False)[0]
            rank = int(first[0].item())
            row = int(first[1].item())
            changed_values = int(
                (actual_rows[rank, row] != expected_rows[rank, row])
                .count_nonzero()
                .item()
            )
            rank_detail = f" rank={rank}" if leading_rank_axis else ""
            return False, (
                f"    unmapped physical {pool_name} row changed:{rank_detail} "
                f"row={row} changed_values={changed_values} mapping='{mapping_name}'"
            )

        if not written_rows.any().item():
            return True, ""

        mapped_actual = actual_rows[written_rows]
        mapped_expected = expected_rows[written_rows]
        for label, rows in (("actual", mapped_actual), ("expected", mapped_expected)):
            if torch.is_floating_point(rows):
                nonfinite = ~torch.isfinite(rows)
                if nonfinite.any().item():
                    return False, (
                        f"    {label} mapped rows in {pool_name} from '{mapping_name}' "
                        f"contain {int(nonfinite.count_nonzero().item())} non-finite value(s)"
                    )

        ok, detail = mapped_compare(mapped_actual, mapped_expected, **kwargs)
        if ok:
            return True, ""
        return False, f"    mapped rows in {pool_name} from '{mapping_name}':\n{detail}"

    compare.__name__ = comparator_name
    return compare


def mapped_pool_ratio_allclose(
    mapping_name: str,
    *,
    mapping_shape: tuple[int, ...],
    block_size: int,
    leading_rank_axis: bool = False,
    pool_name: str = "pool",
    atol: float | None = None,
    rtol: float | None = None,
    max_error_ratio: float = 0.005,
) -> Callable:
    """Compare mapped rows of a block-major pool and preserve all other rows.

    ``mapping_shape`` makes the mapping contract explicit instead of relying on
    model globals captured by a caller.  A non-ranked pool has layout
    ``[blocks, block_size, ...]``.  With ``leading_rank_axis=True``, the layout
    is ``[ranks, blocks, block_size, ...]`` and duplicate mappings are checked
    independently for every rank.

    Only allocator-mapped rows use the ratio-based numerical comparison and
    floating-point finiteness checks.  Every unmapped row must remain exactly
    equal to its golden snapshot, which detects writes outside the mapping.
    """
    mapped_compare = ratio_allclose(
        atol=atol,
        rtol=rtol,
        max_error_ratio=max_error_ratio,
    )
    comparator_name = (
        f"mapped_pool_ratio_allclose(mapping={mapping_name}, shape={mapping_shape}, "
        f"block_size={block_size}, leading_rank_axis={leading_rank_axis}, "
        f"atol={atol}, rtol={rtol}, max_error_ratio={max_error_ratio})"
    )
    return _mapped_pool_compare(
        mapping_name,
        mapping_shape=mapping_shape,
        block_size=block_size,
        leading_rank_axis=leading_rank_axis,
        pool_name=pool_name,
        mapped_compare=mapped_compare,
        comparator_name=comparator_name,
    )


def ratio_reldiff(
    diff_thd: float = 0.01,
    pct_thd: float = 0.05,
    max_diff_hd: float = float("inf"),
    max_show: int = 10,
    valid_rows: int | None = None,
    valid_axis: int = 0,
    zero_tail: bool = False,
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
    NaN / Inf in ``actual`` or ``expected`` always fail.

    Upstream reference: ``data_compare()`` in cann-recipes-infer.

    Args:
        diff_thd: Per-point relative-difference threshold.
        pct_thd: Allowed fraction of points exceeding ``diff_thd``.
        max_diff_hd: Hard cap on worst per-point rdiff. Defaults to ``+inf``
            (no cap); pass an explicit value for a single-point catastrophic
            failure check.
        max_show: Maximum mismatched points to print on failure.
        valid_rows: Compare only the leading ``valid_rows`` entries along
            ``valid_axis``. ``None`` (default) compares the whole tensor, ``0``
            compares nothing and passes. Use it for packed buffers whose
            inactive tail would otherwise dilute the error ratio.
        valid_axis: Axis ``valid_rows`` slices (default 0). Pass 1 when a
            leading rank axis precedes the token axis.
        zero_tail: Additionally require the dropped tail to be all zeros. Only
            meaningful with ``valid_rows``; catches a kernel writing past the
            active token count.
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
        actual, expected, prefix_error = _valid_prefix(actual, expected, valid_rows, valid_axis, zero_tail)
        if prefix_error:
            return False, prefix_error
        if actual.numel() == 0:
            return True, ""

        actual_f = actual.cpu().to(torch.float32)
        expected_f = expected.cpu().to(torch.float32)

        nonfinite_error = _nonfinite_error(actual_f, expected_f)
        if nonfinite_error:
            return False, nonfinite_error

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
        f"ratio_reldiff(diff_thd={diff_thd}, pct_thd={pct_thd}, max_diff_hd={max_diff_hd}, "
        f"valid_rows={valid_rows}, valid_axis={valid_axis}, zero_tail={zero_tail})"
    )
    return cmp


def mapped_pool_ratio_reldiff(
    mapping_name: str,
    *,
    mapping_shape: tuple[int, ...],
    block_size: int,
    leading_rank_axis: bool = False,
    pool_name: str = "pool",
    diff_thd: float = 0.01,
    pct_thd: float = 0.05,
    max_diff_hd: float = float("inf"),
) -> Callable:
    """Compare mapped pool rows with ``ratio_reldiff`` and preserve all others."""
    mapped_compare = ratio_reldiff(
        diff_thd=diff_thd,
        pct_thd=pct_thd,
        max_diff_hd=max_diff_hd,
    )
    comparator_name = (
        f"mapped_pool_ratio_reldiff(mapping={mapping_name}, shape={mapping_shape}, "
        f"block_size={block_size}, leading_rank_axis={leading_rank_axis}, "
        f"diff_thd={diff_thd}, pct_thd={pct_thd}, max_diff_hd={max_diff_hd})"
    )
    return _mapped_pool_compare(
        mapping_name,
        mapping_shape=mapping_shape,
        block_size=block_size,
        leading_rank_axis=leading_rank_axis,
        pool_name=pool_name,
        mapped_compare=mapped_compare,
        comparator_name=comparator_name,
    )


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
