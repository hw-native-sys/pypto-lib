# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tensor comparison metrics and the pass/fail decision."""
from __future__ import annotations

import torch

from .base import OutputSelector, SelectorResult, Tolerance


class Comparator:
    """Computes metrics and pass/fail for a single (ref, tgt) tensor pair."""

    def __init__(self, tolerance: Tolerance):
        self.tolerance = tolerance

    def compare(
        self,
        ref: torch.Tensor,
        tgt: torch.Tensor,
        selector: OutputSelector,
    ) -> SelectorResult:
        note = ""
        if ref.shape != tgt.shape:
            return SelectorResult(
                selector=selector.name,
                metrics={},
                passed=False,
                note=f"shape mismatch ref={tuple(ref.shape)} tgt={tuple(tgt.shape)}",
            )

        ref_f = ref.detach().to(selector.cast_to).reshape(-1)
        tgt_f = tgt.detach().to(selector.cast_to).reshape(-1)
        diff = (ref_f - tgt_f).abs()

        metrics: dict[str, float] = {}
        requested = set(self.tolerance.metrics)

        if "max_abs" in requested:
            metrics["max_abs"] = float(diff.max().item()) if diff.numel() else 0.0
        if "mean_abs" in requested:
            metrics["mean_abs"] = float(diff.mean().item()) if diff.numel() else 0.0
        if "max_rel" in requested:
            denom = ref_f.abs().clamp(min=1e-6)
            metrics["max_rel"] = float((diff / denom).max().item()) if diff.numel() else 0.0
        if "cosine" in requested:
            metrics["cosine"] = _cosine(ref_f, tgt_f)

        threshold = self.tolerance.atol + self.tolerance.rtol * ref_f.abs()
        within = diff <= threshold
        pass_rate = float(within.float().mean().item()) if diff.numel() else 1.0
        if "pass_rate" in requested:
            metrics["pass_rate"] = pass_rate

        passed = pass_rate >= self.tolerance.pass_rate_threshold

        offenders: list[tuple[int, float, float]] = []
        if not passed and self.tolerance.worst_offenders > 0 and diff.numel() > 0:
            k = min(self.tolerance.worst_offenders, diff.numel())
            top = torch.topk(diff, k)
            for idx in top.indices.tolist():
                offenders.append((int(idx), float(ref_f[idx].item()), float(tgt_f[idx].item())))

        return SelectorResult(
            selector=selector.name,
            metrics=metrics,
            passed=passed,
            worst_offenders=offenders,
            note=note,
        )


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return 1.0
    na = a.norm()
    nb = b.norm()
    if na.item() == 0.0 or nb.item() == 0.0:
        # Degenerate: define cosine as 1.0 iff both zero, else 0.0.
        return 1.0 if (na.item() == 0.0 and nb.item() == 0.0) else 0.0
    return float((a @ b / (na * nb)).item())
