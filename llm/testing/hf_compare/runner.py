# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Runner: drive a ComparisonCase end-to-end."""
from __future__ import annotations

from collections.abc import Mapping

import torch

from .base import (
    ChainedComparisonCase,
    CompareReport,
    ComparisonCase,
    HFWeightSource,
    OutputSelector,
    SelectorResult,
)
from .comparator import Comparator


def _resolve_hf_weights(source: HFWeightSource) -> Mapping[str, torch.Tensor]:
    if source is None:
        return {}
    if callable(source):
        return source()
    if isinstance(source, str):
        from .weight_adapter import load_hf_state
        return load_hf_state(source)
    return source


def _apply_selector(t: torch.Tensor, sel: OutputSelector) -> torch.Tensor:
    if sel.slice_ is not None:
        t = t[sel.slice_]
    if sel.postprocess is not None:
        t = sel.postprocess(t)
    return t


def run_case(case: ComparisonCase) -> CompareReport:
    hf_state = _resolve_hf_weights(case.hf_weights)

    # Materialize identical inputs for both sides. Target may mutate in place,
    # so give each side its own clone.
    raw_inputs = case.input_spec.materialize()
    if case.on_inputs is not None:
        case.on_inputs(raw_inputs)
    ref_inputs = {k: v.clone() for k, v in raw_inputs.items()}
    tgt_inputs = {k: v.clone() for k, v in raw_inputs.items()}

    # Prepare reference (HF) side.
    case.reference.prepare(hf_state)
    ref_out = case.reference.forward(ref_inputs)

    # Prepare target (PyPTO) side; adapt weights via declarative adapter.
    kernel_weights = case.weight_adapter.adapt(hf_state)
    case.target.prepare()
    try:
        tgt_out = case.target.run(tgt_inputs, kernel_weights)
    finally:
        case.target.teardown()

    # Compare each selected output.
    comparator = Comparator(case.tolerance)
    results: list[SelectorResult] = []
    for sel in case.selectors:
        if sel.ref_key not in ref_out:
            results.append(
                SelectorResult(
                    selector=sel.name,
                    metrics={},
                    passed=False,
                    note=f"ref_key {sel.ref_key!r} not in reference output (keys={sorted(ref_out)})",
                )
            )
            continue
        if sel.tgt_key not in tgt_out:
            results.append(
                SelectorResult(
                    selector=sel.name,
                    metrics={},
                    passed=False,
                    note=f"tgt_key {sel.tgt_key!r} not in target output (keys={sorted(tgt_out)})",
                )
            )
            continue
        ref_t = _apply_selector(ref_out[sel.ref_key], sel)
        tgt_t = _apply_selector(tgt_out[sel.tgt_key], sel)
        results.append(comparator.compare(ref_t, tgt_t, sel))

    passed = all(r.passed for r in results) and len(results) > 0
    report = CompareReport(
        case_name=case.name,
        passed=passed,
        results=results,
        meta={
            "reference": case.reference.name,
            "target": case.target.name,
            "num_selectors": len(case.selectors),
        },
    )
    if case.on_report is not None:
        case.on_report(report)
    return report


def run_chain(chain: ChainedComparisonCase) -> list[CompareReport]:
    reports: list[CompareReport] = []
    prev_ref_out: dict[str, torch.Tensor] = {}
    for step in chain.steps:
        if step.forward_map and prev_ref_out:
            # Inject values from the previous step's reference outputs into
            # this step's input spec by overriding materialize() via on_inputs.
            mapping = step.forward_map
            snapshot = {dst: prev_ref_out[src].clone() for src, dst in mapping.items() if src in prev_ref_out}
            original_hook = step.case.on_inputs

            def _hook(inputs: dict[str, torch.Tensor], _snap=snapshot, _orig=original_hook) -> None:
                for name, value in _snap.items():
                    inputs[name] = value
                if _orig is not None:
                    _orig(inputs)

            step.case.on_inputs = _hook
        report = run_case(step.case)
        reports.append(report)
        if not report.passed and chain.stop_on_fail:
            break
        # Re-run the reference just to capture its outputs for the next step.
        # (ComparisonCase.run did the work but did not return ref outputs, so
        # we re-materialize on the fly. For now we skip and rely on tests
        # using this chain to either keep state externally or extend the API.)
        prev_ref_out = {}
    return reports
