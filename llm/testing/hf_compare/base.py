# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Core abstractions for the HF comparison framework.

A ComparisonCase bundles four pluggable pieces:
    - ReferenceModel: the HF-side callable that produces golden outputs.
    - TargetModel:    the PyPTO-side callable under test (kernel / scope / e2e).
    - WeightAdapter:  maps HF state_dict entries onto tensors the target expects.
    - InputSpec:      declarative description of the inputs fed to both sides.

The runner feeds identical inputs to both sides, collects named outputs, and
uses Comparator + Tolerance to produce a structured CompareReport.

Note: this module intentionally has zero dependencies on pypto / transformers /
safetensors. Concrete adapters and targets that need them live in sibling
modules and are imported on demand.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch


# ---------------------------------------------------------------------------
# Tensor I/O descriptors
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TensorSpec:
    """Shape + dtype + (optional) value generator for one input tensor."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    # How to synthesize values. None -> zeros.
    sampler: Callable[[tuple[int, ...], torch.dtype, torch.Generator], torch.Tensor] | None = None
    # Optional semantic tag so adapters/selectors can identify the tensor
    # without relying on its dict key (e.g. "kv_cache", "rope_table").
    role: str | None = None


@dataclass
class InputSpec:
    """Declarative input bundle, evaluated once and shared by ref & target."""

    tensors: dict[str, TensorSpec]
    seed: int = 0

    def materialize(self, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        out: dict[str, torch.Tensor] = {}
        for name, spec in self.tensors.items():
            if spec.sampler is not None:
                t = spec.sampler(spec.shape, spec.dtype, gen)
            else:
                t = torch.zeros(spec.shape, dtype=spec.dtype)
            out[name] = t.to(device) if str(device) != "cpu" else t
        return out


# ---------------------------------------------------------------------------
# Weight adapter
# ---------------------------------------------------------------------------
@runtime_checkable
class WeightAdapter(Protocol):
    """Transforms an HF state_dict slice into kernel-ready tensors.

    Implementations can be declarative (see weight_adapter.DictAdapter) or
    fully custom for fused / sharded / MoE-style layouts.
    """

    def adapt(self, hf_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...

    # Optional inverse direction: rebuild an HF-shaped tensor from kernel
    # buffers. Useful when the target owns KV cache or fused weights and the
    # reference needs them in HF layout.
    def unadapt(self, kernel_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:  # noqa: ARG002
        return {}


# ---------------------------------------------------------------------------
# Reference (HF side)
# ---------------------------------------------------------------------------
@runtime_checkable
class ReferenceModel(Protocol):
    """Golden-reference callable.

    The returned dict's keys become the right-hand side of OutputSelector
    pairs. A reference can wrap any of:
        - an HF nn.Module (e.g. Qwen3DecoderLayer)
        - a subset of one (e.g. just attention scope1)
        - any torch function (for new ops with no HF counterpart)
    """

    name: str

    def prepare(self, hf_state: Mapping[str, torch.Tensor]) -> None: ...

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...


# ---------------------------------------------------------------------------
# Target (PyPTO side)
# ---------------------------------------------------------------------------
@runtime_checkable
class TargetModel(Protocol):
    """The system under test.

    ``prepare()`` is called once and may compile a pypto program (results
    should be cached so re-running the same case is cheap). ``run()`` is
    invoked per case and returns the dict to be compared.
    """

    name: str

    def prepare(self) -> None: ...

    def run(
        self,
        inputs: Mapping[str, torch.Tensor],
        weights: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]: ...

    def teardown(self) -> None: ...


# ---------------------------------------------------------------------------
# Output selection & tolerance
# ---------------------------------------------------------------------------
SliceExpr = Any
"""A slice, int, tuple of them, or ``Ellipsis`` for full-range selection."""


@dataclass(frozen=True)
class OutputSelector:
    """Which output(s) to compare, plus optional reshape/slice."""

    name: str                              # report label, e.g. "layer_out"
    ref_key: str                           # key in ReferenceModel.forward() output
    tgt_key: str                           # key in TargetModel.run() output
    slice_: SliceExpr | None = None        # applied to BOTH tensors before compare
    cast_to: torch.dtype = torch.float32   # cast before computing metrics
    # Optional post-processing hook applied AFTER slicing (e.g. gather written
    # KV rows or transpose to a common layout). Run independently on ref/tgt.
    postprocess: Callable[[torch.Tensor], torch.Tensor] | None = None


@dataclass(frozen=True)
class Tolerance:
    atol: float = 5e-3
    rtol: float = 5e-3
    # Metrics to compute and include in the report. Unknown names ignored.
    metrics: tuple[str, ...] = ("max_abs", "mean_abs", "max_rel", "cosine", "pass_rate")
    # pass_rate: fraction of elements satisfying |a-b| <= atol + rtol*|b|.
    pass_rate_threshold: float = 0.999
    # How many worst-offender entries to record per selector.
    worst_offenders: int = 5


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
@dataclass
class SelectorResult:
    selector: str
    metrics: dict[str, float]
    passed: bool
    # (flat_index, ref_val, tgt_val) tuples for the largest absolute diffs.
    worst_offenders: list[tuple[int, float, float]] = field(default_factory=list)
    # Optional free-form note (e.g. shape mismatch, cast applied, ...).
    note: str = ""


@dataclass
class CompareReport:
    case_name: str
    passed: bool
    results: list[SelectorResult]
    # Free-form diagnostics (compile time, device, versions, env, ...).
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "case": self.case_name,
            "passed": self.passed,
            "results": [
                {
                    "selector": r.selector,
                    "passed": r.passed,
                    "metrics": r.metrics,
                    "worst_offenders": r.worst_offenders,
                    "note": r.note,
                }
                for r in self.results
            ],
            "meta": self.meta,
        }

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"[{status}] case={self.case_name}"]
        for r in self.results:
            tag = "PASS" if r.passed else "FAIL"
            metric_str = " ".join(f"{k}={v:.4e}" for k, v in r.metrics.items())
            lines.append(f"  [{tag}] {r.selector:24s} {metric_str}")
            if r.note:
                lines.append(f"         note: {r.note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparison case (top-level)
# ---------------------------------------------------------------------------
# HF weight source: a path on disk, a preloaded state dict, or a callable
# returning one. Concrete loaders live in ``weight_adapter`` / case modules.
HFWeightSource = (
    str
    | Mapping[str, torch.Tensor]
    | Callable[[], Mapping[str, torch.Tensor]]
    | None
)


@dataclass
class ComparisonCase:
    name: str
    reference: ReferenceModel
    target: TargetModel
    input_spec: InputSpec
    weight_adapter: WeightAdapter
    selectors: list[OutputSelector]
    tolerance: Tolerance = field(default_factory=Tolerance)
    hf_weights: HFWeightSource = None
    # Optional pre/post hooks for diagnostics or in-place tweaks.
    on_inputs: Callable[[dict[str, torch.Tensor]], None] | None = None
    on_report: Callable[[CompareReport], None] | None = None

    def run(self) -> CompareReport:
        """Drive the comparison. Implementation lives in runner.py."""
        # Imported lazily to keep base.py free of heavy dependencies.
        from .runner import run_case

        return run_case(self)


# ---------------------------------------------------------------------------
# Chained cases (scope1 -> scope2 -> scope3)
# ---------------------------------------------------------------------------
@dataclass
class ChainStep:
    case: ComparisonCase
    # Map produced reference outputs onto the next step's InputSpec tensor
    # names. ``{ref_output_key: next_input_name}``.
    forward_map: dict[str, str] = field(default_factory=dict)


@dataclass
class ChainedComparisonCase:
    """Run multiple cases where step N's reference outputs feed step N+1."""

    name: str
    steps: list[ChainStep]
    stop_on_fail: bool = True

    def run(self) -> list[CompareReport]:
        from .runner import run_chain

        return run_chain(self)


# ---------------------------------------------------------------------------
# Registry (so CLI can discover cases by name)
# ---------------------------------------------------------------------------
_CASE_REGISTRY: dict[str, Callable[..., ComparisonCase]] = {}


def register_case(
    name: str,
) -> Callable[[Callable[..., ComparisonCase]], Callable[..., ComparisonCase]]:
    """Decorator: ``@register_case("qwen3_14b.decode")`` on a factory function.

    The decorated function should return a fully-constructed ComparisonCase
    when called (typically with optional kwargs like ``hf_model_path=...``).
    """

    def deco(fn: Callable[..., ComparisonCase]) -> Callable[..., ComparisonCase]:
        if name in _CASE_REGISTRY:
            raise ValueError(f"Case {name!r} already registered.")
        _CASE_REGISTRY[name] = fn
        return fn

    return deco


def _autodiscover_cases() -> None:
    """Import every module in ``llm.testing.hf_compare.cases`` to register cases."""
    import importlib
    import pkgutil

    try:
        cases_pkg = importlib.import_module("llm.testing.hf_compare.cases")
    except ImportError:
        return
    for modinfo in pkgutil.iter_modules(cases_pkg.__path__):
        importlib.import_module(f"llm.testing.hf_compare.cases.{modinfo.name}")


def get_case(name: str, **kwargs: Any) -> ComparisonCase:
    if name not in _CASE_REGISTRY:
        _autodiscover_cases()
    if name not in _CASE_REGISTRY:
        raise KeyError(
            f"Unknown case: {name!r}. Registered: {sorted(_CASE_REGISTRY)}"
        )
    return _CASE_REGISTRY[name](**kwargs)


def list_cases() -> list[str]:
    _autodiscover_cases()
    return sorted(_CASE_REGISTRY)
