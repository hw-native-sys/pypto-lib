"""Comparison helpers for CPU goldens and vLLM A2/A3 intermediate dumps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class ErrorMetrics:
    max_abs: float
    mean_abs: float
    relative_l2: float
    cosine: float
    exact: bool
    within_tolerance: bool = True


def compare_tensors(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float = 5.0e-2,
    atol: float = 5.0e-2,
    exact_required: bool = False,
) -> ErrorMetrics:
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: actual={tuple(actual.shape)}, expected={tuple(expected.shape)}")
    lhs = actual.detach().float().cpu().reshape(-1)
    rhs = expected.detach().float().cpu().reshape(-1)
    delta = lhs - rhs
    denominator = rhs.norm().clamp_min(torch.finfo(torch.float32).tiny)
    cosine = torch.nn.functional.cosine_similarity(lhs.unsqueeze(0), rhs.unsqueeze(0), dim=1)[0]
    exact = bool(torch.equal(actual.cpu(), expected.cpu()))
    close = exact if exact_required else bool(torch.allclose(lhs, rhs, rtol=rtol, atol=atol))
    return ErrorMetrics(
        max_abs=float(delta.abs().max().item()) if delta.numel() else 0.0,
        mean_abs=float(delta.abs().mean().item()) if delta.numel() else 0.0,
        relative_l2=float(delta.norm().div(denominator).item()),
        cosine=float(cosine.item()) if delta.numel() else 1.0,
        exact=exact,
        within_tolerance=close,
    )


def load_intermediate_dump(path: str | Path) -> dict[str, Any]:
    """Load a vLLM dump written by ``torch.save`` or a tensor directory."""

    dump_path = Path(path)
    if dump_path.is_dir():
        payload = {
            item.stem: torch.load(item, map_location="cpu", weights_only=False)
            for item in sorted(dump_path.glob("*.pt"))
        }
    else:
        payload = torch.load(dump_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"expected a tensor dictionary in {path}, got {type(payload).__name__}")


def _flatten_tensor_nodes(payload: dict[str, Any], prefix: str = "") -> dict[str, torch.Tensor]:
    """Flatten tensor leaves while preserving diagnostic group names."""

    nodes: dict[str, torch.Tensor] = {}
    for name, value in payload.items():
        key = f"{prefix}{name}"
        if isinstance(value, torch.Tensor):
            nodes[key] = value
        elif isinstance(value, dict):
            nodes.update(_flatten_tensor_nodes(value, key + "/"))
    return nodes


def _select_tensor_nodes(payload: dict[str, Any], case: str | None) -> dict[str, torch.Tensor]:
    """Select tensor nodes from either a flat dump or a vLLM ``cases`` dump."""

    if "cases" in payload:
        cases = payload["cases"]
        if not isinstance(cases, dict) or not cases:
            raise ValueError("vLLM dump contains an empty or invalid cases mapping")
        if case is None:
            if len(cases) != 1:
                raise ValueError(f"dump has multiple cases; pass --case from {sorted(cases)}")
            case = next(iter(cases))
        if case not in cases:
            raise KeyError(f"unknown dump case {case!r}; available cases: {sorted(cases)}")
        selected = cases[case]
        if not isinstance(selected, dict):
            raise TypeError(f"case {case!r} must contain a tensor dictionary")
        payload = selected.get("nodes", selected)
        if not isinstance(payload, dict):
            raise TypeError(f"case {case!r} nodes must be a tensor dictionary")
    elif isinstance(payload.get("nodes"), dict):
        # Validators save a single vLLM/PyPTO capture as {metadata, nodes}.
        # Treat both sides identically so compare_vllm_dump can compare the
        # files emitted by the per-stage NPU validators directly.
        payload = payload["nodes"]
    tensors = _flatten_tensor_nodes(payload)
    if not tensors:
        raise ValueError("dump contains no tensor nodes")
    return tensors


def compare_dump(
    actual: dict[str, torch.Tensor],
    expected_path: str | Path,
    case: str | None = None,
    *,
    rtol: float = 5.0e-2,
    atol: float = 5.0e-2,
) -> dict[str, ErrorMetrics]:
    """Compare named PyPTO outputs with a vLLM A2/A3 intermediate dump."""

    expected = _select_tensor_nodes(load_intermediate_dump(expected_path), case)
    if "cases" in actual:
        actual_nodes = _select_tensor_nodes(actual, case)
    elif isinstance(actual.get("nodes"), dict):
        actual_nodes = _flatten_tensor_nodes(actual["nodes"])
    else:
        actual_nodes = _flatten_tensor_nodes(actual)
    missing = sorted(set(expected) - set(actual_nodes))
    if missing:
        raise KeyError(f"missing PyPTO outputs: {missing}")
    results = {}
    for name in expected:
        lhs = actual_nodes[name]
        rhs = expected[name]
        # Integer and cache metadata are ABI, rather than numerical, values.
        # FP16 index-cache scales are also required byte-identical because they
        # are published scheduler state, not an intermediate approximation.
        exact_required = (
            not (lhs.is_floating_point() and rhs.is_floating_point())
            or name.endswith(("payload", "scales", "indices", "topk", "ids", "routes", "slots", "positions", "lengths"))
        )
        results[name] = compare_tensors(
            lhs,
            rhs,
            rtol=rtol,
            atol=atol,
            exact_required=exact_required,
        )
    return results


__all__ = ["ErrorMetrics", "compare_dump", "compare_tensors", "load_intermediate_dump"]
