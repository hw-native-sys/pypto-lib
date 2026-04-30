# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Declarative weight adapters: HF state_dict -> kernel-ready tensor dict.

The common case is a per-tensor spec describing:
    - where to read from (an HF key, optionally with a layer index prefix)
    - a sequence of ops to transform the value (cast / transpose / reshape / view)
    - the output key expected by the PyPTO kernel

Example (Qwen3-14B decode, layer 0):

    DictAdapter({
        "input_rms_weight":
            Map("input_layernorm.weight", ops=[View([1, HIDDEN]), Cast(torch.float32)]),
        "wq":
            Map("self_attn.q_proj.weight", ops=[Transpose(), Contiguous(), Cast(torch.bfloat16)]),
        ...
    }, prefix="model.layers.0.")
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import torch

from .base import WeightAdapter


# ---------------------------------------------------------------------------
# Ops: single-argument Callable[[Tensor], Tensor]
# ---------------------------------------------------------------------------
Op = Callable[[torch.Tensor], torch.Tensor]


def Cast(dtype: torch.dtype) -> Op:
    def _op(t: torch.Tensor) -> torch.Tensor:
        return t.to(dtype)
    _op.__name__ = f"Cast({dtype})"
    return _op


def Transpose(*dims: int) -> Op:
    """Transpose the last two dims by default; pass dims for arbitrary perm."""
    def _op(t: torch.Tensor) -> torch.Tensor:
        if not dims:
            if t.ndim < 2:
                return t
            return t.t()
        return t.permute(*dims)
    _op.__name__ = f"Transpose{dims or ''}"
    return _op


def Contiguous() -> Op:
    def _op(t: torch.Tensor) -> torch.Tensor:
        return t.contiguous()
    _op.__name__ = "Contiguous"
    return _op


def View(shape: Sequence[int]) -> Op:
    shape_t = tuple(shape)

    def _op(t: torch.Tensor) -> torch.Tensor:
        return t.view(*shape_t)
    _op.__name__ = f"View({shape_t})"
    return _op


def Reshape(shape: Sequence[int]) -> Op:
    shape_t = tuple(shape)

    def _op(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(*shape_t)
    _op.__name__ = f"Reshape({shape_t})"
    return _op


def Clone() -> Op:
    def _op(t: torch.Tensor) -> torch.Tensor:
        return t.clone()
    _op.__name__ = "Clone"
    return _op


# ---------------------------------------------------------------------------
# Mapping spec
# ---------------------------------------------------------------------------
@dataclass
class Map:
    """Read one HF key and transform it into one kernel tensor."""

    src: str
    ops: Sequence[Op] = ()

    def apply(self, hf: Mapping[str, torch.Tensor], prefix: str) -> torch.Tensor:
        key = prefix + self.src
        if key not in hf:
            raise KeyError(f"HF weight {key!r} not found (prefix={prefix!r}).")
        t = hf[key]
        for op in self.ops:
            t = op(t)
        return t


@dataclass
class Compute:
    """Compute a kernel tensor from arbitrary HF inputs (escape hatch).

    ``fn`` receives the full HF mapping and the prefix and returns a tensor.
    Use this for fused/merged weights, RoPE tables synthesized from config,
    or constants not present in the state dict.
    """

    fn: Callable[[Mapping[str, torch.Tensor], str], torch.Tensor]

    def apply(self, hf: Mapping[str, torch.Tensor], prefix: str) -> torch.Tensor:
        return self.fn(hf, prefix)


Spec = Map | Compute


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------
@dataclass
class DictAdapter(WeightAdapter):
    """Declarative adapter: a dict of ``kernel_key -> Map/Compute`` entries."""

    mapping: dict[str, Spec]
    prefix: str = ""
    # Optional extras (e.g. zero-filled output buffers) merged as-is.
    extras: dict[str, Callable[[], torch.Tensor]] = field(default_factory=dict)

    def adapt(self, hf_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for key, spec in self.mapping.items():
            out[key] = spec.apply(hf_state, self.prefix)
        for key, factory in self.extras.items():
            out[key] = factory()
        return out

    def unadapt(self, kernel_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {}


@dataclass
class PassthroughAdapter(WeightAdapter):
    """Forward the raw HF state to the target unchanged."""

    clone_tensors: bool = False

    def adapt(self, hf_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.clone_tensors:
            return {k: v.clone() for k, v in hf_state.items()}
        return dict(hf_state)

    def unadapt(self, kernel_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return dict(kernel_state)


# ---------------------------------------------------------------------------
# HF safetensors loading helper
# ---------------------------------------------------------------------------
def load_hf_state(model_path: str, key_filter: Callable[[str], bool] | None = None) -> dict[str, torch.Tensor]:
    """Load safetensors shards from ``model_path`` and return a state dict.

    ``key_filter`` lets callers load only a subset (e.g. one layer).
    """
    import json
    from pathlib import Path

    from safetensors import safe_open

    idx_path = Path(model_path) / "model.safetensors.index.json"
    if idx_path.exists():
        with open(idx_path) as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]
        keys = [k for k in weight_map if key_filter is None or key_filter(k)]
        file_to_keys: dict[str, list[str]] = {}
        for k in keys:
            file_to_keys.setdefault(weight_map[k], []).append(k)
        out: dict[str, torch.Tensor] = {}
        for fname, subkeys in sorted(file_to_keys.items()):
            path = Path(model_path) / fname
            with safe_open(str(path), framework="pt", device="cpu") as f:
                for k in subkeys:
                    out[k] = f.get_tensor(k)
        return out

    # Single-file fallback.
    path = Path(model_path) / "model.safetensors"
    out = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for k in f.keys():
            if key_filter is None or key_filter(k):
                out[k] = f.get_tensor(k)
    return out
