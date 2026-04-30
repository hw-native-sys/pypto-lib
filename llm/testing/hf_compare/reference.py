# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reference-side wrappers.

Two flavors are provided:

    - ``CallableReference``: wrap any function ``(inputs, hf_state) -> dict``.
      Zero dependencies beyond torch. Suitable for custom golden refs.

    - ``HFModuleReference``: wrap a Hugging Face ``nn.Module`` (or factory for
      one), handle ``.load_state_dict`` with prefix stripping, run forward in
      a chosen dtype, and return selected outputs. Requires ``transformers``
      at call time (imported lazily).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import torch

from .base import ReferenceModel


# ---------------------------------------------------------------------------
# Callable reference (the most flexible option)
# ---------------------------------------------------------------------------
@dataclass
class CallableReference(ReferenceModel):
    """Golden reference defined by a plain Python function.

    The forward function receives the InputSpec-materialized tensors plus the
    raw HF state dict, and returns a dict of named output tensors.
    """

    name: str
    fn: Callable[[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]], dict[str, torch.Tensor]]
    _state: Mapping[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def prepare(self, hf_state: Mapping[str, torch.Tensor]) -> None:
        self._state = hf_state

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.fn(inputs, self._state)


# ---------------------------------------------------------------------------
# Hugging Face module reference
# ---------------------------------------------------------------------------
@dataclass
class HFModuleReference(ReferenceModel):
    """Wrap a ``torch.nn.Module`` loaded with HF weights.

    Parameters:
        name: report label (e.g. ``"hf.Qwen3DecoderLayer"``).
        module_factory: callable that returns the uninitialized module. The
            caller decides how to build it (usually from ``AutoConfig`` +
            class constructor).
        forward_fn: callable ``(module, inputs) -> dict`` that invokes the
            module and extracts named output tensors. This is the only place
            model-specific plumbing lives.
        state_dict_prefix: prefix to strip from HF keys before
            ``load_state_dict`` (e.g. ``"model.layers.0."``).
        dtype: compute dtype (default fp32 for a tight reference).
    """

    name: str
    module_factory: Callable[[], torch.nn.Module]
    forward_fn: Callable[[torch.nn.Module, Mapping[str, torch.Tensor]], dict[str, torch.Tensor]]
    state_dict_prefix: str = ""
    dtype: torch.dtype = torch.float32
    strict_load: bool = True

    _module: torch.nn.Module | None = field(default=None, init=False, repr=False)

    def prepare(self, hf_state: Mapping[str, torch.Tensor]) -> None:
        module = self.module_factory().to(self.dtype).eval()
        sd: dict[str, torch.Tensor] = {}
        prefix = self.state_dict_prefix
        for k, v in hf_state.items():
            if prefix and not k.startswith(prefix):
                continue
            stripped = k[len(prefix):] if prefix else k
            sd[stripped] = v.to(self.dtype)
        missing, unexpected = module.load_state_dict(sd, strict=False)
        if self.strict_load:
            if missing:
                raise RuntimeError(f"[{self.name}] missing HF keys: {missing}")
            if unexpected:
                raise RuntimeError(f"[{self.name}] unexpected HF keys: {unexpected}")
        self._module = module

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self._module is None:
            raise RuntimeError(f"[{self.name}] prepare() must be called first.")
        with torch.no_grad():
            return self.forward_fn(self._module, inputs)
