# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Target-side wrappers: the system under test.

Three flavors:

    - ``CallableTarget``:    any ``(inputs, weights) -> dict`` function
                             (useful for CPU PyTorch goldens inside pypto-lib).
    - ``PyPTOKernelTarget``: compile a pypto program once and execute it for
                             each run; maps input+weight dicts onto the
                             kernel's positional argument order.
    - ``TorchTarget``:       wrap an arbitrary ``nn.Module`` built from HF
                             weights (used as a sanity-check of the harness).
"""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .base import TargetModel


# ---------------------------------------------------------------------------
# Callable target
# ---------------------------------------------------------------------------
@dataclass
class CallableTarget(TargetModel):
    name: str
    fn: Callable[[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]], dict[str, torch.Tensor]]

    def prepare(self) -> None:
        return None

    def run(
        self,
        inputs: Mapping[str, torch.Tensor],
        weights: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return self.fn(inputs, weights)

    def teardown(self) -> None:
        return None


# ---------------------------------------------------------------------------
# PyPTO kernel target
# ---------------------------------------------------------------------------
_BACKEND_MAP: dict[str, str] = {
    # platform arg string -> pypto.backend.BackendType attribute name
    "a2a3": "Ascend910B",
    "a2a3sim": "Ascend910B",
    "a5": "Ascend950",
    "a5sim": "Ascend950",
}


def _resolve_device_id(device_id: int | str | None) -> int:
    if device_id is not None:
        return int(device_id)

    for env_name in ("ASCEND_DEVICE_ID", "DEVICE_ID"):
        env_value = os.environ.get(env_name)
        if env_value is not None and env_value != "":
            return int(env_value)
    return 0


@dataclass
class PyPTOKernelTarget(TargetModel):
    """Compile a pypto program once and execute it per run.

    Args:
        name: report label.
        build_program: callable returning the program IR (no args).
        spec_order: ordered list of tensor names matching the kernel signature.
        platform: one of ``a2a3 / a2a3sim / a5 / a5sim``.
        device_id: NPU device index. If omitted, resolve from
            ``ASCEND_DEVICE_ID`` / ``DEVICE_ID`` and fall back to ``0``.
        output_keys: names of tensors to expose in the returned dict. Default
            is all names in ``spec_order`` (so callers can read mutated
            buffers like ``out`` / ``k_cache`` / ``v_cache``).
        post_run: optional hook ``(tensors) -> dict`` that derives additional
            named outputs from the mutated tensor dict (e.g. gather KV rows).
    """

    name: str
    build_program: Callable[[], Any]
    spec_order: Sequence[str]
    platform: str = "a2a3"
    device_id: int | str | None = None
    output_keys: Sequence[str] | None = None
    post_run: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None

    _compiled_dir: str | None = field(default=None, init=False, repr=False)
    _execute_fn: Callable[..., None] | None = field(default=None, init=False, repr=False)

    def prepare(self) -> None:
        if self._compiled_dir is not None:
            return
        from pypto import ir
        from pypto.backend import BackendType
        from pypto.runtime import execute_compiled

        backend_name = _BACKEND_MAP.get(self.platform)
        if backend_name is None:
            raise ValueError(f"Unknown platform {self.platform!r}; known: {sorted(_BACKEND_MAP)}")
        backend = getattr(BackendType, backend_name)

        program = self.build_program()
        compiled = ir.compile(program, backend_type=backend)
        self._compiled_dir = compiled.output_dir
        self._execute_fn = execute_compiled

    def run(
        self,
        inputs: Mapping[str, torch.Tensor],
        weights: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if self._compiled_dir is None or self._execute_fn is None:
            raise RuntimeError(f"[{self.name}] prepare() must be called first.")

        merged: dict[str, torch.Tensor] = {**weights, **inputs}
        missing = [n for n in self.spec_order if n not in merged]
        if missing:
            raise KeyError(f"[{self.name}] missing tensors for spec_order: {missing}")
        ordered = [merged[n] for n in self.spec_order]

        self._execute_fn(
            self._compiled_dir,
            ordered,
            platform=self.platform,
            device_id=_resolve_device_id(self.device_id),
        )

        keys = list(self.output_keys) if self.output_keys is not None else list(self.spec_order)
        out: dict[str, torch.Tensor] = {k: merged[k] for k in keys if k in merged}
        if self.post_run is not None:
            out.update(self.post_run(merged))
        return out

    def teardown(self) -> None:
        # Compilation artifacts live on disk; keep them for reuse.
        return None


# ---------------------------------------------------------------------------
# Torch module target (sanity check / pure-CPU harness)
# ---------------------------------------------------------------------------
@dataclass
class TorchTarget(TargetModel):
    name: str
    module_factory: Callable[[], torch.nn.Module]
    forward_fn: Callable[[torch.nn.Module, Mapping[str, torch.Tensor]], dict[str, torch.Tensor]]
    state_dict_prefix: str = ""
    dtype: torch.dtype = torch.float32
    strict_load: bool = True

    _module: torch.nn.Module | None = field(default=None, init=False, repr=False)

    def prepare(self) -> None:
        return None

    def run(
        self,
        inputs: Mapping[str, torch.Tensor],
        weights: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if self._module is None:
            module = self.module_factory().to(self.dtype).eval()
            prefix = self.state_dict_prefix
            sd: dict[str, torch.Tensor] = {}
            for k, v in weights.items():
                if prefix and not k.startswith(prefix):
                    continue
                stripped = k[len(prefix):] if prefix else k
                sd[stripped] = v.to(self.dtype)
            missing, unexpected = module.load_state_dict(sd, strict=False)
            if self.strict_load:
                if missing:
                    raise RuntimeError(f"[{self.name}] missing keys: {missing}")
                if unexpected:
                    raise RuntimeError(f"[{self.name}] unexpected keys: {unexpected}")
            self._module = module
        with torch.no_grad():
            return self.forward_fn(self._module, inputs)

    def teardown(self) -> None:
        self._module = None
