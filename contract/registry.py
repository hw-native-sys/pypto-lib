# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""External contract registry."""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

from contract.base import ContractRegistration, ModelContract


class ContractNotImplementedError(NotImplementedError):
    """Raised when a known model contract is registered but not implemented."""


def _normalize(value: str) -> str:
    return value.lower().replace("_", "-")


@lru_cache(maxsize=1)
def _qwen3_14b_registration() -> ContractRegistration:
    root = Path(__file__).resolve().parents[1]
    variant_dir = root / "models" / "qwen3" / "14b"
    module = _load_registration_module(
        "_pypto_lib_qwen3_14b_contract",
        variant_dir,
        variant_dir / "contract.py",
    )
    return module.QWEN3_14B_REGISTRATION


def _load_registration_module(module_name: str, variant_dir: Path, module_path: Path) -> object:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load external contract from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(variant_dir))
    try:
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(variant_dir))
        except ValueError:
            pass
    return module


def _registrations() -> tuple[ContractRegistration, ...]:
    return (_qwen3_14b_registration(),)


def get_contract(model_family: str, model_variant: str | None = None) -> ModelContract:
    """Return an external model contract by explicit family and variant."""
    family = _normalize(model_family)
    variant = _normalize(model_variant or "")
    for registration in _registrations():
        if (_normalize(registration.family), _normalize(registration.variant)) == (family, variant):
            return _registration_contract(registration)
    raise KeyError(f"unsupported external contract: family={model_family!r}, variant={model_variant!r}")


def find_contract_for_model_config(model_config: object) -> ModelContract:
    """Return the external contract that matches parsed model metadata."""
    for registration in _registrations():
        if registration.matcher is not None and registration.matcher(model_config):
            return _registration_contract(registration)
    raise KeyError(
        "unsupported external contract for model config: "
        f"model_id={getattr(model_config, 'model_id', None)!r}, "
        f"architecture={getattr(model_config, 'architecture', None)!r}, "
        f"model_type={getattr(model_config, 'model_type', None)!r}"
    )


def _registration_contract(registration: ContractRegistration) -> ModelContract:
    if not registration.implemented:
        raise ContractNotImplementedError(
            "external contract is registered but not implemented: "
            f"family={registration.family!r}, variant={registration.variant!r}"
        )
    return registration.factory()
