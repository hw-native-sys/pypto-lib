# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runner for compiling and executing PyPTO programs with golden validation.

Public entry point: :func:`run`.  Builds tensors from :class:`TensorSpec`,
compiles the program via :func:`pypto.ir.compile`, executes on device,
computes the golden reference, and validates with :func:`validate_golden`.

:class:`RunConfig` carries two free-form kwarg dicts that are forwarded
verbatim to pypto:

- ``compile`` → :func:`pypto.ir.compile` kwargs.
- ``runtime`` → :class:`pypto.runtime.RunConfig` kwargs for the compiled callable.

Any field accepted by pypto is available without adding glue here.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .tensor_spec import TensorSpec
from .validation import validate_golden


@dataclass
class RunConfig:
    """Harness-level configuration for :func:`run`.

    Attributes:
        rtol: Relative tolerance for golden comparison.
        atol: Absolute tolerance for golden comparison.
        compile_only: If ``True``, stop after code generation without
            executing on device or validating against golden.
        compile: Kwargs forwarded to :func:`pypto.ir.compile` (e.g.
            ``backend_type``, ``dump_passes``, ``output_dir``, ``strategy``).
            When ``backend_type`` is not set and ``runtime['platform']`` is,
            :func:`run` fills it in by inferring from the platform prefix
            (``a5*`` → Ascend950, otherwise Ascend910B).
        runtime: Kwargs forwarded to :class:`pypto.runtime.RunConfig` for
            the compiled callable (e.g. ``platform``, ``device_id``,
            ``runtime_profiling``, ``pto_isa_commit``).
    """

    __test__ = False  # Not a pytest test class

    rtol: float = 1e-5
    atol: float = 1e-5
    compile_only: bool = False
    compile: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Result of a :func:`run` invocation."""

    __test__ = False  # Not a pytest test class

    passed: bool
    error: str | None = None
    execution_time: float | None = None

    def __str__(self) -> str:
        time_str = f" ({self.execution_time:.2f}s)" if self.execution_time is not None else ""
        if self.passed:
            return "PASS" + time_str
        msg = "FAIL"
        if self.error:
            msg += f": {self.error}"
        return msg + time_str


def _save_tensors(dest_dir: Path, tensors: dict[str, torch.Tensor]) -> None:
    """Save a ``{name: tensor}`` dict as ``dest_dir/{name}.pt``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name, tensor in tensors.items():
        torch.save(tensor, dest_dir / f"{name}.pt")


def run(
    program: Any,
    tensor_specs: list[TensorSpec],
    golden_fn: Callable | None = None,
    config: RunConfig | None = None,
) -> RunResult:
    """Compile *program*, run on device, and optionally validate against *golden_fn*.

    Args:
        program: A ``@pl.program`` decorated class or an ``ir.Program``.
        tensor_specs: Ordered list of tensor specifications matching the
            orchestration function's parameter order.
        golden_fn: Optional callable ``golden_fn(tensors)`` that computes
            expected outputs in-place into its ``tensors`` argument.  When
            ``None``, :func:`run` skips golden computation and validation;
            the device is still executed and ``data/in/`` is still persisted.
        config: Run configuration.  Uses default :class:`RunConfig` if ``None``.

    Returns:
        :class:`RunResult` with ``passed=True`` on success, or ``passed=False``
        with an ``error`` message on failure.
    """
    from pypto import ir  # noqa: PLC0415
    from pypto.backend import BackendType  # noqa: PLC0415
    from pypto.runtime import RunConfig as PyPTORunConfig  # noqa: PLC0415

    if config is None:
        config = RunConfig()

    start = time.time()

    # Compile
    compile_kwargs = dict(config.compile)
    platform = config.runtime.get("platform")
    if platform is not None and "backend_type" not in compile_kwargs:
        compile_kwargs["backend_type"] = (
            BackendType.Ascend950 if platform.startswith("a5") else BackendType.Ascend910B
        )
    compiled = ir.compile(program, **compile_kwargs)

    if config.compile_only:
        return RunResult(passed=True, execution_time=time.time() - start)

    # Generate Inputs
    tensors = {spec.name: spec.create_tensor() for spec in tensor_specs}
    input_snapshot = {
        spec.name: tensors[spec.name].clone()
        for spec in tensor_specs
        if not spec.is_output or spec.init_value is not None
    }
    _save_tensors(compiled.output_dir / "data" / "in", input_snapshot)

    # Runtime
    ordered = [tensors[spec.name] for spec in tensor_specs]
    compiled(*ordered, config=PyPTORunConfig(**config.runtime))

    if golden_fn is None:
        return RunResult(passed=True, execution_time=time.time() - start)

    device_outputs = {spec.name: tensors[spec.name] for spec in tensor_specs if spec.is_output}

    # Compute Golden
    scratch: dict[str, torch.Tensor] = {}
    for spec in tensor_specs:
        if spec.is_output and spec.init_value is None:
            scratch[spec.name] = torch.zeros(spec.shape, dtype=spec.dtype)
        else:
            scratch[spec.name] = input_snapshot[spec.name].clone()
    golden_fn(scratch)
    golden_outputs = {spec.name: scratch[spec.name] for spec in tensor_specs if spec.is_output}
    _save_tensors(compiled.output_dir / "data" / "out", golden_outputs)

    # Validate
    try:
        validate_golden(device_outputs, golden_outputs, rtol=config.rtol, atol=config.atol)
        return RunResult(passed=True, execution_time=time.time() - start)
    except AssertionError as e:
        return RunResult(passed=False, error=str(e), execution_time=time.time() - start)
