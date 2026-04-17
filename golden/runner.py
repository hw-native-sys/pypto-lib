# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile PyPTO programs, run them on device, and validate against goldens.

Public entry point: :func:`run`.
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
            ``backend_type``, ``dump_passes``, ``output_dir``, ``strategy``,
            ``profiling``).
        runtime: Kwargs forwarded to :func:`pypto.runtime.execute_compiled`
            (e.g. ``platform``, ``device_id``, ``runtime_profiling``).
    """

    rtol: float = 1e-5
    atol: float = 1e-5
    compile_only: bool = False
    compile: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Result of a :func:`run` invocation."""

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


def _load_tensors(src_dir: Path, subdir: str, names: list[str]) -> dict[str, torch.Tensor]:
    """Load ``src_dir/subdir/{name}.pt`` for each name."""
    return {n: torch.load(src_dir / subdir / f"{n}.pt", weights_only=True) for n in names}


def _required_files(spec: TensorSpec) -> list[tuple[str, str]]:
    """Return ``[(subdir, filename), ...]`` required for *spec* in a golden-data dir.

    - Pure input: ``in/{name}.pt``
    - Pure output: ``out/{name}.pt``
    - Inout (is_output + init_value): both ``in/{name}.pt`` and ``out/{name}.pt``
    """
    files: list[tuple[str, str]] = []
    if not spec.is_output:
        files.append(("in", f"{spec.name}.pt"))
    else:
        files.append(("out", f"{spec.name}.pt"))
        if spec.init_value is not None:
            files.append(("in", f"{spec.name}.pt"))
    return files


def _backend_for_platform(platform: str) -> Any:
    """Return the :class:`pypto.backend.BackendType` for a platform string."""
    from pypto.backend import BackendType

    mapping = {
        "a2a3": BackendType.Ascend910B,
        "a2a3sim": BackendType.Ascend910B,
        "a5": BackendType.Ascend950,
        "a5sim": BackendType.Ascend950,
    }
    try:
        return mapping[platform]
    except KeyError:
        raise ValueError(
            f"Unknown runtime platform {platform!r}; expected one of {sorted(mapping)}"
        ) from None


def run(
    program: Any,
    tensor_specs: list[TensorSpec],
    config: RunConfig | None = None,
    golden_fn: Callable | None = None,
    golden_data: str | None = None,
    runtime_dir: str | None = None,
) -> RunResult:
    """Compile *program*, run on device, and optionally validate goldens.

    Args:
        program: A ``@pl.program`` decorated class or an ``ir.Program``.
        tensor_specs: Ordered list of tensor specifications matching the
            orchestration function's parameter order.
        config: Run configuration.  Uses default :class:`RunConfig` if ``None``.
        golden_fn: Optional callable ``golden_fn(tensors)`` that computes
            expected outputs in-place.  When ``None``, golden is sourced from
            *golden_data* if set; if neither is provided, validation is skipped.
        golden_data: Optional directory with persisted ``in/{name}.pt`` and
            ``out/{name}.pt``.  When set, :func:`run` loads tensors from it
            instead of generating inputs or computing goldens (read-only).
            Takes precedence over *golden_fn* when both are provided.
        runtime_dir: Optional path to a pre-compiled build_output directory.
            When set, compilation is skipped and execution runs against this
            directory; ``config.compile`` is ignored and ``compile_only`` is
            rejected.

    Returns:
        :class:`RunResult` with ``passed=True`` on success, or ``passed=False``
        with an ``error`` message on failure.
    """
    from pypto import ir
    from pypto.runtime import execute_compiled

    if config is None:
        config = RunConfig()

    data_dir = Path(golden_data) if golden_data is not None else None

    start = time.time()

    def _stage(name: str):
        """Context manager-like helper: print begin/done around a block."""
        class _Ctx:
            def __enter__(self_):
                print(f"[RUN] {name} ...", flush=True)
                self_._t0 = time.time()
                return self_
            def __exit__(self_, *_exc):
                dt = time.time() - self_._t0
                print(f"[RUN] {name} done ({dt:.2f}s)", flush=True)
                return False
        return _Ctx()

    def _fail(error: str) -> RunResult:
        return RunResult(passed=False, error=error, execution_time=time.time() - start)

    # Compile
    if runtime_dir is not None:
        if config.compile_only:
            return _fail("runtime_dir is incompatible with config.compile_only")
        work_dir = Path(runtime_dir)
        if not work_dir.is_dir():
            return _fail(f"runtime_dir does not exist: {work_dir}")
        print(f"[RUN] runtime_only: skipping compile, using {work_dir}", flush=True)
    else:
        with _stage("compile"):
            compile_kwargs = dict(config.compile)
            platform = config.runtime.get("platform")
            if platform is not None:
                compile_kwargs.setdefault("backend_type", _backend_for_platform(platform))
            compiled = ir.compile(program, **compile_kwargs)

        if config.compile_only:
            total = time.time() - start
            print(f"[RUN] PASS ({total:.2f}s)", flush=True)
            return RunResult(passed=True, execution_time=total)

        work_dir = compiled.output_dir

    # Generate Inputs
    with _stage("generate inputs"):
        if data_dir is not None:
            missing = [
                str(data_dir / sub / name)
                for spec in tensor_specs
                for sub, name in _required_files(spec)
                if not (data_dir / sub / name).is_file()
            ]
            if missing:
                return _fail(f"golden_data is missing files: {missing}")
            print(f"[RUN]   cache hit: {data_dir / 'in'}", flush=True)
            # Load inputs + inout initial values from {dir}/in/; pure outputs stay zero-init.
            input_names = [s.name for s in tensor_specs if not s.is_output or s.init_value is not None]
            tensors = _load_tensors(data_dir, "in", input_names)
            for spec in tensor_specs:
                if spec.is_output and spec.init_value is None:
                    tensors[spec.name] = torch.zeros(spec.shape, dtype=spec.dtype)
        else:
            tensors = {spec.name: spec.create_tensor() for spec in tensor_specs}
            input_snapshot = {
                spec.name: tensors[spec.name].clone()
                for spec in tensor_specs
                if not spec.is_output or spec.init_value is not None
            }
            _save_tensors(work_dir / "data" / "in", input_snapshot)

    # Runtime
    with _stage("runtime"):
        ordered = [tensors[spec.name] for spec in tensor_specs]
        execute_compiled(work_dir, ordered, **config.runtime)

    if golden_fn is None and golden_data is None:
        total = time.time() - start
        print(f"[RUN] PASS ({total:.2f}s, validation skipped: no golden_fn or golden_data)", flush=True)
        return RunResult(passed=True, execution_time=total)

    device_outputs = {spec.name: tensors[spec.name] for spec in tensor_specs if spec.is_output}

    # Compute Golden (or load from cache)
    with _stage("compute golden"):
        if data_dir is not None:
            print(f"[RUN]   cache hit: {data_dir / 'out'}", flush=True)
            output_names = [s.name for s in tensor_specs if s.is_output]
            golden_outputs = _load_tensors(data_dir, "out", output_names)
        else:
            scratch: dict[str, torch.Tensor] = {}
            for spec in tensor_specs:
                if spec.is_output and spec.init_value is None:
                    scratch[spec.name] = torch.zeros(spec.shape, dtype=spec.dtype)
                else:
                    scratch[spec.name] = input_snapshot[spec.name].clone()
            golden_fn(scratch)
            golden_outputs = {spec.name: scratch[spec.name] for spec in tensor_specs if spec.is_output}
            _save_tensors(work_dir / "data" / "out", golden_outputs)

    # Validate
    with _stage("validate"):
        try:
            validate_golden(device_outputs, golden_outputs, rtol=config.rtol, atol=config.atol)
        except AssertionError as e:
            return _fail(str(e))

    total = time.time() - start
    print(f"[RUN] PASS ({total:.2f}s)", flush=True)
    return RunResult(passed=True, execution_time=total)
