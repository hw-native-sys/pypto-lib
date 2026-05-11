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

from .spec import ScalarSpec, TensorSpec
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
        compare_fn: Per-output-name custom comparators that override
            ``torch.allclose`` for those tensors. See
            :func:`golden.validation.validate_golden` for the callable
            signature, and :func:`golden.validation.topk_pair_compare` for
            a built-in helper covering top-k index/value outputs.
    """

    rtol: float = 1e-5
    atol: float = 1e-5
    compile_only: bool = False
    compile: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    compare_fn: dict[str, Callable] = field(default_factory=dict)


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


def _required_files(spec: TensorSpec | ScalarSpec) -> list[tuple[str, str]]:
    """Return ``[(subdir, filename), ...]`` required for *spec* in a golden-data dir.

    - :class:`ScalarSpec`: ``in/{name}.pt`` (the 0-dim
      :attr:`ScalarSpec.value` tensor).
    - :class:`TensorSpec` pure input: ``in/{name}.pt``.
    - :class:`TensorSpec` pure output: ``out/{name}.pt``.
    - :class:`TensorSpec` inout (``is_output`` + ``init_value``):
      both ``in/{name}.pt`` and ``out/{name}.pt``.
    """
    if isinstance(spec, ScalarSpec):
        return [("in", f"{spec.name}.pt")]
    files: list[tuple[str, str]] = []
    if not spec.is_output:
        files.append(("in", f"{spec.name}.pt"))
    else:
        files.append(("out", f"{spec.name}.pt"))
        if spec.init_value is not None:
            files.append(("in", f"{spec.name}.pt"))
    return files


class _Stage:
    """Context manager: print begin/done around a stage block."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._t0 = 0.0

    def __enter__(self) -> "_Stage":
        print(f"[RUN] {self._name} ...", flush=True)
        self._t0 = time.time()
        return self

    def __exit__(self, *_exc: Any) -> bool:
        dt = time.time() - self._t0
        print(f"[RUN] {self._name} done ({dt:.2f}s)", flush=True)
        return False


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
    specs: list[TensorSpec | ScalarSpec],
    config: RunConfig | None = None,
    golden_fn: Callable | None = None,
    golden_data: str | None = None,
    runtime_dir: str | None = None,
) -> RunResult:
    """Compile *program*, run on device, and optionally validate goldens.

    Args:
        program: A ``@pl.program`` decorated class or an ``ir.Program``.
        specs: Ordered list of :class:`TensorSpec` and :class:`ScalarSpec`
            entries matching the orchestration function's parameter order.
        config: Run configuration.  Uses default :class:`RunConfig` if ``None``.
        golden_fn: Optional callable ``golden_fn(values)`` that computes
            expected outputs in-place.  ``values`` is a dict containing both
            tensor clones and scalar Python values keyed by spec name.  When
            ``None``, golden is sourced from *golden_data* if set; if neither
            is provided, validation is skipped.
        golden_data: Optional directory with persisted ``in/{name}.pt`` and
            ``out/{name}.pt`` files (scalars are stored as 0-dim tensors in
            the same format).  When set, :func:`run` loads inputs from it
            instead of generating them (read-only).  Takes precedence over
            *golden_fn* when both are provided.
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

    tensor_specs = [s for s in specs if isinstance(s, TensorSpec)]
    scalar_specs = [s for s in specs if isinstance(s, ScalarSpec)]

    start = time.time()
    _stage = _Stage  # local alias so the with-statement reads `_stage(...)`

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
    input_snapshot: dict[str, torch.Tensor] = {}
    scalar_specs_eff: dict[str, ScalarSpec] = {}
    with _stage("generate inputs"):
        if data_dir is not None:
            required: list[tuple[str, str]] = []
            for spec in (*tensor_specs, *scalar_specs):
                required.extend(_required_files(spec))
            missing = [
                str(data_dir / sub / name)
                for sub, name in required
                if not (data_dir / sub / name).is_file()
            ]
            if missing:
                return _fail(f"golden_data is missing files: {missing}")
            print(f"[RUN]   cache hit: {data_dir / 'in'}", flush=True)
            # Load inputs + inout initial values from {dir}/in/; pure outputs stay zero-init.
            input_names = [
                s.name for s in tensor_specs
                if not s.is_output or s.init_value is not None
            ]
            tensors = _load_tensors(data_dir, "in", input_names)
            for spec in tensor_specs:
                if spec.is_output and spec.init_value is None:
                    tensors[spec.name] = torch.zeros(spec.shape, dtype=spec.dtype)
            # Load each scalar from its own {name}.pt; verify dtype matches the
            # spec, then reconstruct a ScalarSpec (cached value overrides the
            # spec value, dtype must be identical).
            for s in scalar_specs:
                cached = torch.load(data_dir / "in" / f"{s.name}.pt", weights_only=True)
                if not isinstance(cached, torch.Tensor) or cached.ndim != 0:
                    shape = tuple(cached.shape) if isinstance(cached, torch.Tensor) else type(cached).__name__
                    return _fail(
                        f"{s.name}.pt must contain a 0-dim torch.Tensor, got {shape}"
                    )
                if cached.dtype != s.dtype:
                    return _fail(
                        f"{s.name}.pt dtype mismatch: spec={s.dtype} cache={cached.dtype}"
                    )
                scalar_specs_eff[s.name] = ScalarSpec(
                    name=s.name, dtype=s.dtype, value=cached
                )
        else:
            tensors = {spec.name: spec.create_tensor() for spec in tensor_specs}
            scalar_specs_eff = {s.name: s for s in scalar_specs}
            input_snapshot = {
                spec.name: tensors[spec.name].clone()
                for spec in tensor_specs
                if not spec.is_output or spec.init_value is not None
            }
            in_dir = work_dir / "data" / "in"
            _save_tensors(in_dir, input_snapshot)
            _save_tensors(in_dir, {s.name: s.value for s in scalar_specs})

    # Runtime
    with _stage("runtime"):
        # Detect L3 programs (HOST Orchestrator): ir.compile() returns a
        # DistributedCompiledProgram.  These cannot use execute_compiled()
        # (which expects kernel_config.py at the top level); instead call
        # compiled() directly with all scalars passed as 0-dim tensors and
        # args reordered to the function declaration order via _get_metadata().
        _is_l3 = False
        if runtime_dir is None:
            try:
                from pypto.ir.distributed_compiled_program import (  # noqa: PLC0415
                    DistributedCompiledProgram as _DC,
                )
                _is_l3 = isinstance(compiled, _DC)
            except ImportError:
                pass

        if _is_l3:
            from pypto.runtime import RunConfig as _PRC  # noqa: PLC0415

            # Build name->value dict: tensors unchanged, scalars as 0-dim tensors.
            _arg_map: dict[str, Any] = {s.name: tensors[s.name] for s in tensor_specs}
            _arg_map.update(
                {s.name: scalar_specs_eff[s.name].value for s in scalar_specs}
            )
            # Reorder to function declaration order from param_infos.
            # SSA names have the form ``orig_name__ssa_vN``; strip the suffix.
            _param_infos, _, _ = compiled._get_metadata()
            _ordered_l3: list[Any] = [
                _arg_map[p.name.split("__ssa_")[0]] for p in _param_infos
            ]
            _platform = config.runtime.get("platform", "a2a3")
            # Forward every RunConfig field the caller supplied so L3 behaves
            # consistently with the non-L3 path (which forwards via run_jit's
            # whitelist).  Unknown keys fail fast inside RunConfig(**...).
            import dataclasses as _dc  # noqa: PLC0415
            _allowed = {f.name for f in _dc.fields(_PRC)}
            _kwargs = {k: v for k, v in config.runtime.items() if k in _allowed}
            _kwargs.setdefault("platform", _platform)
            _kwargs.setdefault("device_id", 0)
            _kwargs["backend_type"] = _backend_for_platform(_platform)
            _run_cfg = _PRC(**_kwargs)
            compiled(*_ordered_l3, config=_run_cfg)
        else:
            ordered: list[Any] = [
                tensors[s.name] if isinstance(s, TensorSpec) else scalar_specs_eff[s.name].to_ctypes()
                for s in specs
            ]
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
            scratch: dict[str, Any] = {}
            for spec in specs:
                if isinstance(spec, ScalarSpec):
                    scratch[spec.name] = scalar_specs_eff[spec.name].to_python()
                elif spec.is_output and spec.init_value is None:
                    scratch[spec.name] = torch.zeros(spec.shape, dtype=spec.dtype)
                else:
                    scratch[spec.name] = input_snapshot[spec.name].clone()
            golden_fn(scratch)
            golden_outputs = {spec.name: scratch[spec.name] for spec in tensor_specs if spec.is_output}
            _save_tensors(work_dir / "data" / "out", golden_outputs)

    # Validate
    with _stage("validate"):
        try:
            input_tensors = {spec.name: tensors[spec.name] for spec in tensor_specs if not spec.is_output}
            validate_golden(
                device_outputs,
                golden_outputs,
                rtol=config.rtol,
                atol=config.atol,
                compare_fn=config.compare_fn,
                inputs=input_tensors,
            )
        except AssertionError as e:
            return _fail(str(e))

    total = time.time() - start
    print(f"[RUN] PASS ({total:.2f}s)", flush=True)
    return RunResult(passed=True, execution_time=total)


def _resolve_jit_work_dir(fn: Any) -> Path:
    """Return the most recently compiled :class:`CompiledProgram`'s output dir
    from a :class:`pypto.jit.JITFunction`'s L1 cache.

    Used after a JIT call to locate ``data/in`` and ``data/out`` for snapshot
    persistence.
    """
    cache = getattr(fn, "_cache", None)
    if not cache:
        raise RuntimeError(
            f"@pl.jit function {getattr(fn, '__name__', fn)!r} has no cached "
            "CompiledProgram; was it executed?"
        )
    compiled = next(reversed(cache.values()))
    return Path(compiled.output_dir)


def _jit_compile_only(fn: Any, jit_args: list[Any], platform: str | None) -> Path:
    """Drive the JIT decorator's compile step without executing on device.

    ``CompiledProgram.__call__`` always dispatches to ``execute_compiled`` —
    ``codegen_only`` is only honored by the higher-level
    :func:`pypto.runtime.run` entry point.  To get a true compile-only path
    on the JIT side we replicate :meth:`JITFunction.__call__`'s prelude
    (bind args → cache key → ``_compile``) and stop there, populating the
    JIT's L1 cache so a subsequent call hits and runs end-to-end.
    """
    import pypto.language as pl_mod  # noqa: PLC0415
    from pypto.jit.cache import make_cache_key  # noqa: PLC0415

    param_names, _arguments, tensor_meta, scalar_values, scalar_dtypes, dynamic_dims = (
        fn._bind_args(tuple(jit_args), {})
    )
    key = make_cache_key(
        source_hash=fn._get_source_hash(),
        param_names=param_names,
        tensor_shapes={n: m.shape for n, m in tensor_meta.items()},
        tensor_dtypes={n: m.dtype for n, m in tensor_meta.items()},
        dynamic_dims=dynamic_dims,
        scalar_values=scalar_values,
        platform=platform,
    )
    if key not in fn._cache:
        fn._cache[key] = fn._compile(
            tensor_meta, scalar_values, scalar_dtypes, dynamic_dims, pl_mod, platform=platform
        )
    return Path(fn._cache[key].output_dir)


def run_jit(
    fn: Any,
    specs: list[TensorSpec | ScalarSpec],
    config: RunConfig | None = None,
    golden_fn: Callable | None = None,
    golden_data: str | None = None,
    runtime_dir: str | None = None,
) -> RunResult:
    """Run a ``@pl.jit`` entry under the same harness as :func:`run`.

    The JIT decorator owns compilation + caching; this wrapper layers
    :class:`TensorSpec`-driven input generation, ``golden_fn`` validation,
    and persistence to ``data/in`` / ``data/out`` on top.

    Args:
        fn: A ``@pl.jit`` decorated callable (``JITFunction``).
        specs: Ordered list of :class:`TensorSpec` and :class:`ScalarSpec`,
            matching the JIT function's parameter order.
        config: Run configuration.  Both ``config.compile`` and
            ``config.runtime`` are merged and forwarded into a single
            :class:`pypto.runtime.RunConfig`, since the JIT decorator owns
            both compile and execute under one call.  Putting compile-side
            knobs (``dump_passes``, ``save_kernels``, ``codegen_only``,
            ``backend_type``, ``strategy``) in ``config.compile`` keeps the
            split symmetric with :func:`run`.  Conflicts between the two
            dicts raise an error.
        golden_fn: Optional ``golden_fn(values)`` for in-place reference
            computation (same contract as :func:`run`).
        golden_data: Optional cached-data directory (same contract as
            :func:`run`).
        runtime_dir: Optional pre-compiled ``build_output`` directory.  When
            set, bypasses the JIT and dispatches via
            :func:`pypto.runtime.execute_compiled`.

    Returns:
        :class:`RunResult` matching :func:`run`'s convention.
    """
    from pypto.runtime import RunConfig as RTRunConfig  # noqa: PLC0415
    from pypto.runtime import execute_compiled  # noqa: PLC0415

    if config is None:
        config = RunConfig()
    if config.compile_only and runtime_dir is not None:
        return RunResult(
            passed=False,
            error="runtime_dir is incompatible with config.compile_only",
        )

    data_dir = Path(golden_data) if golden_data is not None else None
    tensor_specs = [s for s in specs if isinstance(s, TensorSpec)]
    scalar_specs = [s for s in specs if isinstance(s, ScalarSpec)]

    start = time.time()
    _stage = _Stage

    def _fail(error: str) -> RunResult:
        return RunResult(passed=False, error=error, execution_time=time.time() - start)

    # Compile (or pick runtime_dir) — done first so stage order matches run().
    # We feed JIT a set of zero-cost dummy tensors derived from the specs, just
    # to satisfy _bind_args's tensor-meta extraction.  Real tensors with the
    # same shape/dtype hit the same cache key on the later execute call.
    if runtime_dir is not None:
        work_dir = Path(runtime_dir)
        if not work_dir.is_dir():
            return _fail(f"runtime_dir does not exist: {work_dir}")
        print(f"[RUN] runtime_only: skipping JIT compile, using {work_dir}", flush=True)
        rt_kwargs: dict[str, Any] = {}
    else:
        with _stage("compile"):
            rt_kwargs = {**config.compile, **config.runtime}
            overlap = config.compile.keys() & config.runtime.keys()
            if overlap:
                return _fail(
                    f"config.compile and config.runtime both set: {sorted(overlap)}"
                )
            rt_kwargs.setdefault("rtol", config.rtol)
            rt_kwargs.setdefault("atol", config.atol)
            dummy_args: list[Any] = []
            for spec in specs:
                if isinstance(spec, ScalarSpec):
                    dummy_args.append(spec.value.item())
                else:
                    dummy_args.append(torch.empty(spec.shape, dtype=spec.dtype))
            work_dir = _jit_compile_only(fn, dummy_args, platform=rt_kwargs.get("platform"))

    if config.compile_only:
        total = time.time() - start
        print(f"[RUN] PASS ({total:.2f}s)", flush=True)
        return RunResult(passed=True, execution_time=total)

    # Generate Inputs
    input_snapshot: dict[str, torch.Tensor] = {}
    scalar_specs_eff: dict[str, ScalarSpec] = {}
    with _stage("generate inputs"):
        if data_dir is not None:
            required: list[tuple[str, str]] = []
            for spec in (*tensor_specs, *scalar_specs):
                required.extend(_required_files(spec))
            missing = [
                str(data_dir / sub / name)
                for sub, name in required
                if not (data_dir / sub / name).is_file()
            ]
            if missing:
                return _fail(f"golden_data is missing files: {missing}")
            print(f"[RUN]   cache hit: {data_dir / 'in'}", flush=True)
            input_names = [
                s.name for s in tensor_specs
                if not s.is_output or s.init_value is not None
            ]
            tensors = _load_tensors(data_dir, "in", input_names)
            for spec in tensor_specs:
                if spec.is_output and spec.init_value is None:
                    tensors[spec.name] = torch.zeros(spec.shape, dtype=spec.dtype)
            for s in scalar_specs:
                cached = torch.load(data_dir / "in" / f"{s.name}.pt", weights_only=True)
                if not isinstance(cached, torch.Tensor) or cached.ndim != 0:
                    shape = (
                        tuple(cached.shape) if isinstance(cached, torch.Tensor)
                        else type(cached).__name__
                    )
                    return _fail(
                        f"{s.name}.pt must contain a 0-dim torch.Tensor, got {shape}"
                    )
                if cached.dtype != s.dtype:
                    return _fail(
                        f"{s.name}.pt dtype mismatch: spec={s.dtype} cache={cached.dtype}"
                    )
                scalar_specs_eff[s.name] = ScalarSpec(name=s.name, dtype=s.dtype, value=cached)
            input_snapshot = {
                spec.name: tensors[spec.name].clone()
                for spec in tensor_specs
                if not spec.is_output or spec.init_value is not None
            }
        else:
            tensors = {spec.name: spec.create_tensor() for spec in tensor_specs}
            scalar_specs_eff = {s.name: s for s in scalar_specs}
            input_snapshot = {
                spec.name: tensors[spec.name].clone()
                for spec in tensor_specs
                if not spec.is_output or spec.init_value is not None
            }
            in_dir = work_dir / "data" / "in"
            _save_tensors(in_dir, input_snapshot)
            _save_tensors(in_dir, {s.name: s.value for s in scalar_specs})

    # Runtime
    with _stage("runtime"):
        if runtime_dir is not None:
            ordered: list[Any] = [
                tensors[s.name] if isinstance(s, TensorSpec) else scalar_specs_eff[s.name].to_ctypes()
                for s in specs
            ]
            # execute_compiled only accepts a subset of pypto.runtime.RunConfig
            # fields — strip the compile-only ones (dump_passes, codegen_only,
            # rtol, atol, etc.) before forwarding.
            exec_kwargs = {
                k: v for k, v in config.runtime.items()
                if k in {"platform", "device_id", "pto_isa_commit", "runtime_profiling", "level"}
            }
            execute_compiled(work_dir, ordered, **exec_kwargs)
        else:
            jit_args: list[Any] = []
            for spec in specs:
                if isinstance(spec, ScalarSpec):
                    jit_args.append(scalar_specs_eff[spec.name].to_python())
                else:
                    jit_args.append(tensors[spec.name])
            fn(*jit_args, config=RTRunConfig(**rt_kwargs))

    if golden_fn is None and golden_data is None:
        total = time.time() - start
        print(
            f"[RUN] PASS ({total:.2f}s, validation skipped: no golden_fn or golden_data)",
            flush=True,
        )
        return RunResult(passed=True, execution_time=total)

    device_outputs = {spec.name: tensors[spec.name] for spec in tensor_specs if spec.is_output}

    with _stage("compute golden"):
        if data_dir is not None:
            print(f"[RUN]   cache hit: {data_dir / 'out'}", flush=True)
            output_names = [s.name for s in tensor_specs if s.is_output]
            golden_outputs = _load_tensors(data_dir, "out", output_names)
        else:
            scratch: dict[str, Any] = {}
            for spec in specs:
                if isinstance(spec, ScalarSpec):
                    scratch[spec.name] = scalar_specs_eff[spec.name].to_python()
                elif spec.is_output and spec.init_value is None:
                    scratch[spec.name] = torch.zeros(spec.shape, dtype=spec.dtype)
                else:
                    scratch[spec.name] = input_snapshot[spec.name].clone()
            golden_fn(scratch)
            golden_outputs = {
                spec.name: scratch[spec.name] for spec in tensor_specs if spec.is_output
            }
            _save_tensors(work_dir / "data" / "out", golden_outputs)

    with _stage("validate"):
        try:
            input_tensors = {
                spec.name: tensors[spec.name] for spec in tensor_specs if not spec.is_output
            }
            validate_golden(
                device_outputs,
                golden_outputs,
                rtol=config.rtol,
                atol=config.atol,
                compare_fn=config.compare_fn,
                inputs=input_tensors,
            )
        except AssertionError as e:
            return _fail(str(e))

    total = time.time() - start
    print(f"[RUN] PASS ({total:.2f}s)", flush=True)
    return RunResult(passed=True, execution_time=total)
