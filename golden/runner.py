# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile PyPTO programs, run them on device, and validate against goldens.

Public entry points: :func:`run` and :func:`run_jit`.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .spec import ScalarSpec, TensorSpec
from .validation import validate_golden


@dataclass
class RunResult:
    """Result of a :func:`run` invocation."""

    passed: bool
    error: str | None = None
    execution_time: float | None = None
    work_dir: Path | None = None

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


_DFX_FLAG_KEYS = ("enable_l2_swimlane", "enable_dump_tensor", "enable_pmu", "enable_dep_gen")


def _execute_compiled_kwargs(runtime: dict[str, Any]) -> dict[str, Any]:
    """Translate user-facing ``runtime_cfg`` into ``execute_compiled`` kwargs.

    The four DFX flags get bundled into a single ``dfx: _DfxOpts``; all other
    keys pass through unfiltered, so ``execute_compiled`` raises ``TypeError``
    on unknown keys rather than us silently dropping them.
    """
    out: dict[str, Any] = {k: v for k, v in runtime.items() if k not in _DFX_FLAG_KEYS}
    dfx_flags = {k: runtime[k] for k in _DFX_FLAG_KEYS if runtime.get(k)}
    if dfx_flags:
        try:
            from pypto.runtime.runner import _DfxOpts
        except ImportError as exc:
            raise ValueError(
                "This pypto runtime does not support execute_compiled DFX flags: "
                f"{sorted(dfx_flags)}"
            ) from exc

        out["dfx"] = _DfxOpts(**dfx_flags)
    return out


def _consume_runtime_harness_keys(runtime_cfg: dict[str, Any]) -> None:
    """Pop harness-only keys from *runtime_cfg* and apply their side effects.

    Recognised key (not forwarded to ``execute_compiled``):
      - ``log_level``: PyPTO runtime log threshold, see
        :func:`pypto.runtime.log_config.configure_log`. One of ``debug``,
        ``v0..v9``, ``info``, ``warn``, ``error``, ``null``.

    Mutates *runtime_cfg* in place by popping the recognised key.
    """
    level = runtime_cfg.pop("log_level", None)
    if level is None:
        return
    from pypto.runtime.log_config import configure_log
    configure_log(level)


def _stale_cpps(work_dir: Path) -> list[Path]:
    """Return cpps under ``kernels/`` / ``orchestration/`` that need rebuilding.

    A cpp is considered stale if **either**:

    - its sibling ``.so``/``.o`` is missing entirely (binary never built or
      removed by hand), **or**
    - any existing sibling ``.so``/``.o`` is older than the cpp itself
      (cpp was edited after its last build).

    Both cases require a rebuild; reporting them uniformly through this
    helper keeps the runner's log message honest (previously a missing
    binary would log ``no cpp edits ... reusing cached binaries`` even
    though ``compile_and_assemble`` would silently rebuild it).
    """
    stale: list[Path] = []
    for sub in ("kernels", "orchestration"):
        root = work_dir / sub
        if not root.is_dir():
            continue
        for cpp in root.rglob("*.cpp"):
            siblings = [cpp.with_suffix(ext) for ext in (".so", ".o")]
            existing = [p for p in siblings if p.exists()]
            if not existing:
                stale.append(cpp)
                continue
            cpp_mtime = cpp.stat().st_mtime
            if any(p.stat().st_mtime < cpp_mtime for p in existing):
                stale.append(cpp)
    return stale


def _format_stale_paths(stale: list[Path], work_dir: Path, max_show: int = 5) -> str:
    """Render a comma-separated list of stale cpp paths relative to
    *work_dir*, truncated to *max_show* entries with a ``(+N more)`` tail
    when the list is longer."""
    rels = [str(p.relative_to(work_dir)) for p in stale]
    if len(rels) <= max_show:
        return ", ".join(rels)
    head = ", ".join(rels[:max_show])
    return f"{head} (+{len(rels) - max_show} more)"


def _setup_runtime_dir(runtime_dir: str, *, compile_label: str) -> Path:
    """Validate *runtime_dir*; rebuild kernel cpps from edited ``.pto`` files
    and drop cached binaries for any cpp newer than its ``.so``/``.o``.

    Raises ``ValueError`` if the directory does not exist.
    """
    work_dir = Path(runtime_dir)
    if not work_dir.is_dir():
        raise ValueError(f"runtime_dir does not exist: {work_dir}")
    print(f"[RUN] runtime_only: skipping {compile_label}, using {work_dir}", flush=True)
    # pto -> cpp: splices updated ptoas body into kernel cpps, bumping their
    # mtime so the cpp -> .so check below picks them up.
    from pypto.runtime.debug.pto_rebuild import rebuild_kernel_cpp_from_pto
    rebuild_kernel_cpp_from_pto(work_dir)
    stale = _stale_cpps(work_dir)
    if stale:
        from pypto.runtime.debug.replay import invalidate_binary_cache
        invalidate_binary_cache(work_dir)
        print(
            f"[cpp->.so] cpp edits or missing binaries detected "
            f"({len(stale)} file(s)): {_format_stale_paths(stale, work_dir)}; rebuilding",
            flush=True,
        )
    else:
        print("[cpp->.so] no cpp edits since last build; reusing cached binaries", flush=True)
    return work_dir


def _prepare_inputs(
    specs: list[TensorSpec | ScalarSpec],
    tensor_specs: list[TensorSpec],
    scalar_specs: list[ScalarSpec],
    data_dir: Path | None,
    work_dir: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, ScalarSpec], dict[str, torch.Tensor]]:
    """Build inputs for the runtime stage.

    With *data_dir* set, load tensors and scalars from ``{data_dir}/in/`` and
    leave ``input_snapshot`` empty (golden will be loaded from cache, no need
    to clone inputs for ``golden_fn``). Otherwise generate from *specs* and
    persist into ``{work_dir}/data/in/``.

    Raises ``ValueError`` on missing files or scalar dtype mismatch.
    """
    if data_dir is None:
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
        return tensors, scalar_specs_eff, input_snapshot

    required: list[tuple[str, str]] = []
    for spec in (*tensor_specs, *scalar_specs):
        required.extend(_required_files(spec))
    missing = [
        str(data_dir / sub / name)
        for sub, name in required
        if not (data_dir / sub / name).is_file()
    ]
    if missing:
        raise ValueError(f"golden_data is missing files: {missing}")
    print(f"[RUN]   cache hit: {data_dir / 'in'}", flush=True)

    # Load inputs + inout initial values from {dir}/in/; pure outputs stay zero-init.
    input_names = [s.name for s in tensor_specs if not s.is_output or s.init_value is not None]
    tensors = _load_tensors(data_dir, "in", input_names)
    for spec in tensor_specs:
        if spec.is_output and spec.init_value is None:
            tensors[spec.name] = torch.zeros(spec.shape, dtype=spec.dtype)

    scalar_specs_eff = {}
    for s in scalar_specs:
        cached = torch.load(data_dir / "in" / f"{s.name}.pt", weights_only=True)
        if not isinstance(cached, torch.Tensor) or cached.ndim != 0:
            shape = tuple(cached.shape) if isinstance(cached, torch.Tensor) else type(cached).__name__
            raise ValueError(f"{s.name}.pt must contain a 0-dim torch.Tensor, got {shape}")
        if cached.dtype != s.dtype:
            raise ValueError(f"{s.name}.pt dtype mismatch: spec={s.dtype} cache={cached.dtype}")
        scalar_specs_eff[s.name] = ScalarSpec(name=s.name, dtype=s.dtype, value=cached)

    return tensors, scalar_specs_eff, {}


def _execute_via_runner(
    work_dir: Path,
    specs: list[TensorSpec | ScalarSpec],
    tensors: dict[str, torch.Tensor],
    scalar_specs_eff: dict[str, ScalarSpec],
    runtime_cfg: dict[str, Any],
) -> None:
    """Reorder args to orchestration param order and dispatch via ``execute_compiled``."""
    from pypto.runtime import execute_compiled

    ordered: list[Any] = [
        tensors[s.name] if isinstance(s, TensorSpec) else scalar_specs_eff[s.name].to_ctypes()
        for s in specs
    ]
    execute_compiled(work_dir, ordered, **_execute_compiled_kwargs(runtime_cfg))


def _try_l3_dispatch(
    compiled: Any,
    specs: list[TensorSpec | ScalarSpec],
    tensors: dict[str, torch.Tensor],
    scalar_specs_eff: dict[str, ScalarSpec],
    runtime_cfg: dict[str, Any],
) -> bool:
    """If *compiled* is an L3 ``DistributedCompiledProgram``, dispatch it and return True.

    L3 (HOST Orchestrator) programs cannot use ``execute_compiled`` (no
    top-level ``kernel_config.py``); the compiled object is callable directly
    with ``pypto.runtime.RunConfig``.
    """
    try:
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram
    except ImportError:
        return False
    if not isinstance(compiled, DistributedCompiledProgram):
        return False

    import dataclasses

    from pypto.runtime import RunConfig as PyptoRunConfig

    # Build name->value map; SSA names ``orig__ssa_vN`` get stripped to ``orig``.
    arg_map: dict[str, Any] = {}
    for s in specs:
        if isinstance(s, TensorSpec):
            arg_map[s.name] = tensors[s.name]
        else:
            arg_map[s.name] = scalar_specs_eff[s.name].value
    param_infos, _, _ = compiled._get_metadata()
    ordered = [arg_map[p.name.split("__ssa_")[0]] for p in param_infos]

    platform = runtime_cfg.get("platform", "a2a3")
    allowed = {f.name for f in dataclasses.fields(PyptoRunConfig)}
    kwargs = {k: v for k, v in runtime_cfg.items() if k in allowed}
    kwargs.setdefault("platform", platform)
    kwargs.setdefault("device_id", 0)
    kwargs["backend_type"] = _backend_for_platform(platform)
    compiled(*ordered, config=PyptoRunConfig(**kwargs))
    return True


def _compute_golden(
    specs: list[TensorSpec | ScalarSpec],
    tensor_specs: list[TensorSpec],
    scalar_specs_eff: dict[str, ScalarSpec],
    input_snapshot: dict[str, torch.Tensor],
    work_dir: Path,
    data_dir: Path | None,
    golden_fn: Callable | None,
) -> dict[str, torch.Tensor]:
    """Produce golden output tensors for validation.

    With *data_dir* set, load from ``{data_dir}/out/``. Otherwise call
    *golden_fn* on a scratch dict (inputs cloned from *input_snapshot*,
    outputs zero-init) and persist results into ``{work_dir}/data/out/``.
    """
    with _Stage("compute golden"):
        if data_dir is not None:
            print(f"[RUN]   cache hit: {data_dir / 'out'}", flush=True)
            output_names = [s.name for s in tensor_specs if s.is_output]
            return _load_tensors(data_dir, "out", output_names)

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
        return golden_outputs


def _validate(
    tensor_specs: list[TensorSpec],
    tensors: dict[str, torch.Tensor],
    golden_outputs: dict[str, torch.Tensor],
    rtol: float,
    atol: float,
    compare_fn: dict[str, Callable],
) -> None:
    """Compare device outputs against *golden_outputs*. Raises ``AssertionError``."""
    with _Stage("validate"):
        device_outputs = {spec.name: tensors[spec.name] for spec in tensor_specs if spec.is_output}
        input_tensors = {spec.name: tensors[spec.name] for spec in tensor_specs if not spec.is_output}
        validate_golden(
            device_outputs, golden_outputs,
            rtol=rtol, atol=atol, compare_fn=compare_fn, inputs=input_tensors,
        )


def run(
    program: Any,
    specs: list[TensorSpec | ScalarSpec],
    golden_fn: Callable | None = None,
    golden_data: str | None = None,
    compile_cfg: dict[str, Any] | None = None,
    runtime_cfg: dict[str, Any] | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    compare_fn: dict[str, Callable] | None = None,
    compile_only: bool = False,
    runtime_dir: str | None = None,
) -> RunResult:
    """Compile *program*, run on device, and validate against golden.

    Args:
        program: ``@pl.program`` class or ``ir.Program``.
        specs: :class:`TensorSpec` / :class:`ScalarSpec` list in orchestration
            parameter order.
        golden_fn: ``golden_fn(values)`` that fills outputs in-place; *values*
            maps spec name to tensor clone or Python scalar. Ignored when
            *golden_data* is set; if neither is given, validation is skipped.
        golden_data: Directory with ``in/{name}.pt`` and ``out/{name}.pt``;
            loads inputs and expected outputs (read-only). Takes precedence
            over *golden_fn*.
        compile_cfg: Kwargs forwarded to :func:`pypto.ir.compile`. Unknown
            keys raise there.
        runtime_cfg: Kwargs forwarded to
            :func:`pypto.runtime.execute_compiled` (``platform``, ``device_id``,
            ``enable_l2_swimlane``, ...). Unknown keys raise there, except
            the harness-only key ``log_level``, which is consumed up-front
            to configure the PyPTO runtime logger via
            :func:`pypto.runtime.log_config.configure_log`.
        rtol, atol: Golden comparison tolerances.
        compare_fn: Per-output-name overrides for ``torch.allclose``; see
            :func:`golden.validation.validate_golden`.
        compile_only: Stop after code generation; skip execute and validate.
        runtime_dir: Pre-compiled ``build_output/`` directory to reuse. Skips
            compile and invalidates cached ``.so``/``.bin`` so cpp edits
            rebuild; *compile_cfg* is ignored and *compile_only* is rejected.

    Returns:
        :class:`RunResult`.
    """
    from pypto import ir

    compile_cfg = compile_cfg or {}
    runtime_cfg = dict(runtime_cfg or {})  # copy: we pop harness-only keys
    compare_fn = compare_fn or {}

    _consume_runtime_harness_keys(runtime_cfg)

    if compile_only and runtime_dir is not None:
        return RunResult(passed=False, error="runtime_dir is incompatible with compile_only")

    data_dir = Path(golden_data) if golden_data is not None else None
    tensor_specs = [s for s in specs if isinstance(s, TensorSpec)]
    scalar_specs = [s for s in specs if isinstance(s, ScalarSpec)]

    start = time.time()
    work_dir: Path | None = None

    def _fail(error: str) -> RunResult:
        return RunResult(
            passed=False, error=error,
            execution_time=time.time() - start, work_dir=work_dir,
        )

    # Compile (or pick runtime_dir)
    compiled: Any = None
    if runtime_dir is not None:
        try:
            work_dir = _setup_runtime_dir(runtime_dir, compile_label="compile")
        except ValueError as e:
            return _fail(str(e))
    else:
        with _Stage("compile"):
            compile_kwargs = dict(compile_cfg)
            platform = runtime_cfg.get("platform")
            if platform is not None:
                compile_kwargs.setdefault("backend_type", _backend_for_platform(platform))
                # L3 distributed programs bake the platform into compiled.platform
                # at compile time (the runtime config's platform is ignored when
                # assembling chip callables). Without this, compiled.platform falls
                # back to the backend's default sim platform, so a `-p a2a3` run
                # silently compiles incore kernels for a2a3sim (g++-15) instead of
                # the real device (ccec).
                compile_kwargs.setdefault("platform", platform)
            compiled = ir.compile(program, **compile_kwargs)
            work_dir = Path(compiled.output_dir)
        if compile_only:
            total = time.time() - start
            print(f"[RUN] PASS ({total:.2f}s)", flush=True)
            return RunResult(passed=True, execution_time=total, work_dir=work_dir)

    # Generate Inputs
    try:
        with _Stage("generate inputs"):
            tensors, scalar_specs_eff, input_snapshot = _prepare_inputs(
                specs, tensor_specs, scalar_specs, data_dir, work_dir,
            )
    except ValueError as e:
        return _fail(str(e))

    # Compute Golden
    golden_outputs: dict[str, torch.Tensor] | None = None
    if golden_fn is not None or golden_data is not None:
        golden_outputs = _compute_golden(
            specs, tensor_specs, scalar_specs_eff, input_snapshot,
            work_dir, data_dir, golden_fn,
        )

    # Runtime
    with _Stage("runtime"):
        if compiled is None or not _try_l3_dispatch(
            compiled, specs, tensors, scalar_specs_eff, runtime_cfg,
        ):
            _execute_via_runner(work_dir, specs, tensors, scalar_specs_eff, runtime_cfg)

    # Validate
    if golden_outputs is None:
        total = time.time() - start
        print(f"[RUN] PASS ({total:.2f}s, validation skipped: no golden_fn or golden_data)", flush=True)
        return RunResult(passed=True, execution_time=total, work_dir=work_dir)
    try:
        _validate(tensor_specs, tensors, golden_outputs, rtol, atol, compare_fn)
    except AssertionError as e:
        return _fail(str(e))

    total = time.time() - start
    print(f"[RUN] PASS ({total:.2f}s)", flush=True)
    return RunResult(passed=True, execution_time=total, work_dir=work_dir)


def _jit_compile_only(
    fn: Any, jit_args: list[Any], platform: str | None, **compile_kwargs: Any,
) -> Path:
    """Compile a ``@pl.jit`` function without executing on device.

    Replays :meth:`JITFunction.__call__`'s prelude (bind args → cache key →
    ``_compile``) and stops there, populating the L1 cache. Private-API
    surgery — pypto exposes no platform-aware compile-only entry. Extra
    *compile_kwargs* forward to ``fn._compile``, so unsupported keys raise
    ``TypeError`` there rather than being silently dropped.
    """
    import pypto.language as pl_mod
    from pypto.jit.cache import make_cache_key

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
            tensor_meta, scalar_values, scalar_dtypes, dynamic_dims, pl_mod,
            platform=platform, **compile_kwargs,
        )
    return Path(fn._cache[key].output_dir)


def run_jit(
    fn: Any,
    specs: list[TensorSpec | ScalarSpec],
    golden_fn: Callable | None = None,
    golden_data: str | None = None,
    compile_cfg: dict[str, Any] | None = None,
    runtime_cfg: dict[str, Any] | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    compare_fn: dict[str, Callable] | None = None,
    compile_only: bool = False,
    runtime_dir: str | None = None,
) -> RunResult:
    """JIT-flavoured :func:`run`: compile via ``@pl.jit``, then same harness.

    Args:
        fn: ``@pl.jit`` decorated callable.
        specs: :class:`TensorSpec` / :class:`ScalarSpec` list in the JIT
            function's parameter order.
        golden_fn: ``golden_fn(values)`` that fills outputs in-place; *values*
            maps spec name to tensor clone or Python scalar. Ignored when
            *golden_data* is set; if neither is given, validation is skipped.
        golden_data: Directory with ``in/{name}.pt`` and ``out/{name}.pt``;
            loads inputs and expected outputs (read-only). Takes precedence
            over *golden_fn*.
        compile_cfg: Kwargs forwarded to ``fn._compile``. The JIT path only
            honors ``platform`` (typically supplied via *runtime_cfg*), so
            other keys raise there.
        runtime_cfg: Kwargs forwarded to
            :func:`pypto.runtime.execute_compiled` (``platform``, ``device_id``,
            ``enable_l2_swimlane``, ...). Unknown keys raise there, except
            the harness-only key ``log_level``, which is consumed up-front
            to configure the PyPTO runtime logger via
            :func:`pypto.runtime.log_config.configure_log`.
        rtol, atol: Golden comparison tolerances.
        compare_fn: Per-output-name overrides for ``torch.allclose``; see
            :func:`golden.validation.validate_golden`.
        compile_only: Stop after code generation; skip execute and validate.
        runtime_dir: Pre-compiled ``build_output/`` directory to reuse. Skips
            compile and invalidates cached ``.so``/``.bin`` so cpp edits
            rebuild; *compile_cfg* is ignored and *compile_only* is rejected.

    Returns:
        :class:`RunResult`.
    """
    compile_cfg = compile_cfg or {}
    runtime_cfg = dict(runtime_cfg or {})  # copy: we pop harness-only keys
    compare_fn = compare_fn or {}

    _consume_runtime_harness_keys(runtime_cfg)

    if compile_only and runtime_dir is not None:
        return RunResult(passed=False, error="runtime_dir is incompatible with compile_only")

    data_dir = Path(golden_data) if golden_data is not None else None
    tensor_specs = [s for s in specs if isinstance(s, TensorSpec)]
    scalar_specs = [s for s in specs if isinstance(s, ScalarSpec)]

    start = time.time()
    work_dir: Path | None = None

    def _fail(error: str) -> RunResult:
        return RunResult(
            passed=False, error=error,
            execution_time=time.time() - start, work_dir=work_dir,
        )

    # Compile
    if runtime_dir is not None:
        try:
            work_dir = _setup_runtime_dir(runtime_dir, compile_label="JIT compile")
        except ValueError as e:
            return _fail(str(e))
    else:
        with _Stage("compile"):
            # Dummy args just satisfy _bind_args's tensor-meta extraction; real
            # tensors with the same shape/dtype hit the same cache key later.
            dummy_args = [
                spec.value.item() if isinstance(spec, ScalarSpec)
                else torch.empty(spec.shape, dtype=spec.dtype)
                for spec in specs
            ]
            work_dir = _jit_compile_only(
                fn, dummy_args,
                platform=runtime_cfg.get("platform"),
                **compile_cfg,
            )
        if compile_only:
            total = time.time() - start
            print(f"[RUN] PASS ({total:.2f}s)", flush=True)
            return RunResult(passed=True, execution_time=total, work_dir=work_dir)

    # Generate Inputs
    try:
        with _Stage("generate inputs"):
            tensors, scalar_specs_eff, input_snapshot = _prepare_inputs(
                specs, tensor_specs, scalar_specs, data_dir, work_dir,
            )
    except ValueError as e:
        return _fail(str(e))

    # Compute Golden
    golden_outputs: dict[str, torch.Tensor] | None = None
    if golden_fn is not None or golden_data is not None:
        golden_outputs = _compute_golden(
            specs, tensor_specs, scalar_specs_eff, input_snapshot,
            work_dir, data_dir, golden_fn,
        )

    # Runtime
    with _Stage("runtime"):
        _execute_via_runner(work_dir, specs, tensors, scalar_specs_eff, runtime_cfg)

    # Validate
    if golden_outputs is None:
        total = time.time() - start
        print(f"[RUN] PASS ({total:.2f}s, validation skipped: no golden_fn or golden_data)", flush=True)
        return RunResult(passed=True, execution_time=total, work_dir=work_dir)
    try:
        _validate(tensor_specs, tensors, golden_outputs, rtol, atol, compare_fn)
    except AssertionError as e:
        return _fail(str(e))

    total = time.time() - start
    print(f"[RUN] PASS ({total:.2f}s)", flush=True)
    return RunResult(passed=True, execution_time=total, work_dir=work_dir)
