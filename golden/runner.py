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
import re
import importlib.util
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
    bench_stats: Any = None  # pypto BenchmarkStats when benchmark=... was requested

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


_DFX_FLAG_KEYS = (
    "enable_l2_swimlane",
    "enable_dump_tensor",
    "enable_pmu",
    "enable_dep_gen",
    "enable_scope_stats",
)


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
    # Single-chip / L2 builds keep kernels/ + orchestration/ at the root; an L3
    # distributed build puts one complete sub-build per rank under
    # next_levels/{rank}/. Scan both so hand-edited L3 cpps are detected.
    bases = [work_dir]
    next_levels = work_dir / "next_levels"
    if next_levels.is_dir():
        bases += [d for d in sorted(next_levels.iterdir()) if d.is_dir()]
    for base in bases:
        for sub in ("kernels", "orchestration"):
            root = base / sub
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


def _patch_aicore_bitcast_helpers(work_dir: Path) -> None:
    """Mark ptoas-generated bitcast helpers as aicore-callable.

    Some ptoas builds emit a scalar helper as ``static inline`` inside a kernel
    translation unit. ccec then treats it as a host function and rejects calls
    from ``__aicore__`` kernels. The generated helper is pure local type-punning,
    so adding ``__aicore__`` preserves semantics and lets runtime compilation
    proceed.
    """
    needle = "static inline To ptoas_bitcast(From from) {"
    replacement = "static __aicore__ inline To ptoas_bitcast(From from) {"
    patched: list[Path] = []
    for cpp in work_dir.rglob("*.cpp"):
        try:
            text = cpp.read_text()
        except UnicodeDecodeError:
            continue
        if needle not in text:
            continue
        cpp.write_text(text.replace(needle, replacement))
        patched.append(cpp)
    if patched:
        print(f"[RUN] patched {len(patched)} ptoas_bitcast helper(s) for aicore compilation", flush=True)


def _patch_l3_single_submit_host_orch(work_dir: Path) -> None:
    """Skip stale multi-submit code left after a compile-time single-submit branch.

    The A8W8 L3 prefill smoke currently builds a single-submit host_orch. The
    frontend does not fold the Python closure constant early enough, so the
    generated ``host_orch.py`` may contain the intended first submit followed by
    dead multi-submit alias code that references non-existent Python locals.
    """
    path = work_dir / "orchestration" / "host_orch.py"
    if not path.is_file():
        return
    text = path.read_text()
    marker = (
        '    tensors["hidden_out__ssa_v1"] = tensors["hidden_out__ssa_v0"]\n'
        "    cur__ssa_v0 = hidden_states__ssa_v0\n"
    )
    replacement = (
        '    tensors["hidden_out__ssa_v1"] = tensors["hidden_out__ssa_v0"]\n'
        "    return\n"
        "    cur__ssa_v0 = hidden_states__ssa_v0\n"
    )
    if marker not in text or replacement in text:
        return
    path.write_text(text.replace(marker, replacement, 1))
    print("[RUN] patched L3 host_orch single-submit dead branch", flush=True)


def _patch_l3_host_orch_ssa_aliases(work_dir: Path) -> None:
    """Rewrite generated Python SSA aliases to tensor-map aliases.

    Some L3 host_orch codegen emits Python locals such as
    ``cur__ssa_v0 = hidden_in__ssa_v0`` even though parameters live in the
    ``tensors`` dict. Runtime then raises NameError before submitting work.
    """
    path = work_dir / "orchestration" / "host_orch.py"
    if not path.is_file():
        return
    text = path.read_text()
    alias_re = re.compile(r'^    ([A-Za-z_]\w*__ssa_v\d+) = ([A-Za-z_]\w*__ssa_v\d+)\n', re.MULTILINE)

    def repl(match: re.Match[str]) -> str:
        lhs, rhs = match.groups()
        return f'    tensors["{lhs}"] = tensors["{rhs}"]\n'

    patched, count = alias_re.subn(repl, text)
    if count == 0:
        return
    path.write_text(patched)
    print(f"[RUN] patched {count} L3 host_orch SSA alias(es)", flush=True)


def _install_simpler_chip_contexts_compat() -> None:
    """Bridge PyPTO's legacy L3 runner expectation to newer simpler Worker.

    Older PyPTO L3 code passes ``w.chip_contexts`` into generated host_orch
    functions. Newer simpler versions allocate communication domains
    dynamically and no longer expose that attribute. Comm-less host_orch code
    ignores the argument, so an empty list preserves the old call shape.
    """
    try:
        from simpler.orchestrator import Orchestrator
        from simpler.task_interface import ChipCallable
        from simpler.worker import Worker
    except ImportError:
        return
    if hasattr(Worker, "chip_contexts"):
        chip_contexts_installed = True
    else:
        chip_contexts_installed = False

    if not chip_contexts_installed:
        def _chip_contexts(self: Any) -> list[Any]:  # noqa: ANN001 - runtime compatibility shim
            return []

        setattr(Worker, "chip_contexts", property(_chip_contexts))

    if getattr(Orchestrator.submit_next_level, "_pypto_legacy_chip_callable_compat", False):
        return

    submit_next_level_orig = Orchestrator.submit_next_level

    def _submit_next_level_compat(
        self: Any,
        callable_handle: Any,
        args: Any,
        config: Any = None,
        *,
        worker: int = -1,
    ) -> Any:
        if isinstance(callable_handle, ChipCallable) or hasattr(callable_handle, "buffer_ptr"):
            parent_worker = getattr(self, "_worker", None)
            if parent_worker is None:
                raise TypeError("orch.submit_next_level needs Worker-backed Orchestrator for ChipCallable compat")
            callable_handle = parent_worker.register(callable_handle)
        return submit_next_level_orig(self, callable_handle, args, config, worker=worker)

    setattr(_submit_next_level_compat, "_pypto_legacy_chip_callable_compat", True)
    Orchestrator.submit_next_level = _submit_next_level_compat


def _inherit_l3_next_level_runtime_config(compiled: Any) -> None:
    """Populate L3 CallConfig defaults from generated next-level kernel_config.

    ``DistributedConfig`` defaults to 1/1, but tensormap_and_ringbuffer
    generated kernels normally require block_dim=24 and aicpu_thread_num=4.
    Single-chip ``execute_compiled`` reads those values from kernel_config.py;
    the L3 runner needs the same values before it submits next-level tasks.
    """
    output_dir = getattr(compiled, "output_dir", None)
    dc = getattr(compiled, "_distributed_config", None)
    if output_dir is None or dc is None:
        return
    next_levels = Path(output_dir) / "next_levels"
    if not next_levels.is_dir():
        return
    for cfg_path in sorted(next_levels.glob("*/kernel_config.py")):
        spec = importlib.util.spec_from_file_location(f"_pypto_l3_kernel_config_{cfg_path.parent.name}", cfg_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        runtime_cfg = getattr(module, "RUNTIME_CONFIG", {})
        block_dim = runtime_cfg.get("block_dim")
        aicpu_thread_num = runtime_cfg.get("aicpu_thread_num")
        if block_dim is not None and getattr(dc, "block_dim", 1) == 1:
            dc.block_dim = int(block_dim)
        if aicpu_thread_num is not None and getattr(dc, "aicpu_thread_num", 1) == 1:
            dc.aicpu_thread_num = int(aicpu_thread_num)
        print(
            f"[RUN] L3 runtime config: block_dim={dc.block_dim}, "
            f"aicpu_thread_num={dc.aicpu_thread_num}",
            flush=True,
        )
        return


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
    try:
        from pypto.runtime.debug.pto_rebuild import rebuild_kernel_cpp_from_pto
    except ModuleNotFoundError:
        print("[runtime_only] pypto.runtime.debug unavailable; using existing runtime artifacts", flush=True)
        return work_dir
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
    save_data: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, ScalarSpec], dict[str, torch.Tensor]]:
    """Build inputs for the runtime stage.

    With *data_dir* set, load tensors and scalars from ``{data_dir}/in/`` and
    leave ``input_snapshot`` empty (golden will be loaded from cache, no need
    to clone inputs for ``golden_fn``). Otherwise generate from *specs* and,
    when *save_data* is True, persist into ``{work_dir}/data/in/``. Set
    *save_data* False to skip the on-disk ``.pt`` snapshot (validation still
    works via the in-memory ``input_snapshot``); useful when inputs are large
    (e.g. full-model weights) and golden replay is not needed.

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
        if save_data:
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


def _normalize_bench_cfg(benchmark: "bool | dict[str, Any] | None") -> dict[str, Any] | None:
    """Coerce the user-facing ``benchmark`` arg into a kwargs dict (or None).

    Accepts ``True`` (defaults), a kwargs dict (``rounds`` / ``warmup``), or a
    falsy value (benchmark disabled). Rejects unknown keys up-front so a typo
    surfaces here instead of as a confusing ``benchmark()`` TypeError.
    """
    if not benchmark:
        return None
    cfg = {} if benchmark is True else dict(benchmark)
    allowed = {"rounds", "warmup"}
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(f"benchmark config has unknown keys {sorted(unknown)}; allowed: {sorted(allowed)}")
    return cfg


def _resolve_bench_cfg(benchmark: "bool | dict[str, Any] | None") -> dict[str, Any] | None:
    """Resolve the effective benchmark config from the arg, then the environment.

    An explicit *benchmark* arg always wins, including an explicit ``False``
    (which disables benchmarking even when the env var is set). Only a ``None``
    arg falls back to the env var ``PYPTO_LIB_BENCHMARK`` (the daily-CI a2a3
    sweep enables it once for the whole job), which turns benchmark on for
    *every* harness run with no per-file flag, sized by
    ``PYPTO_LIB_BENCHMARK_ROUNDS`` / ``PYPTO_LIB_BENCHMARK_WARMUP``. Returns the
    kwargs dict or None (disabled).
    """
    if benchmark is not None:
        return _normalize_bench_cfg(benchmark)
    import os

    if os.environ.get("PYPTO_LIB_BENCHMARK"):
        return {
            "rounds": int(os.environ.get("PYPTO_LIB_BENCHMARK_ROUNDS", "100")),
            "warmup": int(os.environ.get("PYPTO_LIB_BENCHMARK_WARMUP", "3")),
        }
    return None


def _run_benchmark(
    compiled: Any,
    specs: list[TensorSpec | ScalarSpec],
    tensors: dict[str, torch.Tensor],
    scalar_specs_eff: dict[str, ScalarSpec],
    runtime_cfg: dict[str, Any],
    bench_cfg: dict[str, Any],
) -> Any:
    """Register *compiled* once and time ``rounds`` launches via pypto's helper.

    Mirrors simpler's ``scene_test --rounds`` mode: a single
    :func:`pypto.runtime.benchmark` registers the program once and dispatches
    ``warmup + rounds`` cheap launches, returning per-launch ``device_wall_us``
    samples (the on-NPU orchestrator wall). The dispatch args are the full
    positional list in orchestration param order — identical to what
    :func:`_execute_via_runner` hands ``execute_compiled`` — so the in-place
    (non-return) calling convention is preserved.

    Returns a ``BenchmarkStats``. Raises ``ValueError`` if *compiled* is None
    (e.g. an L2 ``runtime_dir`` replay, where no live compiled object exists).
    """
    if compiled is None:
        raise ValueError(
            "benchmark requires a freshly compiled program; it is unsupported on the "
            "L2 runtime_dir replay path (no live CompiledProgram to register)"
        )
    # pypto's benchmark() is built on ChipWorker.run_timed, which is L2
    # single-chip only: the base Worker.run_timed raises for L3, because an L3
    # (multi-card) DAG run does not aggregate device_wall_us (it would be 0).
    # Reject an L3 DistributedCompiledProgram up-front with a clear message
    # rather than letting ChipWorker.register fail with an opaque error.
    try:
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram
    except ImportError:
        DistributedCompiledProgram = ()  # type: ignore[assignment]
    if isinstance(compiled, DistributedCompiledProgram):
        raise ValueError(
            "benchmark is unsupported for L3 distributed (multi-card) programs: "
            "run_timed exposes a device wall only on ChipWorker (L2 single-chip)"
        )
    from pypto.runtime import benchmark as pypto_benchmark

    ordered: list[Any] = [
        tensors[s.name] if isinstance(s, TensorSpec) else scalar_specs_eff[s.name].to_ctypes()
        for s in specs
    ]
    stats = pypto_benchmark(
        compiled,
        ordered,
        rounds=int(bench_cfg.get("rounds", 100)),
        warmup=int(bench_cfg.get("warmup", 3)),
        platform=runtime_cfg.get("platform"),
        device_id=runtime_cfg.get("device_id"),
    )
    # Single-line, machine-readable marker the daily-CI perf collector greps for.
    # It is anchored on ``kernel=`` (not the bare "[RUN] benchmark" prefix) so it
    # is unambiguous against the _Stage("benchmark") start/done lines and the
    # "[RUN] benchmark skipped" warning. ``kernel`` is the compiled program's
    # output-dir basename with the per-run ``_YYYYMMDD_HHMMSS`` timestamp stripped
    # so the report row is stable day-to-day (trend tracking joins on a constant
    # name); a file running several kernels emits one line each.
    import re

    kernel = Path(compiled.output_dir).name if getattr(compiled, "output_dir", None) else "unknown"
    kernel = re.sub(r"_\d{8}_\d{6}$", "", kernel)
    if stats.all_zero_device:
        print(
            f"[RUN] benchmark kernel={kernel} no_device_timing=1 rounds={stats.rounds} "
            "(device_wall_us all 0 — runtime built without PTO2_PROFILING)",
            flush=True,
        )
    else:
        print(
            f"[RUN] benchmark kernel={kernel} rounds={stats.rounds} "
            f"mean_us={stats.device_us_mean:.0f} "
            f"min_us={stats.device_us_min:.0f} "
            f"max_us={stats.device_us_max:.0f}",
            flush=True,
        )
    return stats


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
    _install_simpler_chip_contexts_compat()
    _inherit_l3_next_level_runtime_config(compiled)
    compiled(*ordered, config=PyptoRunConfig(**kwargs))
    return True


def _maybe_reload_l3(
    work_dir: Path,
    runtime_cfg: dict[str, Any],
    compile_cfg: dict[str, Any],
) -> Any:
    """Reconstruct an L3 ``DistributedCompiledProgram`` from a ``runtime_dir``.

    Returns ``None`` for a single-chip / L2 build (which keeps using
    ``execute_compiled``). An L3 build is identified by the
    ``distributed_meta.json`` sidecar written at compile time (pypto #1689);
    :meth:`DistributedCompiledProgram.from_dir` rebuilds its metadata without
    re-running the pypto compile, so the existing :func:`_try_l3_dispatch` path
    can dispatch it. The run's ``platform`` and ``distributed_config`` override
    the values persisted at compile time, so ``--runtime-dir ... -p a2a3
    -d 2,3`` replays on the requested target / devices.
    """
    if not (work_dir / "distributed_meta.json").exists():
        return None
    # The meta sidecar proves this is an L3 build, so a missing
    # DistributedCompiledProgram is an unusable-pypto error, not a single-chip
    # fallback: surface it explicitly instead of returning None and failing
    # later in execute_compiled with a confusing single-chip error.
    try:
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram
    except ImportError as e:
        raise ImportError(
            "L3 build detected (distributed_meta.json present), but "
            "DistributedCompiledProgram could not be imported. Ensure your "
            "pypto installation supports L3 distributed execution."
        ) from e
    return DistributedCompiledProgram.from_dir(
        work_dir,
        platform=runtime_cfg.get("platform"),
        distributed_config=compile_cfg.get("distributed_config"),
    )


def _compute_golden(
    specs: list[TensorSpec | ScalarSpec],
    tensor_specs: list[TensorSpec],
    scalar_specs_eff: dict[str, ScalarSpec],
    input_snapshot: dict[str, torch.Tensor],
    work_dir: Path,
    data_dir: Path | None,
    golden_fn: Callable | None,
    save_data: bool = True,
) -> dict[str, torch.Tensor]:
    """Produce golden output tensors for validation.

    With *data_dir* set, load from ``{data_dir}/out/``. Otherwise call
    *golden_fn* on a scratch dict (inputs cloned from *input_snapshot*,
    outputs zero-init) and, when *save_data* is True, persist results into
    ``{work_dir}/data/out/``.
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
        if save_data:
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
    save_data: bool = True,
    save_actual_data: bool = False,
    benchmark: "bool | dict[str, Any] | None" = None,
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
        save_data: When True (default), persist generated inputs to
            ``{work_dir}/data/in/`` and golden outputs to
            ``{work_dir}/data/out/`` for later replay via *golden_data*. Set
            False to skip the on-disk ``.pt`` snapshot when inputs are large
            (e.g. full-model weights) and replay is not needed; validation
            still runs against the in-memory golden.
        save_actual_data: When True with *golden_data*, also persist runtime
            outputs to ``{work_dir}/data/actual`` for downstream consumers.
        benchmark: When truthy, after the normal validate run, register the
            compiled program once and time ``rounds`` launches via
            :func:`pypto.runtime.benchmark`. Pass ``True`` for defaults
            (100 rounds / 3 warmup) or a kwargs dict
            (``{"rounds": N, "warmup": M}``); see :func:`run_jit`.

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
    try:
        bench_cfg = _resolve_bench_cfg(benchmark)
    except ValueError as e:
        return RunResult(passed=False, error=str(e))

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
        # An L3 build has no live compiled object here (compile was skipped);
        # reconstruct it from the build dir so the L3 dispatch path below runs
        # instead of falling through to the single-chip execute_compiled.
        compiled = _maybe_reload_l3(work_dir, runtime_cfg, compile_cfg)
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
    if work_dir is not None:
        _patch_aicore_bitcast_helpers(work_dir)
        _patch_l3_single_submit_host_orch(work_dir)
        _patch_l3_host_orch_ssa_aliases(work_dir)

    # Generate Inputs
    try:
        with _Stage("generate inputs"):
            tensors, scalar_specs_eff, input_snapshot = _prepare_inputs(
                specs, tensor_specs, scalar_specs, data_dir, work_dir, save_data,
            )
    except ValueError as e:
        return _fail(str(e))

    # Compute Golden
    golden_outputs: dict[str, torch.Tensor] | None = None
    if golden_fn is not None or golden_data is not None:
        golden_outputs = _compute_golden(
            specs, tensor_specs, scalar_specs_eff, input_snapshot,
            work_dir, data_dir, golden_fn, save_data,
        )

    # Runtime
    with _Stage("runtime"):
        if compiled is None or not _try_l3_dispatch(
            compiled, specs, tensors, scalar_specs_eff, runtime_cfg,
        ):
            _execute_via_runner(work_dir, specs, tensors, scalar_specs_eff, runtime_cfg)

    # Validate
    validation_skipped = golden_outputs is None
    if save_data and (data_dir is None or save_actual_data):
        actual_outputs = {spec.name: tensors[spec.name] for spec in tensor_specs if spec.is_output}
        _save_tensors(work_dir / "data" / "actual", actual_outputs)
    if not validation_skipped:
        try:
            _validate(tensor_specs, tensors, golden_outputs, rtol, atol, compare_fn)
        except AssertionError as e:
            return _fail(str(e))

    # Benchmark (register-once, rounds timing). Runs after validation so the
    # timed kernel is the one we just proved correct. It is a measurement
    # add-on, never a correctness gate: a benchmark failure (e.g. an L3
    # distributed program, which run_timed does not support) must not flip a
    # validated-correct run to FAIL, so swallow it with a warning.
    bench_stats = None
    if bench_cfg is not None:
        try:
            with _Stage("benchmark"):
                bench_stats = _run_benchmark(
                    compiled, specs, tensors, scalar_specs_eff, runtime_cfg, bench_cfg,
                )
        except Exception as e:  # noqa: BLE001 — benchmark is never a correctness gate
            # Any benchmark failure (L3 unsupported, device hiccup, ...) must not
            # flip a validated-correct run to FAIL; warn and keep the verdict.
            print(f"[RUN] benchmark skipped: {type(e).__name__}: {e}", flush=True)

    total = time.time() - start
    skip_note = ", validation skipped: no golden_fn or golden_data" if validation_skipped else ""
    print(f"[RUN] PASS ({total:.2f}s{skip_note})", flush=True)
    return RunResult(passed=True, execution_time=total, work_dir=work_dir, bench_stats=bench_stats)


def _compile_jit_with_compat(fn: Any, dummy_args: list[Any], cfg: dict[str, Any]) -> Any:
    """Compile a ``@pl.jit`` entry across pypto JIT API variants.

    Prefer the public ``fn.compile(...)`` entry when available. Older/newer
    pypto builds may expose only the specialization helpers on ``JITFunction``;
    in that case, materialize the pre-pass ``ir.Program`` and feed it through
    public :func:`pypto.ir.compile` so the harness still gets a normal compiled
    artifact with an ``output_dir``.
    """
    from pypto.runtime import RunConfig

    run_config = RunConfig(**cfg)
    compile_method = getattr(fn, "compile", None)
    if callable(compile_method):
        return compile_method(*dummy_args, config=run_config)

    bind_args = getattr(fn, "_bind_args", None)
    compile_to_program = getattr(fn, "_compile_to_program", None)
    if not callable(bind_args) or not callable(compile_to_program):
        raise AttributeError(
            "JIT function does not expose compile(), and the compatibility "
            "fallback requires _bind_args() and _compile_to_program()."
        )

    import pypto.language as pl
    from pypto import ir

    _, _, tensor_meta, scalar_values, scalar_dtypes, per_func_dyn = bind_args(tuple(dummy_args), {})
    program = compile_to_program(tensor_meta, scalar_values, scalar_dtypes, per_func_dyn, pl)
    return ir.compile(
        program,
        output_dir=run_config.save_kernels_dir,
        strategy=run_config.strategy,
        backend_type=run_config.backend_type,
        dump_passes=run_config.dump_passes,
        diagnostic_phase=run_config.diagnostic_phase,
        disabled_diagnostics=run_config.disabled_diagnostics,
        platform=run_config.platform,
        profiling=run_config.compile_profiling,
    )


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
    save_data: bool = True,
    save_actual_data: bool = False,
    benchmark: "bool | dict[str, Any] | None" = None,
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
        compile_cfg: Compile-side ``RunConfig`` fields (``dump_passes`` /
            ``distributed_config`` / ``compile_profiling`` / ...) carried into
            JIT compilation; ``platform`` is supplied separately
            (typically via *runtime_cfg*). Unknown keys raise when the
            ``RunConfig`` is built.
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
        save_data: When True (default), persist generated inputs to
            ``{work_dir}/data/in/`` and golden outputs to
            ``{work_dir}/data/out/`` for later replay via *golden_data*. Set
            False to skip the on-disk ``.pt`` snapshot when inputs are large
            (e.g. full-model weights) and replay is not needed; validation
            still runs against the in-memory golden.
        save_actual_data: When True with *golden_data*, also persist runtime
            outputs to ``{work_dir}/data/actual`` for downstream consumers.
        benchmark: When truthy, after the normal validate run, register the
            compiled program once and time ``rounds`` launches via
            :func:`pypto.runtime.benchmark` (simpler's ``scene_test --rounds``
            mode). Pass ``True`` for defaults (100 rounds / 3 warmup) or a
            kwargs dict (``{"rounds": N, "warmup": M}``). The aggregated
            ``BenchmarkStats`` is attached to :attr:`RunResult.bench_stats`.
            Unsupported on the L2 ``runtime_dir`` replay path (no live
            CompiledProgram to register).

    Returns:
        :class:`RunResult`.
    """
    compile_cfg = compile_cfg or {}
    runtime_cfg = dict(runtime_cfg or {})  # copy: we pop harness-only keys
    compare_fn = compare_fn or {}

    _consume_runtime_harness_keys(runtime_cfg)

    if compile_only and runtime_dir is not None:
        return RunResult(passed=False, error="runtime_dir is incompatible with compile_only")
    try:
        bench_cfg = _resolve_bench_cfg(benchmark)
    except ValueError as e:
        return RunResult(passed=False, error=str(e))

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
    compiled: Any = None  # the CompiledProgram, when we compiled it this call
    if runtime_dir is not None:
        try:
            work_dir = _setup_runtime_dir(runtime_dir, compile_label="JIT compile")
        except ValueError as e:
            return _fail(str(e))
        # An L3 build has no live compiled object here (JIT compile was skipped);
        # reconstruct it from the build dir so the L3 dispatch path below runs
        # instead of falling through to the single-chip execute_compiled.
        compiled = _maybe_reload_l3(work_dir, runtime_cfg, compile_cfg)
    else:
        with _Stage("compile"):
            from pypto.runtime import RunConfig

            # Dummy args only carry shape/dtype (and scalar values) into the
            # specialization key; real tensors of the same shape hit the same
            # JIT cache entry at dispatch.
            dummy_args = [
                spec.value.item() if isinstance(spec, ScalarSpec)
                else torch.empty(spec.shape, dtype=spec.dtype)
                for spec in specs
            ]
            cfg = dict(compile_cfg)
            platform = runtime_cfg.get("platform")
            if platform is not None:
                cfg["platform"] = platform
            # Public compile-only entry: same specialize → cache → ir.compile
            # pipeline as __call__, minus on-device dispatch. Returns a
            # DistributedCompiledProgram for an L3 host orchestrator.
            compiled = _compile_jit_with_compat(fn, dummy_args, cfg)
            work_dir = Path(compiled.output_dir)
        if compile_only:
            total = time.time() - start
            print(f"[RUN] PASS ({total:.2f}s)", flush=True)
            return RunResult(passed=True, execution_time=total, work_dir=work_dir)

    # Generate Inputs
    try:
        with _Stage("generate inputs"):
            tensors, scalar_specs_eff, input_snapshot = _prepare_inputs(
                specs, tensor_specs, scalar_specs, data_dir, work_dir, save_data,
            )
    except ValueError as e:
        return _fail(str(e))

    # Compute Golden
    golden_outputs: dict[str, torch.Tensor] | None = None
    if golden_fn is not None or golden_data is not None:
        golden_outputs = _compute_golden(
            specs, tensor_specs, scalar_specs_eff, input_snapshot,
            work_dir, data_dir, golden_fn, save_data,
        )

    # Runtime
    with _Stage("runtime"):
        # An L3 ``DistributedCompiledProgram`` (a @pl.jit.host kernel compiled
        # with distributed_config) dispatches per-rank via _try_l3_dispatch;
        # everything else runs through the single-chip runner.
        if compiled is None or not _try_l3_dispatch(
            compiled, specs, tensors, scalar_specs_eff, runtime_cfg,
        ):
            _execute_via_runner(work_dir, specs, tensors, scalar_specs_eff, runtime_cfg)

    # Validate
    validation_skipped = golden_outputs is None
    if save_data and (data_dir is None or save_actual_data):
        actual_outputs = {spec.name: tensors[spec.name] for spec in tensor_specs if spec.is_output}
        _save_tensors(work_dir / "data" / "actual", actual_outputs)
    if not validation_skipped:
        try:
            _validate(tensor_specs, tensors, golden_outputs, rtol, atol, compare_fn)
        except AssertionError as e:
            return _fail(str(e))

    # Benchmark (register-once, rounds timing). Runs after validation so the
    # timed kernel is the one we just proved correct. It is a measurement
    # add-on, never a correctness gate: a benchmark failure (e.g. an L3
    # distributed program, which run_timed does not support) must not flip a
    # validated-correct run to FAIL, so swallow it with a warning.
    bench_stats = None
    if bench_cfg is not None:
        try:
            with _Stage("benchmark"):
                bench_stats = _run_benchmark(
                    compiled, specs, tensors, scalar_specs_eff, runtime_cfg, bench_cfg,
                )
        except Exception as e:  # noqa: BLE001 — benchmark is never a correctness gate
            # Any benchmark failure (L3 unsupported, device hiccup, ...) must not
            # flip a validated-correct run to FAIL; warn and keep the verdict.
            print(f"[RUN] benchmark skipped: {type(e).__name__}: {e}", flush=True)

    total = time.time() - start
    skip_note = ", validation skipped: no golden_fn or golden_data" if validation_skipped else ""
    print(f"[RUN] PASS ({total:.2f}s{skip_note})", flush=True)
    return RunResult(passed=True, execution_time=total, work_dir=work_dir, bench_stats=bench_stats)
