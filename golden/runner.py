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

import os
import re
import statistics
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
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
    bench: Any = None  # BenchmarkStats when PYPTO_BENCH timed the run; None otherwise

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
    - :class:`TensorSpec` inout: both ``in/{name}.pt`` and ``out/{name}.pt``.
    """
    if isinstance(spec, ScalarSpec):
        return [("in", f"{spec.name}.pt")]
    files: list[tuple[str, str]] = []
    if spec.is_input:
        files.append(("in", f"{spec.name}.pt"))
    if spec.is_output:
        files.append(("out", f"{spec.name}.pt"))
    return files


def _require_files(data_dir: Path, required: Iterable[tuple[str, str]]) -> None:
    """Raise ``ValueError`` listing every missing ``data_dir/{subdir}/{name}``."""
    missing = [
        str(data_dir / sub / name)
        for sub, name in required
        if not (data_dir / sub / name).is_file()
    ]
    if missing:
        raise ValueError(f"golden_data is missing files: {missing}")


def _reject_stepped(
    scalar_specs: Iterable[ScalarSpec],
    reason: str,
    *,
    only_specialized: bool = False,
) -> None:
    """Raise ``ValueError`` naming every stepped scalar in *scalar_specs*.

    *reason* is the caller's explanation; the offending names are appended to
    it. With *only_specialized*, a ``compile_runtime`` scalar is exempt: it
    survives specialization as a runtime parameter, so stepping it is legal.
    """
    stepped = sorted(
        spec.name for spec in scalar_specs
        if spec.has_benchmark_step and not (only_specialized and spec.compile_runtime)
    )
    if stepped:
        raise ValueError(f"{reason}; stepped scalars: {stepped}")


@contextmanager
def _log_level_scope(config: Any | None) -> Iterator[None]:
    """Restore the runtime log threshold after a run that overrode it.

    ``configure_log`` sets a process-global logger level, so a *config* carrying
    ``log_level`` would otherwise leak into every later :func:`run` in this
    process — including the calls that set no level of their own. That is the
    batched sweep the benchmarking guide asks for, and v9 is the ``[STRACE]``
    band, so the leak costs log volume on rounds meant to be timed.

    A no-op when *config* sets no ``log_level``: nothing was changed, so there
    is nothing to put back.
    """
    if not (isinstance(config, dict) and config.get("log_level") is not None):
        yield
        return
    from pypto.runtime.log_config import configure_log, current_level

    prior = current_level()
    try:
        yield
    finally:
        configure_log(prior)


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


def _normalize_config(config: Any | None) -> Any:
    """Build the one ``RunConfig`` that drives both phases.

    PyPTO reads the compile half off it via ``compile_kwargs()`` and the
    dispatch half via ``run_options()`` / ``dfx_options()``, and
    ``JITFunction.compile`` / ``CompiledProgram.__call__`` / ``benchmark`` all
    take the aggregate — so the harness translates once, here, and every phase
    reads the same object. There is no compile-vs-runtime split at the call
    site because there is no split downstream.

    *config* is a ``dict`` of ``RunConfig`` keyword arguments, a ready
    ``RunConfig``, or ``None``. ``log_level`` is the one key that is not a
    ``RunConfig`` field: it sets the PyPTO runtime log threshold and is applied
    here. Every other key goes straight to the constructor, so an unknown one
    is ``RunConfig``'s own ``TypeError`` naming it.
    """
    from pypto.runtime import RunConfig

    if not isinstance(config, dict):
        return config if config is not None else RunConfig()

    fields = dict(config)
    level = fields.pop("log_level", None)
    if level is not None:
        from pypto.runtime.log_config import configure_log
        configure_log(level)
    return RunConfig(**fields)


def _validate_stepped_swimlane(
    scalar_specs: Sequence[ScalarSpec], cfg: Any
) -> None:
    """Reject stepped scalars when one handle call launches multiple passes.

    An onboard swimlane capture runs the workload twice — a dep_gen pass for
    ``deps.json``, then a clean timing pass — and a stepped scalar cannot
    advance between two physical passes of one handle call. Both would carry
    dispatch 0's value and the swimlane would be silently wrong.
    """
    if not cfg.enable_chip_swimlane:
        return
    _reject_stepped(
        scalar_specs,
        "ScalarSpec benchmark_step is incompatible with enable_chip_swimlane; "
        "one handle call may launch multiple physical passes with the same "
        "scalar values",
    )


def _stale_cpps(work_dir: Path) -> list[Path]:
    """Return cpps under ``kernels/`` / ``orchestration/`` that need rebuilding.

    A cpp is considered stale if **either**:

    - its sibling ``.so``/``.o`` is missing entirely (binary never built or
      removed by hand), **or**
    - any existing sibling ``.so``/``.o`` is older than the cpp itself
      (cpp was edited after its last build).

    Both cases require a rebuild, so the caller's log line must report them
    together.
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


def _validate_unique_spec_names(specs: list[TensorSpec | ScalarSpec]) -> None:
    """Reject names that would collapse when specs are converted to mappings."""
    names = [spec.name for spec in specs]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(f"duplicate spec names: {duplicate_names}")


def _effective_scalar_specs(
    scalar_specs: list[ScalarSpec],
    data_dir: Path | None,
) -> dict[str, ScalarSpec]:
    """Return scalar specs with golden-data values applied, preserving flags."""
    if data_dir is None:
        return {spec.name: spec for spec in scalar_specs}

    _require_files(data_dir, (("in", f"{spec.name}.pt") for spec in scalar_specs))

    effective = {}
    for spec in scalar_specs:
        cached = torch.load(data_dir / "in" / f"{spec.name}.pt", weights_only=True)
        if not isinstance(cached, torch.Tensor) or cached.ndim != 0:
            shape = tuple(cached.shape) if isinstance(cached, torch.Tensor) else type(cached).__name__
            raise ValueError(f"{spec.name}.pt must contain a 0-dim torch.Tensor, got {shape}")
        if cached.dtype != spec.dtype:
            raise ValueError(
                f"{spec.name}.pt dtype mismatch: spec={spec.dtype} cache={cached.dtype}"
            )
        effective[spec.name] = ScalarSpec(
            name=spec.name,
            dtype=spec.dtype,
            value=cached,
            compile_runtime=spec.compile_runtime,
            benchmark_step=spec.benchmark_step,
        )
    return effective


def _prepare_inputs(
    specs: list[TensorSpec | ScalarSpec],
    tensor_specs: list[TensorSpec],
    scalar_specs: list[ScalarSpec],
    data_dir: Path | None,
    work_dir: Path,
    save_data: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, ScalarSpec]]:
    """Build the dispatch buffers and effective scalars for the runtime stage.

    With *data_dir* set, load tensors and scalars from ``{data_dir}/in/``.
    Otherwise generate from *specs* and, when *save_data* is True, persist into
    ``{work_dir}/data/in/``. Set *save_data* False to skip the on-disk ``.pt``
    snapshot (validation still runs against the in-memory golden); useful when
    inputs are large (e.g. full-model weights) and golden replay is not needed.

    The returned tensors are the buffers the device is handed, and nothing
    writes to them before :func:`_dispatch` — :func:`_compute_golden` runs
    first and clones what it needs — so no separate pristine copy is kept.

    Raises ``ValueError`` on missing files or scalar dtype mismatch.
    """
    _validate_unique_spec_names(specs)

    if data_dir is None:
        tensors = {spec.name: spec.create_tensor() for spec in tensor_specs}
        scalar_specs_eff = _effective_scalar_specs(scalar_specs, data_dir)
        if save_data:
            in_dir = work_dir / "data" / "in"
            _save_tensors(
                in_dir,
                {s.name: tensors[s.name] for s in tensor_specs if s.is_input},
            )
            _save_tensors(in_dir, {s.name: s.value for s in scalar_specs})
        return tensors, scalar_specs_eff

    required: list[tuple[str, str]] = []
    for spec in (*tensor_specs, *scalar_specs):
        required.extend(_required_files(spec))
    _require_files(data_dir, required)
    print(f"[RUN]   cache hit: {data_dir / 'in'}", flush=True)

    # Load inputs + inout initial values from {dir}/in/. A pure output carries
    # no input data, so its host buffer -- the read-back destination -- stays
    # zero-init rather than re-running the spec's init_value.
    input_names = [s.name for s in tensor_specs if s.is_input]
    tensors = _load_tensors(data_dir, "in", input_names)
    for spec in tensor_specs:
        if not spec.is_input:
            tensors[spec.name] = torch.zeros(spec.shape, dtype=spec.dtype)

    scalar_specs_eff = _effective_scalar_specs(scalar_specs, data_dir)

    return tensors, scalar_specs_eff


def _ordered_args(
    specs: list[TensorSpec | ScalarSpec],
    tensors: dict[str, torch.Tensor],
    scalar_specs_eff: dict[str, ScalarSpec],
    *,
    ctypes_scalars: bool,
    benchmark_dispatch_index: int | None = None,
    overrides: dict[str, Any] | None = None,
) -> list[Any]:
    """Positional dispatch args in spec order.

    Spec order *is* the compiled parameter order:
    :func:`_validate_compiled_spec_abi` rejects any artifact whose parameter
    names differ from the spec names element by element, so no name-keyed
    reordering is needed here.

    An L2 dispatch takes ctypes scalars; an L3 dispatch takes the 0-dim
    value tensor. *benchmark_dispatch_index* advances a stepped scalar to its
    value for that physical benchmark dispatch. *overrides* substitutes a
    device-side handle for a tensor spec's host buffer, which is how the
    resident path passes its ``DeviceTensor`` / ``StackedDeviceTensor``.
    """
    overrides = overrides or {}
    args: list[Any] = []
    for spec in specs:
        if isinstance(spec, TensorSpec):
            args.append(
                overrides[spec.name] if spec.name in overrides else tensors[spec.name]
            )
            continue
        scalar = scalar_specs_eff[spec.name]
        if ctypes_scalars:
            args.append(scalar.to_ctypes())
        elif benchmark_dispatch_index is None:
            args.append(scalar.value)
        else:
            args.append(scalar.value_for_benchmark_dispatch(benchmark_dispatch_index))
    return args


def _dispatch(
    compiled: Any,
    specs: list[TensorSpec | ScalarSpec],
    tensors: dict[str, torch.Tensor],
    scalar_specs_eff: dict[str, ScalarSpec],
    cfg: Any,
) -> None:
    """Dispatch *compiled* once in orchestration param order.

    One call shape for both levels: ``CompiledProgram`` and
    ``DistributedCompiledProgram`` are both callable with a ``RunConfig``. They
    differ only in scalar marshalling — L3 reads args through the fork-inherited
    shared mapping and takes Python scalars, L2 takes ctypes.
    """
    ordered = _ordered_args(
        specs, tensors, scalar_specs_eff, ctypes_scalars=not _is_l3(compiled)
    )
    compiled(*ordered, config=cfg)


def _is_l3(compiled: Any) -> bool:
    """True if *compiled* is an L3 ``DistributedCompiledProgram`` (not L2 single-chip).

    Used to route benchmarking: L2 goes through :func:`_run_benchmark`
    (``ChipWorker``); L3 goes through :func:`_run_benchmark_l3` (non-resident) or
    :func:`_run_l3_resident` (resident), which fold the forked chip workers'
    per-rank ``[STRACE]`` markers into per-round timing.
    """
    try:
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram
    except ImportError:
        return False
    return isinstance(compiled, DistributedCompiledProgram)


# Default benchmark loop sizes shared by L2 and L3, overridable per run via
# PYPTO_BENCH_ROUNDS / PYPTO_BENCH_WARMUP (see :func:`_bench_loop_sizes`). Daily
# CI pins the perf baseline by leaving both unset. L3 differs only in its
# aggregation: each round contributes the fastest valid rank's Effective time.
_BENCH_ROUNDS_DEFAULT = 100
_BENCH_WARMUP_DEFAULT = 5


def _env_flag(name: str) -> bool:
    """True when env var *name* holds anything but empty / ``0`` / ``false``."""
    return os.environ.get(name, "").strip() not in ("", "0", "false", "False")


def _bench_enabled() -> bool:
    """True when ``PYPTO_BENCH`` is set truthy.

    Benchmarking is entirely env-driven so no model file needs a ``--benchmark``
    flag and ``run`` needs no extra parameters: daily CI's a2a3 job sets
    ``PYPTO_BENCH=1`` and every ``run`` call then times the kernel over
    :func:`_bench_loop_sizes` rounds (warmup discarded).
    """
    return _env_flag("PYPTO_BENCH")


def _bench_env_int(name: str, default: int, minimum: int) -> int:
    """Read env var *name* as an int >= *minimum*, falling back to *default*.

    A malformed or out-of-range value warns and uses the default rather than
    raising: a mistyped tuning knob must not fail an otherwise good run.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = minimum - 1
    if value < minimum:
        print(
            f"[RUN]   ignoring {name}={raw!r} (want an integer >= {minimum}); "
            f"using {default}",
            flush=True,
        )
        return default
    return value


def _bench_loop_sizes() -> tuple[int, int]:
    """``(rounds, warmup)`` for this run, from the env or the defaults.

    Overriding matters because one size does not fit every kernel: the
    100-round default is ~0.1 s of device time for a decode step but minutes for
    a long prefill or a multi-card L3 run, and while iterating on a kernel a
    handful of rounds is usually enough. Both are read per run (not cached), so
    a sweep can vary them between :func:`run` calls in one process.

    Daily CI sets neither, so its numbers stay comparable across runs. Warmup is
    allowed to be 0; rounds must be at least 1.
    """
    return (
        _bench_env_int("PYPTO_BENCH_ROUNDS", _BENCH_ROUNDS_DEFAULT, 1),
        _bench_env_int("PYPTO_BENCH_WARMUP", _BENCH_WARMUP_DEFAULT, 0),
    )


def _resident_loop_sizes() -> tuple[int, int]:
    """:func:`_bench_loop_sizes` with ``warmup`` forced to at least 1.

    The resident L3 path spends its first warmup launch on the validation
    dispatch, so ``warmup=0`` would emit ``rounds + 1`` dispatches per rank
    against a declared ``rounds + 0`` and stop segmenting evenly.
    """
    rounds, warmup = _bench_loop_sizes()
    return rounds, max(warmup, 1)


def _bench_raw_enabled() -> bool:
    """True when ``PYPTO_BENCH_RAW`` is set truthy.

    Opt-in companion to :func:`_bench_enabled`, off by default (the raw dump is
    one line per rank holding every sample). Turn it on when a summary looks
    suspicious — start-up drift, a bimodal rank, one card lagging — and the
    individual samples are needed to see the shape.
    """
    return _env_flag("PYPTO_BENCH_RAW")


def _benchmark_unavailable(error: RuntimeError) -> bool:
    """True only for the optional-profiling failure emitted by the runtime."""
    return "no [STRACE] markers captured" in str(error)


def _run_benchmark(
    compiled: Any,
    specs: list[TensorSpec | ScalarSpec],
    tensors: dict[str, torch.Tensor],
    scalar_specs_eff: dict[str, ScalarSpec],
    cfg: Any,
    rounds: int,
    warmup: int,
) -> Any:
    """:func:`_benchmark_and_report` for an L2 single-chip ``CompiledProgram``.

    ``benchmark`` opens one :class:`~pypto.runtime.ChipWorker` for it.
    """
    _reject_stepped(
        scalar_specs_eff.values(),
        "L2 benchmark does not support ScalarSpec benchmark_step",
    )
    return _benchmark_and_report(
        compiled, specs, tensors, scalar_specs_eff, cfg, rounds, warmup, l3=False
    )


def _benchmark_and_report(
    compiled: Any,
    specs: list[TensorSpec | ScalarSpec],
    tensors: dict[str, torch.Tensor],
    scalar_specs_eff: dict[str, ScalarSpec],
    cfg: Any,
    rounds: int,
    warmup: int,
    *,
    l3: bool,
) -> Any:
    """Register *compiled* once, time *rounds* launches, print the report block.

    Delegates to :func:`pypto.runtime.benchmark`, which registers once and reads
    each launch's on-NPU span tree from the runtime's ``[STRACE]`` markers. Args
    are built in spec order by :func:`_ordered_args`, exactly as :func:`_dispatch`
    does. The two levels differ only in marshalling — an L3 dispatch reads IO
    through the fork-inherited shared mapping and takes the 0-dim value tensor,
    an L2 dispatch takes ctypes — and in keeping CommDomains across rounds.
    Returns the :class:`~pypto.runtime.BenchmarkStats`, or ``None`` when the
    runtime emits no markers (built without ``SIMPLER_PROFILING``).
    """
    from pypto.runtime import benchmark

    if l3:
        # The forked workers read these buffers through the inherited mapping;
        # validation (already done) read the device outputs back from them.
        _share_in_place(tensors)
    ordered = _ordered_args(specs, tensors, scalar_specs_eff, ctypes_scalars=not l3)
    # Benchmarks model serving's steady-state dispatch: retain CommDomains
    # across rounds and let kernels clear their own signal windows.
    persistent = {"persistent": True, "reset_persistent_windows": False} if l3 else {}
    stats = None
    with _Stage("benchmark"):
        try:
            # The benchmark is a second, independent dispatch, so it takes the
            # same config as the correctness run — a program that sizes its
            # rings for one must size them for the other or it validates and
            # then deadlocks.
            stats = benchmark(
                compiled, ordered,
                rounds=rounds, warmup=warmup,
                config=cfg,
                **persistent,
            )
        except RuntimeError as e:
            if not _benchmark_unavailable(e):
                raise
            print(f"[RUN]   benchmark unavailable: {e}", flush=True)
            stats = None
    if stats is None:
        return None
    _report_bench(stats, compiled, l3=l3, resident=False)
    return stats


def _report_bench(stats: Any, compiled: Any, *, l3: bool, resident: bool) -> None:
    """Print one benchmark run's report block.

    Four blocks: the ``effective_us`` headline, L3's per-rank table, the opt-in
    raw dump, and L3's context line. The headline goes first and is the only
    line spelling the metric out; every breakdown line says ``eff_us``, so one
    glance down a 60-line multi-card dump finds the headline number. Daily CI
    matches that line's full ``(N rounds) min=... mean=...`` shape, so the
    spelling is a reader affordance, not a constraint on these lines.
    """
    _report_effective(stats)
    if l3:
        _report_l3_per_rank(stats)
    _report_raw_samples(stats)
    _report_task_slots(stats)
    if l3:
        _report_l3_detail(stats, compiled, resident=resident)


_TASK_SLOT_RE = re.compile(r"\.task_slot_(\d+)$")


def _report_task_slots(stats: Any) -> None:
    """Print per-rank task-timing slot windows captured during the benchmark.

    No-op unless the orchestration tags tasks with ``set_task_timing_slot``.
    Per slot: ``fin_us`` is the slot's finish (``ts + dur``) from the run's
    device-clock origin, ``dur_us`` its own dispatch-to-finish window, and
    ``dfin_us`` the per-round finish minus the previous tagged slot's finish.
    Repeated spans of one slot in a dispatch merge into one
    ``min(ts)``..``max(ts + dur)`` window, and ``dfin_us`` pairs only the
    dispatches that carry both slots. Each is the median over measured
    dispatches; ``PYPTO_BENCH_RAW`` adds the per-dispatch ``fin_us`` lists.
    """
    by_pid: dict[int, dict[int, dict[int, tuple[float, float]]]] = {}
    for iv in sorted(stats.invocations or [], key=lambda i: (i.pid, i.inv)):
        windows: dict[int, tuple[float, float]] = {}
        for span in iv.spans:
            m = _TASK_SLOT_RE.search(span.name)
            if m and span.is_device:
                slot = int(m.group(1))
                start, fin = span.ts, span.ts + span.dur
                if slot in windows:
                    start = min(start, windows[slot][0])
                    fin = max(fin, windows[slot][1])
                windows[slot] = (start, fin)
        for slot, (start, fin) in windows.items():
            rank = by_pid.setdefault(iv.pid, {})
            rank.setdefault(slot, {})[iv.inv] = (fin / 1000.0, (fin - start) / 1000.0)
    if not by_pid:
        return
    print(f"[RUN]   task slots: ranks={len(by_pid)}", flush=True)
    for pid in sorted(by_pid):
        slots = by_pid[pid]
        prev = None
        for slot in sorted(slots):
            fins = [fin for fin, _ in slots[slot].values()]
            durs = [dur for _, dur in slots[slot].values()]
            line = (
                f"[RUN]     rank {pid} task_slot {slot}: n={len(fins)} "
                f"fin_us={statistics.median(fins):.1f} dur_us={statistics.median(durs):.1f}"
            )
            if prev is not None:
                deltas = [
                    fin - slots[prev][inv][0]
                    for inv, (fin, _) in slots[slot].items()
                    if inv in slots[prev]
                ]
                if deltas:
                    line += f" dfin_us={statistics.median(deltas):.1f}"
            print(line, flush=True)
            if _bench_raw_enabled():
                print(f"[RUN]       raw fin_us={[round(f, 1) for f in fins]}", flush=True)
            prev = slot


def _eff_summary(samples: Any) -> tuple[int, str] | None:
    """``(n, "min=... median=... mean=... max=...")``, or ``None`` for no timing.

    The one place Effective samples are aggregated. Zeros are dropped first — a
    round with no orch/sched span reads 0 — so *n* is the surviving count, not
    ``len(samples)``.
    """
    eff = [e for e in samples if e > 0.0]
    if not eff:
        return None
    return len(eff), (
        f"min={min(eff):.1f} median={statistics.median(eff):.1f} "
        f"mean={statistics.fmean(eff):.1f} max={max(eff):.1f}"
    )


def _report_effective(stats: Any) -> None:
    """Print the max-rank ``effective_us (...)`` summary.

    Daily CI consumes this line for both L2 and L3. For L3 it is the per-round
    max across ranks (slowest rank bounds the round); the flatten fallback pools
    every rank's per-dispatch samples into the same window.

    The Effective window is the framework's post-graph-build execution window
    (``orch``∪``sched``), read from ``BenchmarkStats.per_round("effective")``
    so the span names come from the installed runtime. The aggregate covers the
    measured rounds; warmup is excluded.
    """
    if stats.all_zero_device:
        print(
            "[RUN]   effective_us unavailable: no device-domain spans "
            "(sim platform or non-profiling build)",
            flush=True,
        )
        return
    summary = _eff_summary(stats.per_round("effective"))
    if summary is None:
        print("[RUN]   effective_us unavailable: no orch/sched spans captured", flush=True)
        return
    rounds, line = summary
    print(f"[RUN]   effective_us ({rounds} rounds) {line}", flush=True)


def _report_raw_samples(stats: Any) -> None:
    """Print every measured dispatch's raw Effective sample, per rank.

    No-op unless :func:`_bench_raw_enabled` (``PYPTO_BENCH_RAW``). Reads
    :attr:`BenchmarkStats.invocations` — the flat per-dispatch list — rather than
    the per-round grid, so it works for L2 (one rank, one dispatch per round),
    for L3, and for the L3 flatten fallback where ``per_rank`` returns ``{}`` and
    the summary lines are the least trustworthy.

    Samples are in ``inv`` order (warmup already dropped), so the sequence shows
    drift directly. The lines carry a ``raw`` token and ``eff_us``, matching the
    breakdown-line spelling described in :func:`_report_bench`.
    """
    if not _bench_raw_enabled() or not stats.invocations:
        return
    by_pid: dict[int, list[Any]] = {}
    for iv in sorted(stats.invocations, key=lambda i: (i.pid, i.inv)):
        by_pid.setdefault(iv.pid, []).append(iv)
    head = (
        f"[RUN]   raw samples: ranks={len(by_pid)} rounds={stats.rounds} warmup={stats.warmup}"
    )
    if stats.fallback_flattened:
        head += " fallback_flattened=1"
    print(head, flush=True)
    for pid in sorted(by_pid):
        eff = [round(iv.effective_us, 1) for iv in by_pid[pid]]
        print(f"[RUN]     rank {pid} raw n={len(eff)} eff_us={eff}", flush=True)


def _report_l3_detail(stats: Any, compiled: Any, *, resident: bool) -> None:
    """Print an L3 context line complementing :func:`_report_effective`.

    Surfaces the L3-only aggregates the new ``BenchmarkStats`` exposes: the
    cross-rank host-timeline ``union`` window and the host wall — plus the rank
    count and a ``fallback_flattened`` note when per-round segmentation was not
    possible. The ``kernel=`` / ``l3_resident=1`` tokens are preserved for
    dashboards that grep them.
    """
    kernel = Path(compiled.output_dir).name if getattr(compiled, "output_dir", None) else "unknown"
    kernel = re.sub(r"_\d{8}_\d{6}$", "", kernel)
    n_ranks = len({pid for ranks in stats.rounds_dispatches for pid in ranks}) if stats.rounds_dispatches else 0
    parts = [
        f"[RUN] benchmark kernel={kernel}",
        "l3_resident=1" if resident else "l3=1",
        f"rounds={stats.rounds}",
        f"ranks={n_ranks}",
    ]
    union = stats.per_round("union")
    if union:
        parts.append(f"host_union_mean_us={statistics.fmean(union):.0f}")
    if stats.host_wall_us:
        parts.append(f"host_mean_us={statistics.fmean(stats.host_wall_us):.0f}")
    if stats.fallback_flattened:
        parts.append("fallback_flattened=1")
    print(" ".join(parts), flush=True)


def _report_l3_per_rank(stats: Any) -> None:
    """Print each rank's Effective summary for an L3 run, with its dispatches.

    Uses ``BenchmarkStats.per_rank("effective")`` — ``{pid: [per-round ...]}``
    where each round entry is that rank's summed dispatch Effective window — to
    surface the cross-card imbalance the headline (per-round max across ranks)
    hides. No-op for L2 and the flatten fallback (``per_rank`` returns ``{}``).

    A rank entry **sums** that card's dispatches, so each rank line is followed
    by one nested ``slot`` line per dispatch, labelled with the orchestration
    ``dispatch_tasks()`` name. ``slot`` is the dispatch's position within its
    rank's round, so slot ``s`` is the same dispatch in every round and nothing
    is summed. Slot lines appear for every rank or none, so the block stays a
    complete table.
    """
    rank_eff = stats.per_rank("effective")
    if not rank_eff:
        return

    def _line(label: str, samples: Any, indent: int) -> None:
        summary = _eff_summary(samples)
        body = f"eff_us {summary[1]}" if summary else "(no timing)"
        print(f"[RUN]{' ' * indent}{label}: {body}", flush=True)

    # Skip the slot breakdown when it would add nothing: an installed pypto
    # predating per_dispatch, no dispatch grid (L2 / flatten fallback), or no
    # rank issuing more than one dispatch per round (every slot line would just
    # restate its rank line).
    per_dispatch_fn = getattr(stats, "per_dispatch", None)
    per_dispatch = (per_dispatch_fn("effective") or {}) if per_dispatch_fn else {}
    if len(per_dispatch) <= len({pid for pid, _slot in per_dispatch}):
        per_dispatch = {}

    tasks = getattr(stats, "dispatch_tasks", dict)() or {}
    by_rank: dict[int, list[tuple[int, str, list[float]]]] = {}
    for key, samples in per_dispatch.items():
        pid, slot = key
        by_rank.setdefault(pid, []).append((slot, tasks.get(key, ""), samples))

    for pid in sorted(rank_eff):
        _line(f"rank {pid}", rank_eff[pid], 5)
        for slot, task, samples in sorted(by_rank.get(pid, []), key=lambda entry: entry[0]):
            _line(f"slot {slot}" + (f" ({task})" if task else ""), samples, 7)


def _run_benchmark_l3(
    compiled: Any,
    specs: list[TensorSpec | ScalarSpec],
    tensors: dict[str, torch.Tensor],
    scalar_specs_eff: dict[str, ScalarSpec],
    cfg: Any,
    rounds: int,
    warmup: int,
) -> Any:
    """:func:`_benchmark_and_report` for a non-resident L3 ``DistributedCompiledProgram``.

    ``benchmark`` opens a ``DistributedWorker`` via ``compiled.prepare()`` and
    folds the forked chip workers' per-rank ``[STRACE]`` markers into per-round
    samples. L3 rejects ``platform=``/``device_id=`` (the device set is fixed at
    compile time), so only ``config=`` is passed.
    """
    # Every stepped driver keeps resident weights and benchmarks on the
    # resident dispatch path, where each physical launch rebuilds its argument
    # list; here one args list would repeat across launches and violate the
    # epoch-stepping contract, so reject rather than support a path nothing
    # reaches (mirrors the L2 reject above).
    _reject_stepped(
        scalar_specs_eff.values(),
        "non-resident L3 benchmark does not support ScalarSpec benchmark_step "
        "(mark the program's weights resident)",
    )
    return _benchmark_and_report(
        compiled, specs, tensors, scalar_specs_eff, cfg, rounds, warmup, l3=True
    )


def _share_in_place(tensors: dict[str, torch.Tensor]) -> None:
    """Make every tensor shared-memory in place (required by the prepared L3 worker).

    A prepared :class:`~pypto.runtime.distributed_runner.DistributedWorker` reads
    per-call IO and resident-weight upload sources through the shared mapping the
    forked chip worker inherits at ``prepare()``, so each buffer must be CPU,
    contiguous and ``share_memory_()`` *before* the fork. Replaces any
    non-contiguous / non-shared tensor with a contiguous shared copy in the same
    dict, so the caller's later :func:`_validate` reads the device-written
    outputs back from these same buffers.
    """
    for name, t in list(tensors.items()):
        if t.is_shared() and t.is_contiguous():
            continue
        tensors[name] = t.cpu().contiguous().share_memory_()


def _strip_ssa_suffix(name: str) -> str:
    """Strip only a terminal ``__ssa_vN`` suffix from a compiled parameter name."""
    base, marker, version = name.rpartition("__ssa_v")
    return base if marker and version.isdigit() else name


def _direction_names() -> dict[Any, str]:
    """``ParamDirection`` -> :attr:`TensorSpec.direction` string, built per call
    so a patched ``ParamDirection`` is never shadowed by a cached map."""
    from pypto.ir import ParamDirection

    return {
        ParamDirection.In: "in",
        ParamDirection.Out: "out",
        ParamDirection.InOut: "inout",
    }


def _abi_error(details: list[str]) -> ValueError:
    """The one ``compiled parameter ABI mismatch (...)`` message shape."""
    return ValueError(
        "compiled parameter ABI mismatch ("
        + "; ".join(details)
        + "); recompile the artifact"
    )


def _check_param_abi(param_infos: Any, specs: list[TensorSpec | ScalarSpec]) -> None:
    """Reject any drift between the artifact's parameters and *specs*.

    Names first — a collision after SSA stripping, a name only one side has, or
    a differing order (compared, never repaired) — then kind, shape and dtype.
    A compiled ``-1`` dimension is dynamic and accepts the matching concrete
    spec dimension. Type mismatches are collected so one failure reports all of
    them rather than one per recompile. Pure: nothing is written to a spec here.
    """
    from pypto.ir.compiled_program import _to_torch_dtype

    compiled_names = [_strip_ssa_suffix(info.name) for info in param_infos]
    if len(set(compiled_names)) != len(compiled_names):
        raise ValueError("compiled parameters collide after stripping SSA suffixes")

    provided_names = [spec.name for spec in specs]
    missing_specs = sorted(set(compiled_names) - set(provided_names))
    stale_specs = sorted(set(provided_names) - set(compiled_names))
    if missing_specs or stale_specs:
        details = []
        if missing_specs:
            details.append(f"compiled parameters without specs={missing_specs}")
        if stale_specs:
            details.append(f"specs absent from compiled artifact={stale_specs}")
        raise _abi_error(details)
    if compiled_names != provided_names:
        raise _abi_error(
            [f"parameter order spec={provided_names} artifact={compiled_names}"]
        )

    directions = _direction_names()
    mismatches: list[str] = []
    for spec, info in zip(specs, param_infos, strict=True):
        name = spec.name
        artifact_shape = None if info.shape is None else tuple(info.shape)
        try:
            artifact_dtype = _to_torch_dtype(info.dtype)
        except (KeyError, TypeError, ValueError):
            artifact_dtype = None

        if isinstance(spec, ScalarSpec):
            if artifact_shape is not None:
                mismatches.append(
                    f"{name}: expected scalar, artifact is tensor shape={artifact_shape}"
                )
            if directions.get(info.direction) != "in":
                mismatches.append(
                    f"{name}: scalar direction must be In, artifact={info.direction!r}"
                )
        else:
            expected_shape = tuple(spec.shape)
            if artifact_shape is None:
                mismatches.append(f"{name}: expected tensor shape={expected_shape}, artifact is scalar")
            elif len(artifact_shape) != len(expected_shape) or any(
                artifact_dim != -1 and artifact_dim != expected_dim
                for artifact_dim, expected_dim in zip(
                    artifact_shape, expected_shape, strict=True
                )
            ):
                mismatches.append(
                    f"{name}: shape spec={expected_shape} artifact={artifact_shape}"
                )

        if artifact_dtype != spec.dtype:
            mismatches.append(
                f"{name}: dtype spec={spec.dtype} artifact={artifact_dtype}"
            )

    if mismatches:
        raise _abi_error(mismatches)


def _stamp_directions(
    specs: list[TensorSpec | ScalarSpec],
    param_infos: Any,
) -> None:
    """Copy each artifact parameter's direction onto its :class:`TensorSpec`.

    The kernel signature owns direction, so it is copied, never compared: every
    later ``spec.is_output`` / ``spec.is_input`` read resolves against what is
    written here. Runs after :func:`_check_param_abi`, so no spec is stamped
    from an artifact already known to disagree.
    """
    directions = _direction_names()
    unknown: list[str] = []
    for spec, info in zip(specs, param_infos, strict=True):
        if not isinstance(spec, TensorSpec):
            continue
        spec.direction = directions.get(info.direction)
        if spec.direction is None:
            unknown.append(f"{spec.name}: unknown artifact direction {info.direction!r}")
    if unknown:
        raise _abi_error(unknown)


def _validate_compiled_spec_abi(
    compiled: Any,
    specs: list[TensorSpec | ScalarSpec],
) -> None:
    """Validate the spec ABI of a live compiled artifact and stamp directions.

    Signature-driven JIT compilation does not consume tensor sample arguments,
    so a successful compile alone cannot prove that the caller's specs still
    match the annotated program. Compare the normalized parameter name, kind,
    shape, and dtype before either compile-only success or replay, then copy
    each parameter's direction onto its spec.

    Lightweight test doubles without metadata are ignored; real L2 and L3
    compiled programs both expose ``_get_metadata``.
    """
    metadata_getter = getattr(compiled, "_get_metadata", None)
    if not callable(metadata_getter):
        return
    metadata = metadata_getter()
    if not isinstance(metadata, tuple) or len(metadata) != 3:
        return

    param_infos, _, _ = metadata
    _check_param_abi(param_infos, specs)
    _stamp_directions(specs, param_infos)


def _l3_pure_out_names(compiled: Any) -> set[str]:
    """Names of write-only L3 parameters that need no resident initialization."""
    from pypto.ir import ParamDirection

    param_infos, _, _ = compiled._get_metadata()
    return {
        _strip_ssa_suffix(p.name)
        for p in param_infos
        if p.direction == ParamDirection.Out
    }


def _alloc_empty_stacked_tensor(rt: Any, spec: TensorSpec) -> Any:
    """Allocate one uninitialized shard per rank for a pure ``Out`` resident."""
    from pypto.runtime import StackedDeviceTensor

    shape = tuple(spec.shape)
    if len(shape) < 2 or shape[0] < 1:
        raise ValueError(
            f"TensorSpec {spec.name!r}: resident=\"stacked\" needs shape [B, *tail], got {shape}"
        )
    worker_ids = tuple(range(int(shape[0])))
    shards = []
    try:
        for wid in worker_ids:
            shards.append(
                rt.alloc_tensor(shape[1:], spec.dtype, init=None, worker_id=wid)
            )
        return StackedDeviceTensor(shards, shape, worker_ids)
    except Exception:
        for shard, wid in zip(shards, worker_ids, strict=False):
            try:
                rt.free_tensor(shard, worker_id=wid)
            except Exception:  # noqa: BLE001 - preserve the allocation/construction error
                pass
        raise


def _readback_resident_outputs(
    rt: Any,
    resident_specs: list[TensorSpec],
    resident_handles: list[tuple[str, Any, bool, int]],
    tensors: dict[str, torch.Tensor],
) -> None:
    """D2H the final device state of every resident+output spec into its host tensor.

    A resident spec marked ``is_output`` is a read-write state buffer (e.g. a KV
    cache): uploaded once, updated in place on-device, and — unlike a plain output
    — never read back per dispatch. Before validation we read each such buffer
    back **once** into ``tensors[name]`` (the shared host buffer :func:`_validate`
    reads as the device output). ``"stacked"`` uses ``copy_stacked_from`` (per-shard
    D2H); a whole-tensor buffer uses ``copy_from`` on its owning card.
    """
    out_names = {s.name for s in resident_specs if s.is_output}
    for name, handle, is_stacked, wid in resident_handles:
        if name not in out_names:
            continue
        if is_stacked:
            if not hasattr(rt, "copy_stacked_from"):
                raise ValueError(
                    f"TensorSpec {name!r}: resident=\"stacked\" read-back validation needs "
                    f"a pypto runtime exposing DistributedWorker.copy_stacked_from; "
                    f"this runtime lacks it."
                )
            rt.copy_stacked_from(handle, tensors[name])
        else:
            rt.copy_from(
                tensors[name].data_ptr(), handle.data_ptr, handle.nbytes, worker_id=wid
            )


def _run_l3_resident(
    compiled: Any,
    specs: list[TensorSpec | ScalarSpec],
    tensors: dict[str, torch.Tensor],
    scalar_specs_eff: dict[str, ScalarSpec],
    cfg: Any,
    golden_outputs: dict[str, torch.Tensor] | None,
    rtol: float,
    atol: float,
    compare_fn: dict[str, Callable],
    benchmark_enabled: bool | None = None,
) -> Any:
    """Dispatch an L3 program keeping resident weights device-resident.

    Routes through :meth:`DistributedCompiledProgram.prepare` — the only path
    that can build worker-resident :class:`~pypto.runtime.DeviceTensor` buffers.
    Each resident input / ``InOut`` spec is uploaded once via
    ``rt.alloc_tensor(init=...)`` and reused across the validation dispatch and
    every benchmark round; a pure ``Out`` resident is allocated uninitialized.
    Resident outputs are read back once before golden validation via
    :func:`_readback_resident_outputs`.

    When *benchmark_enabled* is true (default: :func:`_bench_enabled`), the
    resident weights are reused for :func:`_bench_loop_sizes` timed rounds.
    :func:`pypto.runtime.benchmark` cannot serve this — it owns its own
    ``prepare()``, and a buffer allocated on our worker is invisible to a
    second, separately forked one — so the capture is mirrored here by hand:
    raise the runtime log level to ``v9`` and wrap ``prepare()`` in the
    fd-level ``[STRACE]`` capture (the forked chip workers inherit fd 2 at fork
    time), then parse the markers into a :class:`BenchmarkStats`.

    Validation runs on the first dispatch and propagates its ``AssertionError``;
    a failure in the benchmark rounds that follow is logged, not raised. Returns
    a :class:`BenchmarkStats` or ``None``.
    """
    try:
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram
    except ImportError as e:
        raise ValueError(
            "resident specs require L3 distributed execution, but "
            "DistributedCompiledProgram could not be imported."
        ) from e
    if not isinstance(compiled, DistributedCompiledProgram):
        raise ValueError(
            "resident is only supported for L3 distributed programs "
            "(a @pl.jit.host kernel compiled with distributed_config)."
        )

    # Per-call IO + resident upload sources must be shared memory before prepare().
    _share_in_place(tensors)

    pure_out_names = _l3_pure_out_names(compiled)
    run_config = cfg
    tensor_specs = [spec for spec in specs if isinstance(spec, TensorSpec)]
    resident_specs = [spec for spec in tensor_specs if spec.is_resident]
    bench = _bench_enabled() if benchmark_enabled is None else benchmark_enabled

    def _dispatch_resident(
        dispatch_fn: "Callable[[Any, Callable[[int | None], list[Any]], list], None]",
    ) -> None:
        """Enter ``prepare()``, upload resident weights, run *dispatch_fn*, free.

        *dispatch_fn* is called as
        ``dispatch_fn(rt, ordered_args, resident_handles)`` inside the live
        ``prepare()`` context. ``ordered_args(i)`` advances stepped scalars for
        physical benchmark dispatch ``i``; ``ordered_args(None)`` returns their
        ordinary value.

        The upload / free bracket the dispatch so the resident buffers exist for
        every launch and are always released — even if *dispatch_fn* raises.
        """
        # Benchmarks model serving's steady-state dispatch: retain CommDomains
        # across rounds and let kernels clear their own signal windows. Keep the
        # ordinary validation path one-shot.
        if bench:
            prepared = compiled.prepare(
                run_config,
                persistent=True,
                reset_persistent_windows=False,
            )
        else:
            prepared = compiled.prepare()
        with prepared as rt:
            # (name, handle, is_stacked, worker_id) — is_stacked picks the matching
            # free below; worker_id is the card a whole-tensor buffer was allocated on.
            resident_handles: list[tuple[str, Any, bool, int]] = []
            try:
                for s in resident_specs:
                    if s.resident == "stacked":
                        # Leading-dim sharded: shard i of a [world_size, *tail] weight
                        # placed on card i (identity worker_ids), matching a
                        # ``for r: child(x[r], device=r)`` orchestrator.
                        if s.name in pure_out_names:
                            handle = _alloc_empty_stacked_tensor(rt, s)
                        elif not hasattr(rt, "alloc_stacked_tensor"):
                            raise ValueError(
                                f"TensorSpec {s.name!r}: resident=\"stacked\" needs a pypto runtime "
                                f"exposing DistributedWorker.alloc_stacked_tensor; this runtime lacks it."
                            )
                        else:
                            handle = rt.alloc_stacked_tensor(tensors[s.name])
                        resident_handles.append((s.name, handle, True, 0))
                    else:
                        # Whole-tensor resident on a single card: resident is the int
                        # worker id (0, 1, ...) the consuming kernel is dispatched to.
                        wid = int(s.resident)
                        init = None if s.name in pure_out_names else tensors[s.name]
                        handle = rt.alloc_tensor(
                            tuple(s.shape), s.dtype, init=init, worker_id=wid
                        )
                        resident_handles.append((s.name, handle, False, wid))
                # Resident weights replace their host buffer in the arg list;
                # everything else stays per-call IO / scalars.
                resident_args = {name: handle for name, handle, _, _ in resident_handles}

                def _dispatch_args(benchmark_dispatch_index: int | None) -> list[Any]:
                    return _ordered_args(
                        specs, tensors, scalar_specs_eff,
                        ctypes_scalars=False,
                        benchmark_dispatch_index=benchmark_dispatch_index,
                        overrides=resident_args,
                    )

                dispatch_fn(rt, _dispatch_args, resident_handles)
            finally:
                # Free every resident tensor; a failure on one must not leak the rest.
                for name, handle, is_stacked, wid in resident_handles:
                    try:
                        if is_stacked:
                            rt.free_stacked_tensor(handle)
                        else:
                            rt.free_tensor(handle, worker_id=wid)
                    except Exception as e:  # noqa: BLE001 — best-effort cleanup
                        print(f"[RUN] warning: failed to free resident tensor {name}: {e}", flush=True)

    def _validate_once(rt: Any, resident_handles: list[tuple[str, Any, bool, int]]) -> None:
        if golden_outputs is None:
            return
        # A resident spec that is also an output is a read-write state buffer
        # (e.g. a KV cache): updated in place on-device and skipping the
        # per-dispatch D2H, so its host tensor is stale. Read the final device
        # state back once into that host tensor — while the prepare() context and
        # its handles are still live — so _validate compares what the kernel
        # actually produced (one end-of-run D2H, not a per-dispatch one).
        _readback_resident_outputs(rt, resident_specs, resident_handles, tensors)
        _validate(
            tensor_specs,
            tensors,
            golden_outputs,
            rtol,
            atol,
            compare_fn,
            scalar_specs_eff,
        )

    # Non-benchmark: one validation dispatch, no capture.
    if not bench:
        def _plain_dispatch(rt: Any, ordered_args: Callable, resident_handles: list) -> None:
            rt(*ordered_args(None), config=run_config)
            _validate_once(rt, resident_handles)

        _dispatch_resident(_plain_dispatch)
        return None

    # Benchmark: mirror pypto.runtime.benchmark's L3 capture around prepare().
    import sys  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    # Private helpers: this resident path deliberately reuses benchmark()'s own
    # capture/parse rather than reimplementing the [STRACE] wire handling, but
    # cannot call benchmark() itself (see the docstring).
    from pypto.runtime.bench import (  # noqa: PLC0415
        _STRACE_LOG_LEVEL,
        _capture_fd_stderr,
        _parse_stats_from_strace,
    )
    from pypto.runtime.log_config import configure_log, current_level  # noqa: PLC0415

    rounds, warmup = _resident_loop_sizes()

    def _bench_dispatch(rt: Any, ordered_args: Callable, resident_handles: list) -> None:
        # warmup[0] doubles as the validation dispatch: run once, validate its
        # output (a correctness gate — propagates), then complete warmup + rounds.
        # The parser drops the leading `warmup` dispatches, so this launch is
        # excluded from the samples; the total stays warmup + rounds, which keeps
        # each rank's marker stream evenly segmentable into rounds.
        dispatch_index = 0

        def _run_one() -> None:
            nonlocal dispatch_index
            rt(*ordered_args(dispatch_index), config=run_config)
            dispatch_index += 1

        _run_one()
        _validate_once(rt, resident_handles)
        for _ in range(warmup - 1):
            _run_one()
        for _ in range(rounds):
            _run_one()

    prior_level = current_level()
    configure_log(_STRACE_LOG_LEVEL)
    try:
        with tempfile.TemporaryDirectory(prefix="pypto-bench-") as tmp:
            log_path = Path(tmp) / "strace.log"
            try:
                with _capture_fd_stderr(log_path):
                    _dispatch_resident(_bench_dispatch)
            except AssertionError:
                # A golden mismatch already carries its own diagnostic. The
                # capture here runs at v9, so echoing it (tens of MB on a
                # multi-card run) would bury the failing-tensor line above it.
                print(
                    "[RUN]   benchmark [STRACE] capture suppressed: validation "
                    "failed (re-run without PYPTO_BENCH for the runtime log)",
                    file=sys.stderr,
                )
                raise
            except Exception:
                # Echo the diverted setup/runtime stderr so a dispatch failure
                # keeps its diagnostics (matches benchmark()'s L3 path).
                captured = log_path.read_text(encoding="utf-8", errors="replace")
                if captured:
                    print(captured, file=sys.stderr, end="")
                raise
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
    finally:
        configure_log(prior_level)

    stats = _parse_stats_from_strace(
        log_text, rounds=rounds, warmup=warmup, distributed=True
    )
    if not stats.host_wall_us:
        print(
            "[RUN] benchmark unavailable: no [STRACE] markers captured "
            "(runtime built without SIMPLER_PROFILING)",
            flush=True,
        )
        return None
    _report_bench(stats, compiled, l3=True, resident=True)
    return stats


def _reload_from_dir(work_dir: Path, cfg: Any) -> Any:
    """Rebuild the compiled artifact from a ``runtime_dir`` without recompiling.

    ``from_dir`` reads the metadata sidecar each level persists at compile time
    (``distributed_meta.json`` for L3, ``compiled_meta.json`` for L2), so the
    replay handle is callable — and carries the param metadata
    :func:`_validate_compiled_spec_abi` needs to stamp spec directions, which a
    dispatch straight at the directory cannot supply.

    The run's ``platform`` overrides the value persisted at compile time, so
    ``--runtime-dir ... -p a2a3 -d 2,3`` replays on the requested target.
    """
    if (work_dir / "distributed_meta.json").exists():
        from pypto.ir import DistributedCompiledProgram

        return DistributedCompiledProgram.from_dir(
            work_dir,
            platform=cfg.platform,
            distributed_config=cfg.distributed_config,
        )
    from pypto.ir import CompiledProgram

    return CompiledProgram.from_dir(work_dir, platform=cfg.platform)


def _compute_golden(
    specs: list[TensorSpec | ScalarSpec],
    tensor_specs: list[TensorSpec],
    scalar_specs_eff: dict[str, ScalarSpec],
    tensors: dict[str, torch.Tensor],
    work_dir: Path,
    data_dir: Path | None,
    golden_fn: Callable | None,
    save_data: bool = True,
) -> dict[str, torch.Tensor]:
    """Produce golden output tensors for validation.

    With *data_dir* set, load from ``{data_dir}/out/``. Otherwise call
    *golden_fn* on a scratch dict (input tensors cloned from *tensors*, pure
    outputs from their own ``init_value``) and, when *save_data* is True,
    persist results into ``{work_dir}/data/out/``.

    The clone is what keeps *tensors* — the buffers the dispatch is about to be
    handed — pristine while *golden_fn* writes its scratch in place. This runs
    before the dispatch, so *tensors* still holds the generated inputs.
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
            elif spec.is_input:
                scratch[spec.name] = tensors[spec.name].clone()
            else:
                scratch[spec.name] = spec.create_tensor()
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
    scalar_specs_eff: dict[str, ScalarSpec] | None = None,
) -> None:
    """Compare device outputs against *golden_outputs*. Raises ``AssertionError``."""
    with _Stage("validate"):
        device_outputs = {spec.name: tensors[spec.name] for spec in tensor_specs if spec.is_output}
        validation_inputs = {
            spec.name: tensors[spec.name]
            for spec in tensor_specs
            if not spec.is_output
        }
        validation_inputs.update(
            {
                name: spec.value
                for name, spec in (scalar_specs_eff or {}).items()
            }
        )
        validate_golden(
            device_outputs, golden_outputs,
            rtol=rtol,
            atol=atol,
            compare_fn=compare_fn,
            inputs=validation_inputs,
        )


def _run_pipeline(
    specs: list[TensorSpec | ScalarSpec],
    *,
    compile_step: Callable[[Any, Any], Any],
    compile_label: str,
    prologue: Callable[[list[ScalarSpec], Path | None], Any] | None,
    golden_fn: Callable | None,
    golden_data: str | None,
    config: Any | None,
    rtol: float,
    atol: float,
    compare_fn: dict[str, Callable] | None,
    compile_only: bool,
    runtime_dir: str | None,
    save_data: bool,
) -> RunResult:
    """Compile / run / validate body behind :func:`run`.

    *prologue* runs entry-specific spec validation, may raise ``ValueError``,
    and returns whatever state its *compile_step* needs. *compile_step* then
    returns the ``CompiledProgram`` for a fresh compile, given the normalized
    ``RunConfig`` and that state. Everything around the two is identical for
    the ``@pl.jit`` and ``@pl.program`` entries.

    Every other argument is :func:`run`'s and is documented there.

    Returns:
        :class:`RunResult`.
    """
    compare_fn = compare_fn or {}

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

    compile_state: Any = None
    try:
        cfg = _normalize_config(config)
        _validate_unique_spec_names(specs)
        _validate_stepped_swimlane(scalar_specs, cfg)
        if prologue is not None:
            compile_state = prologue(scalar_specs, data_dir)
    except ValueError as e:
        return _fail(str(e))

    compiled: Any
    if runtime_dir is not None:
        try:
            work_dir = _setup_runtime_dir(runtime_dir, compile_label=compile_label)
        except ValueError as e:
            return _fail(str(e))
        # Compile was skipped, so there is no live compiled object; rebuild it
        # from the build dir. It is what stamps spec directions and what the
        # dispatch below calls, so this is not an L3-only concern.
        compiled = _reload_from_dir(work_dir, cfg)
    else:
        with _Stage("compile"):
            compiled = compile_step(cfg, compile_state)
            work_dir = Path(compiled.output_dir)

    # Neither a signature-driven compile (which trusts annotations over tensor
    # samples) nor a runtime-dir replay (which trusts a persisted artifact) can
    # prove the specs still describe the program. Reject stale ones before
    # allocating any input or letting compile-only report success.
    try:
        _validate_compiled_spec_abi(compiled, specs)
    except ValueError as e:
        return _fail(str(e))
    if compile_only:
        total = time.time() - start
        print(f"[RUN] PASS ({total:.2f}s)", flush=True)
        return RunResult(passed=True, execution_time=total, work_dir=work_dir)

    try:
        with _Stage("generate inputs"):
            tensors, scalar_specs_eff = _prepare_inputs(
                specs, tensor_specs, scalar_specs, data_dir, work_dir, save_data,
            )
    except ValueError as e:
        return _fail(str(e))

    golden_outputs: dict[str, torch.Tensor] | None = None
    if golden_fn is not None or golden_data is not None:
        golden_outputs = _compute_golden(
            specs, tensor_specs, scalar_specs_eff, tensors,
            work_dir, data_dir, golden_fn, save_data,
        )

    benchmark_enabled = _bench_enabled()
    stepped = sorted(n for n, s in scalar_specs_eff.items() if s.has_benchmark_step)
    if benchmark_enabled and runtime_dir is not None and stepped:
        print(
            "[RUN]   benchmark skipped: runtime_dir replay cannot prove stepped "
            f"scalar(s) {stepped} reach the kernel; recompile to benchmark",
            flush=True,
        )
        benchmark_enabled = False

    def _pass(bench: Any) -> RunResult:
        total = time.time() - start
        skip_note = (
            ", validation skipped: no golden_fn or golden_data"
            if golden_outputs is None else ""
        )
        print(f"[RUN] PASS ({total:.2f}s{skip_note})", flush=True)
        return RunResult(
            passed=True, execution_time=total, work_dir=work_dir, bench=bench,
        )

    # Resident-weight path: keep resident specs device-resident across the
    # validation dispatch and any benchmark rounds via the L3 prepare() worker
    # (validation + benchmark are handled inside; return early).
    if any(s.is_resident for s in tensor_specs):
        with _Stage("runtime"):
            try:
                bench = _run_l3_resident(
                    compiled, specs, tensors, scalar_specs_eff,
                    cfg, golden_outputs, rtol, atol, compare_fn,
                    benchmark_enabled=benchmark_enabled,
                )
            except (AssertionError, ValueError) as e:
                return _fail(str(e))
        return _pass(bench)

    with _Stage("runtime"):
        _dispatch(compiled, specs, tensors, scalar_specs_eff, cfg)

    # Validate the dedicated correctness dispatch before benchmark launches
    # mutate output or inout tensors in place.
    if golden_outputs is not None:
        try:
            _validate(
                tensor_specs, tensors, golden_outputs,
                rtol, atol, compare_fn, scalar_specs_eff,
            )
        except AssertionError as e:
            return _fail(str(e))

    # Benchmark (L2 via _run_benchmark, non-resident L3 via _run_benchmark_l3).
    # Runs only after the correctness dispatch has been validated, for a fresh
    # compile and a runtime-dir replay alike. Entirely env-gated via
    # PYPTO_BENCH=1 (daily CI).
    bench = None
    if benchmark_enabled:
        rounds, warmup = _bench_loop_sizes()
        run_bench = _run_benchmark_l3 if _is_l3(compiled) else _run_benchmark
        bench = run_bench(
            compiled, specs, tensors, scalar_specs_eff, cfg, rounds, warmup,
        )
    return _pass(bench)


def _program_entry(
    fn: Any, specs: list[TensorSpec | ScalarSpec]
) -> tuple[Callable[[Any, Any], Any], None, str]:
    """``(compile_step, prologue, label)`` for a ``@pl.program`` class / ``ir.Program``."""
    del specs  # the program path derives everything from the compiled artifact

    def _compile(cfg: Any, _state: Any) -> Any:
        from pypto import ir

        # compile_kwargs() carries the platform, which an L3 program bakes into
        # compiled.platform at compile time: without it, incore kernels for a
        # `-p a2a3` run would be built for the backend's default sim platform.
        return ir.compile(fn, **cfg.compile_kwargs())

    return _compile, None, "Program compile"


def _jit_entry(
    fn: Any, specs: list[TensorSpec | ScalarSpec]
) -> tuple[
    Callable[[Any, Any], Any],
    Callable[[list[ScalarSpec], Path | None], Any],
    str,
]:
    """``(compile_step, prologue, label)`` for a ``@pl.jit`` callable.

    The prologue resolves the scalar values the specialization key needs and
    hands them to the compile step.
    """

    def _prologue(
        scalar_specs: list[ScalarSpec], data_dir: Path | None
    ) -> dict[str, ScalarSpec]:
        compile_scalars = _effective_scalar_specs(scalar_specs, data_dir)
        # A stepped scalar must survive specialization as a runtime parameter;
        # a literal-specialized one would bake dispatch 0's value into the code.
        _reject_stepped(
            scalar_specs,
            "ScalarSpec benchmark_step requires compile_runtime=True",
            only_specialized=True,
        )
        return compile_scalars

    def _compile(cfg: Any, compile_scalars: dict[str, ScalarSpec]) -> Any:
        scalar_specs = [s for s in specs if isinstance(s, ScalarSpec)]
        if any(spec.compile_runtime for spec in scalar_specs):
            import pypto.language as pl

            # pl.RUNTIME is accepted only by annotation-driven signature
            # compilation: omit every tensor sample and provide all scalar
            # parameters by name. Unmarked scalars retain their literal
            # specialization semantics.
            scalar_compile_args = {
                spec.name: (
                    pl.RUNTIME
                    if spec.compile_runtime
                    else compile_scalars[spec.name].to_python()
                )
                for spec in scalar_specs
            }
            return fn.compile(config=cfg, **scalar_compile_args)
        # Dummy args carry shape/dtype and scalar values into the specialization
        # key; real tensors of the same shape hit the same JIT cache entry at
        # dispatch.
        dummy_args = [
            compile_scalars[spec.name].to_python()
            if isinstance(spec, ScalarSpec)
            else torch.empty(spec.shape, dtype=spec.dtype)
            for spec in specs
        ]
        return fn.compile(*dummy_args, config=cfg)

    return _compile, _prologue, "JIT compile"


def run(
    fn: Any,
    specs: list[TensorSpec | ScalarSpec],
    golden_fn: Callable | None = None,
    golden_data: str | None = None,
    config: Any | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    compare_fn: dict[str, Callable] | None = None,
    compile_only: bool = False,
    runtime_dir: str | None = None,
    save_data: bool = False,
) -> RunResult:
    """Compile *fn*, run it on device, and validate against golden.

    Accepts either kernel form. A ``@pl.jit`` callable exposes ``compile`` and
    is specialized through ``JITFunction.compile``; a ``@pl.program`` class or
    an ``ir.Program`` goes straight to :func:`pypto.ir.compile`. The compile
    step is the only difference; both read the same *config*.

    Args:
        fn: ``@pl.jit`` callable, ``@pl.program`` class, or ``ir.Program``.
        specs: :class:`TensorSpec` / :class:`ScalarSpec` list in *fn*'s
            parameter order. A mismatched order is rejected, never reordered.
        golden_fn: ``golden_fn(values)`` that fills outputs in-place; *values*
            maps spec name to tensor clone or Python scalar. Ignored when
            *golden_data* is set; if neither is given, validation is skipped.
        golden_data: Directory with ``in/{name}.pt`` and ``out/{name}.pt``;
            loads inputs and expected outputs (read-only). Takes precedence
            over *golden_fn*.
        config: Every setting for both phases, in one dict of
            :class:`pypto.runtime.RunConfig` keyword arguments — ``platform``,
            ``device_id``, ``enable_chip_swimlane``, ``dump_passes``,
            ``distributed_config``, the ``ring_*`` overrides, ... A ready
            ``RunConfig`` is accepted too. PyPTO reads the compile half off
            it via ``compile_kwargs()`` and the dispatch half via
            ``run_options()`` / ``dfx_options()``, so nothing is restated per
            phase and no key has to be filed under a phase by the caller. An unknown key
            is ``RunConfig``'s own ``TypeError`` naming it. The one key that is
            not a ``RunConfig`` field is the harness's ``log_level`` — the
            PyPTO runtime log threshold (``debug``, ``v0``..``v9``, ``info``,
            ``warn``, ``error``, ``null``), see
            :func:`pypto.runtime.log_config.configure_log`.
        rtol, atol: Golden comparison tolerances.
        compare_fn: Per-output-name overrides for ``torch.allclose``; see
            :func:`golden.validation.validate_golden`.
        compile_only: Stop after code generation; skip execute and validate.
        runtime_dir: Pre-compiled ``build_output/`` directory to reuse. Skips
            compile and invalidates cached ``.so``/``.bin`` so cpp edits
            rebuild; the compile-side config is ignored and *compile_only* is
            rejected. ``PYPTO_BENCH`` benchmarks the replayed build.
        save_data: When True, persist generated inputs to
            ``{work_dir}/data/in/`` and golden outputs to
            ``{work_dir}/data/out/`` for later replay via *golden_data*.
            Defaults to False, skipping the on-disk ``.pt`` snapshot;
            validation still runs against the in-memory golden. Enable it
            when you need to replay the exact inputs/outputs later.

    Returns:
        :class:`RunResult`.
    """
    # A JITFunction exposes compile(); a @pl.program class evaluates to an
    # ir.Program, which does not.
    entry = _jit_entry if callable(getattr(fn, "compile", None)) else _program_entry
    compile_step, prologue, compile_label = entry(fn, specs)
    with _log_level_scope(config):
        return _run_pipeline(
            specs,
            compile_step=compile_step,
            compile_label=compile_label,
            prologue=prologue,
            golden_fn=golden_fn,
            golden_data=golden_data,
            config=config,
            rtol=rtol,
            atol=atol,
            compare_fn=compare_fn,
            compile_only=compile_only,
            runtime_dir=runtime_dir,
            save_data=save_data,
        )
