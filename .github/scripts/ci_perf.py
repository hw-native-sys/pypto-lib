# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Daily operator performance: run each fixed workload and report its effective time.

Commands: ci submits every case through task-submit and writes report.md;
prepare freezes a GLM case on CPU; case captures its allocated device replay.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import runpy
import shlex
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
# Model capture runs from a private case directory, not the checkout.
sys.path.insert(0, str(ROOT))
MANIFEST = Path(__file__).with_name("perf_cases.json")
CASE_TIMEOUT = 1800  # task-submit --max-time: one model's compile, golden and benchmark
PREPARE_TIMEOUT = 3600
QUEUE_TIMEOUT = 3600  # task-submit --timeout: queue wait plus execution
# Ring sizes pinned for every case, independent of the host's PTO2_RING_* exports.
RING_CONFIG = {"ring_task_window": 16384, "ring_dep_pool": 16384, "ring_heap": 1 << 30}


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def ring_config(case):
    """Resolve only explicitly supported ring overrides for one fixed workload."""
    overrides = case.get("runtime_config", {})
    if not isinstance(overrides, dict):
        raise ValueError("case runtime_config must be an object")
    unknown = set(overrides) - set(RING_CONFIG)
    if unknown:
        raise ValueError(f"unsupported performance runtime_config keys: {sorted(unknown)}")
    for name, value in overrides.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"runtime_config {name} must be a positive integer")
    return {**RING_CONFIG, **overrides}


def load_manifest(platform=None):
    manifest = read_json(MANIFEST)
    ids = [case["case_id"] for case in manifest["cases"]]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case_id in the manifest")
    if platform is not None:
        manifest["cases"] = [case for case in manifest["cases"]
                             if case.get("platform", manifest["platform"]) == platform]
        if not manifest["cases"]:
            raise ValueError(f"no performance cases for platform {platform}")
    return manifest


@contextmanager
def cached_native_only():
    """Fail on a cache miss before a device replay can start CPU compilation."""
    from pypto.runtime import device_runner
    from pypto.runtime.debug import pto_rebuild

    def forbidden(*_args, **_kwargs):
        raise RuntimeError("prepared performance case requires native compilation; prepare it again on CPU")

    # Registration wrappers also load cached sidecars. Guard the common native
    # compiler/linker subprocess boundary, not those cache-aware wrappers.
    compiler = device_runner.KernelCompiler
    original = compiler._run
    original_ptoas = pto_rebuild._run_ptoas
    try:
        compiler._run = forbidden
        pto_rebuild._run_ptoas = forbidden
        yield
    finally:
        compiler._run = original
        pto_rebuild._run_ptoas = original_ptoas


def model_argv(manifest, case, selected):
    entry = ROOT / case["entrypoint"]
    platform = case.get("platform", manifest["platform"])
    argv = [str(entry), "-p", platform, "-d", ",".join(map(str, selected)), *case["arguments"]]
    if platform == "a2a3" and case.get("suite") != "glm5.3-flash":
        argv += ["--enable-chip-swimlane", "0"]
    return argv


def prepare_case(args):
    """Compile, assemble and freeze the original golden case without a device."""
    # cloudpickle is provided by the Simpler runtime, also used for its callables.
    import cloudpickle
    import golden
    from golden import ScalarSpec, TensorSpec, runner
    from pypto.runtime.device_runner import _compile_and_assemble

    manifest = load_manifest()
    case = next(case for case in manifest["cases"] if case["case_id"] == args.case)
    if case.get("suite") != "glm5.3-flash" or case["device_count"] != 1:
        raise ValueError("CPU preparation currently supports the single-device GLM suite")
    prepared = args.output.resolve()
    prepared.parent.mkdir(parents=True, exist_ok=True)
    if prepared.exists():
        raise ValueError(f"prepared case already exists: {prepared}")
    platform = case.get("platform", manifest["platform"])
    original_run = golden.run
    calls = 0
    metadata = None
    os.environ["PYPTO_BENCH"] = "0"

    def freeze(**kwargs):
        nonlocal calls, metadata
        calls += 1
        if calls != 1:
            raise ValueError("a case must call golden.run exactly once")
        if kwargs.get("golden_fn") is None:
            raise ValueError("GLM performance preparation requires an independent golden")
        cfg = {**kwargs["config"], **ring_config(case), "device_id": 0, "execution_mode": False,
               "enable_chip_swimlane": 0}
        if cfg.get("distributed_config") is not None:
            raise ValueError("GLM local-shard performance cases must not allocate distributed workers")
        compile_kwargs = {**kwargs, "config": cfg, "compile_only": True, "save_data": False,
                          "runtime_dir": None, "golden_data": None}
        result = original_run(**compile_kwargs)
        if not result.passed:
            raise ValueError(result.error or "CPU compilation failed")
        build = Path(result.work_dir).resolve()
        # Populate the same registration/sidecar caches that ordinary replay
        # reads. Exporting a separate prebuilt bundle does not fill every
        # version's normal loader cache.
        _compile_and_assemble(build, platform)
        with cached_native_only():
            runner._setup_runtime_dir(str(build), compile_label="prepared case", platform=platform)
            _compile_and_assemble(build, platform)

        specs = kwargs["specs"]
        tensor_specs = [spec for spec in specs if isinstance(spec, TensorSpec)]
        scalar_specs = [spec for spec in specs if isinstance(spec, ScalarSpec)]
        if any(spec.has_benchmark_step for spec in scalar_specs):
            raise ValueError("prepared GLM cases cannot use stepped scalar replay")
        frozen = prepared.parent / "frozen"
        tensors, scalars = runner._prepare_inputs(specs, tensor_specs, scalar_specs, None, frozen, True)
        outputs = runner._compute_golden(
            specs, tensor_specs, scalars, tensors, frozen, None, kwargs["golden_fn"], True,
        )
        if not outputs:
            raise ValueError("GLM performance preparation produced no golden outputs")
        clean_specs = [copy.copy(spec) for spec in specs]
        for spec in clean_specs:
            if isinstance(spec, TensorSpec):
                spec.init_value = None
        payload = {key: value for key, value in kwargs.items()
                   if key not in ("fn", "golden_fn", "compile_only", "runtime_dir", "golden_data", "save_data")}
        payload.update(specs=clean_specs, config=cfg)
        payload_path = prepared.parent / "prepared-payload.pkl"
        with payload_path.open("wb") as stream:
            cloudpickle.dump(payload, stream)
        metadata = {"case_id": case["case_id"], "function": kwargs["fn"].__name__,
                    "runtime_dir": str(build)}
        return result

    sys.argv = model_argv(manifest, case, [0])
    sys.path.insert(0, str((ROOT / case["entrypoint"]).parent))
    golden.run = freeze
    try:
        runpy.run_path(str(ROOT / case["entrypoint"]), run_name="__main__")
    finally:
        golden.run = original_run
    if calls != 1 or metadata is None:
        raise ValueError("model produced no prepared case")
    write_json(prepared, metadata)
    return 0


def replay_prepared(case, selected, output, prepared):
    """Replay the CPU-prepared inputs and original precision callbacks."""
    import cloudpickle
    import golden

    metadata = read_json(prepared)
    if metadata["case_id"] != case["case_id"] or len(selected) != 1:
        raise ValueError("prepared GLM case or device count differs from the requested workload")
    data = prepared.parent / "frozen/data"
    if not any((data / "out").glob("*.pt")):
        raise ValueError("prepared case has no correctness reference")
    module = importlib.import_module(case["entrypoint"].removesuffix(".py").replace("/", "."))
    fn = getattr(module, metadata["function"])
    with (prepared.parent / "prepared-payload.pkl").open("rb") as stream:
        payload = cloudpickle.load(stream)
    payload["config"].update(device_id=selected[0], execution_mode=False, enable_chip_swimlane=0)
    with cached_native_only():
        result = make_capture(golden.run, output, selected, case)(
            fn=fn, golden_fn=None, golden_data=str(data), runtime_dir=metadata["runtime_dir"], **payload,
        )
    if not result.passed:
        raise ValueError(f"correctness failed: {result.error}")


# ---------------------------------------------------------------------------
# case: runs inside the task-submit allocation, one fresh process per model.


def benchmark_payload(stats):
    """Serialize RunResult.bench as rounds of {rank: [dispatch effective_us]}."""
    if stats is None:
        raise ValueError("benchmark unavailable or skipped")
    grid = stats.rounds_dispatches
    if grid:
        rounds = [{str(pid): [item.effective_us for item in items] for pid, items in row.items()}
                  for row in grid]
    else:
        rounds = [{str(item.pid): [item.effective_us]} for item in stats.invocations]
    return {"rounds": stats.rounds, "warmup": stats.warmup,
            "fallback_flattened": stats.fallback_flattened,
            "unstable_dispatch_slots": getattr(stats, "unstable_dispatch_slots", False),
            "all_zero_device": stats.all_zero_device, "samples": rounds}


def make_capture(original_run, output, selected, case):
    """Wrap golden.run: pin the ring settings and save the benchmark."""
    fixed_ring = ring_config(case)
    def capture(**kwargs):
        if output.exists():
            raise ValueError("a case must call golden.run exactly once")
        cfg = kwargs["config"]
        distributed = cfg.get("distributed_config")
        actual = list(distributed.device_ids) if distributed else [cfg["device_id"]]
        if actual != selected:
            raise ValueError(f"model runs on devices {actual}, allocation is {selected}")
        kwargs["config"] = {**cfg, **fixed_ring}
        result = original_run(**kwargs)
        payload = {"passed": result.passed, "error": result.error}
        if result.passed:
            payload["benchmark"] = benchmark_payload(result.bench)
        write_json(output, payload)
        return result

    return capture


def run_case(args):
    manifest = load_manifest()
    case = next(case for case in manifest["cases"] if case["case_id"] == args.case)
    selected = [int(item) for item in os.environ["TASK_DEVICE"].split(",")]
    if len(selected) != case["device_count"]:
        raise ValueError(f"allocated {len(selected)} devices, case needs {case['device_count']}")
    sys.argv = model_argv(manifest, case, selected)
    if case.get("suite") == "glm5.3-flash":
        if args.prepared is None:
            raise ValueError("GLM performance must be prepared on CPU before device allocation")
        replay_prepared(case, selected, args.output, args.prepared.resolve())
        return 0
    import golden

    golden.run = make_capture(golden.run, args.output, selected, case)
    entry = ROOT / case["entrypoint"]
    sys.path.insert(0, str(entry.parent))
    runpy.run_path(str(entry), run_name="__main__")
    if not args.output.is_file():
        raise ValueError("model produced no benchmark result")
    return 0


# ---------------------------------------------------------------------------
# ci: submits every case and renders the report.


def effective_us(payload, case, sampling):
    """Per rank and round, sum all dispatches.

    Single card reports the median; multi-card reports the fastest rank's mean.
    """
    if payload.get("passed") is not True:
        raise ValueError(f"correctness failed: {payload.get('error')}")
    bench = payload["benchmark"]
    if (bench["rounds"], bench["warmup"]) != (sampling["rounds"], sampling["warmup"]):
        raise ValueError("benchmark rounds/warmup differ from the manifest")
    if bench["fallback_flattened"] or bench["unstable_dispatch_slots"] or bench["all_zero_device"]:
        raise ValueError("benchmark timing is flattened, unstable or all zero")
    rows = bench["samples"]
    if len(rows) != sampling["rounds"] or len(rows[0]) != case["device_count"]:
        raise ValueError("benchmark rounds or ranks are incomplete")
    totals = {}
    for row in rows:
        if set(row) != set(rows[0]):
            raise ValueError("a rank is missing from a measured round")
        for pid, dispatches in row.items():
            totals.setdefault(pid, []).append(sum(dispatches))
    if case["device_count"] == 1:
        return statistics.median(next(iter(totals.values())))
    return min(statistics.mean(values) for values in totals.values())


def submit_command(case, case_dir, sampling):
    count = case["device_count"]
    env = {"PYTHONPATH": str(ROOT), "PYPTO_BENCH": "1", "PYPTO_BENCH_RAW": "1",
           "PYPTO_BENCH_ROUNDS": str(sampling["rounds"]), "PYPTO_BENCH_WARMUP": str(sampling["warmup"]),
           "PYPTO_LOG_LEVEL": "error", "PYPTO_RUNTIME_LOG": "error", "SIMPLER_DEVICE_STRACE_ENABLE": "1"}
    # task-submit sources ~/.bashrc; forward the job's resolved toolchain.
    env.update({key: os.environ[key] for key in ("PYPTO_SRC", "PTO_ISA_COMMIT", "PTOAS_ROOT")
                if key in os.environ})
    capture = [sys.executable, str(ROOT / ".github/scripts/ci_perf.py"), "case",
               "--case", case["case_id"], "--output", str(case_dir / "raw-result.json")]
    if case.get("suite") == "glm5.3-flash":
        capture += ["--prepared", str(case_dir / "prepared.json")]
        env.update(OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8")
    payload = (f"cd {shlex.quote(str(ROOT))} && source activate.sh && cd {shlex.quote(str(case_dir))} && "
               + shlex.join(["env", *(f"{key}={value}" for key, value in env.items()), *capture]))
    command = ["task-submit", "--device", "auto"]
    if count > 1:
        command += ["--device-num", str(count)]
    if count >= 8:
        command += ["--ignore-whitelist"]
    return command + ["--timeout", str(QUEUE_TIMEOUT), "--max-time", str(CASE_TIMEOUT), "--run", payload]


def render_report(manifest, results):
    lines = ["| Case | Effective (us) |", "| --- | --- |"]
    for case in manifest["cases"]:
        value = results.get(case["case_id"], "-")
        lines.append(f"| {case['case_id']} | {value:.2f} |" if isinstance(value, float)
                     else f"| {case['case_id']} | {value} |")
    return "\n".join(lines) + "\n"


def run_ci(output, platform="a2a3"):
    manifest = load_manifest(platform)
    sampling = manifest["sampling"]
    work, published = output / "work", output / "published"
    work.mkdir(parents=True)
    published.mkdir(parents=True)
    env = {key: value for key, value in os.environ.items() if key not in ("GH_TOKEN", "GITHUB_TOKEN")}
    results = {}
    (published / "report.md").write_text(render_report(manifest, results))
    for case in manifest["cases"]:
        case_id = case["case_id"]
        case_dir = work / case_id
        case_dir.mkdir()
        print(f"[perf] {case_id}: running on {case['device_count']} device(s)", flush=True)
        print(f"::group::{case_id}", flush=True)
        try:
            if case.get("suite") == "glm5.3-flash":
                command = [sys.executable, str(ROOT / ".github/scripts/ci_perf.py"), "prepare",
                           "--case", case_id, "--output", str(case_dir / "prepared.json")]
                with (case_dir / "prepare.log").open("w", encoding="utf-8") as log:
                    prepared = subprocess.run(command, cwd=case_dir,
                                              env={**env, "PYPTO_BENCH": "0", "OMP_NUM_THREADS": "8",
                                                   "OPENBLAS_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"},
                                              stdout=log, stderr=subprocess.STDOUT, timeout=PREPARE_TIMEOUT)
                if prepared.returncode:
                    raise ValueError(f"CPU preparation exited {prepared.returncode}; see prepare.log")
            with (case_dir / "run.log").open("w", encoding="utf-8") as log, \
                 subprocess.Popen(submit_command(case, case_dir, sampling), env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1) as proc:
                for line in proc.stdout:
                    sys.stdout.write(line)
                    log.write(line)
                sys.stdout.flush()
            rc = proc.returncode
            if rc != 0:
                raise ValueError(f"task-submit exited {rc}")
            results[case_id] = effective_us(read_json(case_dir / "raw-result.json"), case, sampling)
            print(f"[perf] {case_id}: {results[case_id]:.2f} us", flush=True)
        except (OSError, UnicodeError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
            results[case_id] = "FAIL"
            print(f"[perf] {case_id}: FAIL ({error}); see {case_id}/run.log in the artifact", flush=True)
        finally:
            print("::endgroup::", flush=True)
        # Publish logs and raw results only, never build trees.
        (published / case_id).mkdir()
        for name in ("prepare.log", "prepared.json", "run.log", "raw-result.json"):
            if (case_dir / name).is_file():
                shutil.copyfile(case_dir / name, published / case_id / name)
        (published / "report.md").write_text(render_report(manifest, results))
    return 0 if all(isinstance(value, float) for value in results.values()) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    ci = commands.add_parser("ci", help="Run every case through task-submit and write report.md")
    ci.add_argument("--platform", choices=("a2a3", "a5"), default="a2a3")
    ci.add_argument("--output-dir", required=True, type=Path)
    prepare = commands.add_parser("prepare", help="Prepare and freeze one GLM case on CPU")
    prepare.add_argument("--case", required=True)
    prepare.add_argument("--output", required=True, type=Path)
    capture = commands.add_parser("case", help="Capture one model inside its task-submit allocation")
    capture.add_argument("--case", required=True)
    capture.add_argument("--output", required=True, type=Path)
    capture.add_argument("--prepared", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "ci":
        return run_ci(args.output_dir.resolve(), args.platform)
    if args.mode == "prepare":
        return prepare_case(args)
    return run_case(args)


if __name__ == "__main__":
    raise SystemExit(main())
