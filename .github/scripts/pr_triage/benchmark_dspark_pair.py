# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compare one DSpark hint on identical cards, frozen inputs and one process.

Run inside a task-submit allocation. This artifact is not a model entry point.
Both checkouts must have the same helpers; only the two reviewed DSpark entry
files may differ. The first entry saves its passing golden once; the second
replays it. CPU mode is a harness smoke test and never reports performance.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import runpy
import statistics
import subprocess
import sys


MODEL = Path("models/deepseek_v4_flash_dspark")
ALLOWED_DIFFERENCES = {"decode_swa.py", "decode_hca.py"}
RING_CONFIG = {"ring_task_window": 16384, "ring_dep_pool": 16384, "ring_heap": 1 << 30}


class ResumeBaselineCaptured(Exception):
    """Stop a restored baseline CLI after collecting its original run kwargs."""


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def revision(root):
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def working_tree(root):
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True,
    )
    diff = subprocess.check_output(["git", "-C", str(root), "diff", "--binary", "HEAD", "--", str(MODEL)])
    return {"tracked_dirty": bool(status.strip()), "model_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "model_diff": diff.decode("utf-8")}


def sdk_pins():
    """Read compiler/runtime identities without opening a device context."""
    from simpler_setup import pto_isa

    source = os.environ.get("PYPTO_SRC")
    pins = {"pypto": None, "simpler": None, "pto_isa": pto_isa.read_pto_isa_pin()}
    if source and (Path(source) / ".git").exists():
        pins["pypto"] = revision(Path(source))
        pins["simpler"] = revision(Path(source) / "runtime")
    ptoas = Path(os.environ.get("PTOAS_ROOT", "")) / "bin/ptoas"
    text = subprocess.check_output([str(ptoas), "--version"], text=True, stderr=subprocess.STDOUT)
    match = re.search(r"\b(\d+\.\d+(?:\.\d+)?)\b", text)
    if match is None:
        raise ValueError("unable to identify the active PTOAS compiler")
    pins["ptoas"] = match.group(1).removesuffix(".0") if match.group(1).count(".") == 2 else match.group(1)
    return pins


def summarize(stats, rounds, warmup, devices):
    if stats is None:
        raise ValueError("no hardware benchmark was captured")
    if (stats.rounds, stats.warmup) != (rounds, warmup):
        raise ValueError("benchmark sampling differs from the requested contract")
    if stats.fallback_flattened or stats.all_zero_device or getattr(stats, "unstable_dispatch_slots", False):
        raise ValueError("flattened, zero or unstable benchmark dispatches")
    grid = stats.rounds_dispatches
    if not grid:
        invocations = stats.invocations
        if len(devices) != 1 or len(invocations) != rounds:
            raise ValueError("a single-card benchmark requires one invocation per round")
        if len({item.pid for item in invocations}) != 1:
            raise ValueError("a single-card benchmark contains multiple rank IDs")
        if len({(item.pid, item.inv) for item in invocations}) != rounds:
            raise ValueError("a single-card benchmark repeats an invocation")
        grid = [{item.pid: [item]} for item in invocations]
    if not grid or len(grid) != rounds:
        raise ValueError("rank/round samples are unavailable")
    keys = set(grid[0])
    if len(keys) != len(devices):
        raise ValueError("benchmark rank count differs from the allocation")
    samples = {str(key): [] for key in keys}
    dispatches = []
    for row in grid:
        if set(row) != keys:
            raise ValueError("a benchmark round is missing a rank")
        record = {}
        for key, values in row.items():
            values = [value.effective_us for value in values]
            if not values or any(value <= 0 for value in values):
                raise ValueError("invalid effective_us sample")
            samples[str(key)].append(sum(values))
            record[str(key)] = values
        dispatches.append(record)
    medians = {key: statistics.median(values) for key, values in samples.items()}
    return {"rounds": rounds, "warmup": warmup, "per_rank_median_us": medians,
            "fastest_rank_median_us": min(medians.values()), "dispatch_samples_us": dispatches}


def restore_baseline(args, devices, baseline, helper_hashes):
    """Recover a proven baseline without executing it or recomputing its golden."""
    root = args.resume_root.resolve()
    previous_json = root / "private/measurement/result.json"
    previous_log = root / "published/run.log"
    previous = json.loads(previous_json.read_text())
    log = previous_log.read_text()
    for name, value in (("case", args.case), ("tp", args.tp), ("batch", args.batch),
                        ("start_pos", args.start_pos), ("devices", devices),
                        ("platform", "a2a3sim" if args.sim else "a2a3")):
        if previous.get(name) != value:
            raise ValueError(f"resume contract changed: {name}")
    if previous["baseline_revision"] != revision(baseline):
        raise ValueError("resume baseline revision changed")
    if previous["helper_hashes"] != helper_hashes:
        raise ValueError("resume baseline helper sources changed")
    baseline_log = log.split("[PAIR] candidate source=", 1)[0]
    match = re.search(r"\[PAIR\] baseline source=(.*?) sha256=([0-9a-f]{64})", baseline_log)
    if match is None or "[RUN] PASS (" not in baseline_log:
        raise ValueError("the previous baseline has no completed passing validation")
    source = baseline / MODEL / f"decode_{args.case}.py"
    if Path(match.group(1)).resolve() != source or match.group(2) != sha256(source):
        raise ValueError("resume baseline source path or SHA256 changed")
    snapshot = Path(previous["golden_snapshot"])
    if not (snapshot / "in").is_dir() or not (snapshot / "out").is_dir():
        raise ValueError("resume golden snapshot is unavailable")
    for name, digest in previous["snapshot_sha256"].items():
        if sha256(snapshot / name) != digest:
            raise ValueError(f"resume snapshot bytes changed: {name}")
    entry = dict(previous.get("baseline", {}))
    entry.update(passed=True, source_sha256=sha256(source), work_dir=str(snapshot.parent))
    provenance = {"root": str(root), "previous_result_sha256": sha256(previous_json),
                  "previous_run_log_sha256": sha256(previous_log), "baseline_execution_skipped": True,
                  "snapshot_rehashed": True, "previous_benchmark_script_sha256": previous.get("benchmark_script_sha256"),
                  "generation_seed": previous.get("generation_seed", "unknown")}
    if not args.sim and "benchmark" not in entry:
        if args.case != "swa" or args.tp != 1:
            raise ValueError("only the known single-card L2 serialization failure can recover printed samples")
        raw = re.findall(r"\[RUN\]\s+rank (\d+) raw n=(\d+) eff_us=(\[[^\n]+\])", baseline_log)
        if len(raw) != 1 or int(raw[0][1]) != args.rounds:
            raise ValueError("the previous baseline raw sample count is invalid")
        samples = json.loads(raw[0][2])
        if len(samples) != args.rounds or any(value <= 0 for value in samples):
            raise ValueError("the previous baseline samples are incomplete or nonpositive")
        contract = re.search(r"raw samples: ranks=1 rounds=(\d+) warmup=(\d+)", baseline_log)
        if contract is None or tuple(map(int, contract.groups())) != (args.rounds, args.warmup):
            raise ValueError("the previous baseline rounds/warmup changed")
        median = statistics.median(samples)
        entry["benchmark"] = {"rounds": args.rounds, "warmup": args.warmup,
                              "per_rank_median_us": {raw[0][0]: median}, "fastest_rank_median_us": median,
                              "dispatch_samples_us": [{raw[0][0]: [value]} for value in samples]}
        provenance["baseline_samples_source"] = "original passing run's PYPTO_BENCH_RAW stdout"
        provenance["baseline_samples_print_precision_us"] = 0.1
    else:
        provenance["baseline_samples_source"] = "original structured passing baseline result"
    return entry, snapshot, previous["snapshot_sha256"], provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--case", choices=("swa", "hca"), required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--start-pos", type=int, default=131072)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline-control", action="store_true")
    parser.add_argument("--resume-root", type=Path)
    parser.add_argument("--expected-sdk", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--cpu-runtime-helper", type=Path)
    args = parser.parse_args()
    if args.case == "swa" and args.tp != 1:
        parser.error("the reviewed SWA hint is in the TP1 path")
    if args.case == "hca" and args.tp not in (2, 4):
        parser.error("the reviewed HCA hint is in the distributed path")
    if min(args.batch, args.rounds, args.tp) < 1 or args.warmup < 0:
        parser.error("batch/rounds/tp must be positive and warmup nonnegative")
    if not 0 <= args.seed < (1 << 32):
        parser.error("seed must fit the unsigned 32-bit NumPy seed range")
    baseline, candidate = args.baseline_root.resolve(), args.candidate_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        parser.error("use a new output directory; existing evidence is preserved")
    if args.sim:
        devices = list(range(args.tp))
        os.environ.pop("PYPTO_BENCH", None)
        os.environ["PTO_CPU_SIM_ARCH"] = "a2a3"
        if args.cpu_runtime_helper:
            builder = runpy.run_path(str(args.cpu_runtime_helper.resolve()))["configure"]()
            original_init = builder.__init__

            def cpu_only_init(self, platform="a2a3"):
                if platform != "a2a3sim":
                    raise RuntimeError(f"CPU smoke test rejects platform {platform!r}")
                return original_init(self, platform)

            builder.__init__ = cpu_only_init
    else:
        devices = [int(item) for item in os.environ["TASK_DEVICE"].split(",")]
        if len(devices) != args.tp or len(set(devices)) != len(devices):
            parser.error("TASK_DEVICE must contain exactly TP distinct allocated devices")
        os.environ["PYPTO_BENCH"] = "1"
        os.environ["PYPTO_BENCH_ROUNDS"] = str(args.rounds)
        os.environ["PYPTO_BENCH_WARMUP"] = str(args.warmup)
        os.environ["PYPTO_BENCH_RAW"] = "1"
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    pins = sdk_pins()
    if args.resume_root and not args.sim and args.expected_sdk is None:
        parser.error("hardware resume requires the original --expected-sdk manifest")
    if args.expected_sdk:
        expected = json.loads(args.expected_sdk.read_text())
        if pins != expected:
            raise ValueError(f"SDK changed since the retained baseline: expected {expected}, got {pins}")
    # Sharing imported helper objects is safe only when their source is equal.
    helper_hashes = {}
    for old in (baseline / MODEL).glob("*.py"):
        new = candidate / MODEL / old.name
        if old.name not in ALLOWED_DIFFERENCES:
            if not new.is_file() or sha256(old) != sha256(new):
                raise ValueError(f"helper differs between the checkouts: {old.name}")
            helper_hashes[old.name] = sha256(old)
    for path in ("golden/runner.py", "golden/spec.py", "golden/validation.py"):
        if sha256(baseline / path) != sha256(candidate / path):
            raise ValueError(f"golden harness differs between the checkouts: {path}")
    restored = restore_baseline(args, devices, baseline, helper_hashes) if args.resume_root else None
    sys.path[:0] = [str(baseline / MODEL), str(baseline)]
    import golden

    original_run = golden.run
    generation_seed = restored[3]["generation_seed"] if restored is not None else args.seed
    if restored is None:
        import numpy as np
        import torch

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    results = {"case": args.case, "tp": args.tp, "batch": args.batch,
               "start_pos": args.start_pos, "devices": devices, "platform": "a2a3sim" if args.sim else "a2a3",
               "process_id": os.getpid(), "ring_config": RING_CONFIG, "helper_hashes": helper_hashes,
               "metric": "minimum over per-rank medians of per-round dispatch effective_us sums",
               "baseline_revision": revision(baseline), "candidate_revision": revision(candidate),
               "baseline_working_tree": working_tree(baseline), "candidate_working_tree": working_tree(candidate),
               "benchmark_script_sha256": sha256(Path(__file__).resolve()), "sdk": pins}
    results["generation_seed"] = generation_seed
    snapshot = None
    baseline_kwargs = None
    try:
        if restored is not None:
            entry, snapshot, snapshot_hashes, provenance = restored
            results.update(baseline=entry, golden_snapshot=str(snapshot), snapshot_sha256=snapshot_hashes,
                           resume=provenance)
        for label, repo in (("baseline", baseline), ("candidate", candidate)):
            source = repo / MODEL / f"decode_{args.case}.py"
            os.environ["PYPTO_PROG_BUILD_DIR"] = str(output / f"{label}_build")
            os.environ["PYPTO_CACHE_DIR"] = str(output / f"{label}_cache")
            captured = []

            def capture(**kwargs):
                nonlocal baseline_kwargs
                if captured:
                    raise ValueError("each entry must call golden.run exactly once")
                python_entry = getattr(kwargs["fn"], "_func", None)
                entry_code = getattr(python_entry, "__code__", None)
                if entry_code is not None:
                    if Path(entry_code.co_filename).resolve() != source:
                        raise ValueError("the compiled entry is from a different checkout")
                    print(f"[PAIR] {label} compiled_entry={entry_code.co_filename}", flush=True)
                config = dict(kwargs["config"])
                distributed = config.get("distributed_config")
                selected = list(distributed.device_ids) if distributed else [config["device_id"]]
                if selected != devices:
                    raise ValueError(f"entry selected {selected}, allocation is {devices}")
                kwargs["config"] = {**config, **RING_CONFIG}
                kwargs["save_data"] = label == "baseline"
                kwargs["golden_data"] = snapshot if label == "candidate" else None
                if label == "baseline":
                    baseline_kwargs = dict(kwargs)
                    if restored is not None:
                        raise ResumeBaselineCaptured
                result = original_run(**kwargs)
                captured.append(result)
                return result

            golden.run = capture
            sys.path[:0] = [str(repo / MODEL), str(repo)]
            sys.argv = [str(source), "-p", "a2a3", "--tp", str(args.tp),
                        "-d", ",".join(map(str, devices)), "--start-pos",
                        ",".join([str(args.start_pos)] * args.batch), "--enable-chip-swimlane", "0"]
            if args.sim:
                sys.argv.append("--sim")
            print(f"[PAIR] {label} source={source} sha256={sha256(source)}", flush=True)
            print(f"[PAIR] argv={sys.argv}", flush=True)
            os.chdir(repo)
            try:
                runpy.run_path(str(source), run_name="__main__")
            except ResumeBaselineCaptured:
                print(f"[PAIR] baseline restored from {args.resume_root}; no execution or compilation", flush=True)
                continue
            if len(captured) != 1 or not captured[0].passed:
                raise RuntimeError(f"{label} did not pass its original comparators")
            result = captured[0]
            if label == "baseline":
                snapshot = Path(result.work_dir) / "data"
                if not (snapshot / "in").is_dir() or not (snapshot / "out").is_dir():
                    raise ValueError("passing baseline did not save its golden snapshot")
                results["golden_snapshot"] = str(snapshot)
                results["snapshot_sha256"] = {str(path.relative_to(snapshot)): sha256(path)
                                               for path in sorted(snapshot.rglob("*.pt"))}
            entry = {"passed": result.passed, "work_dir": str(result.work_dir), "source_sha256": sha256(source)}
            if not args.sim:
                entry["benchmark"] = summarize(result.bench, args.rounds, args.warmup, devices)
            results[label] = entry
            print(f"[PAIR] {label} PASS; build={result.work_dir}", flush=True)
        if args.baseline_control:
            if baseline_kwargs is None or snapshot is None:
                raise ValueError("the baseline control requires a passing captured baseline")
            baseline_work_dir = Path(results["baseline"]["work_dir"])
            os.environ["PYPTO_PROG_BUILD_DIR"] = str(output / "baseline_build")
            control_kwargs = {**baseline_kwargs, "runtime_dir": baseline_work_dir,
                              "golden_data": snapshot, "save_data": False}
            print(f"[PAIR] baseline_control reuses runtime_dir={baseline_work_dir}", flush=True)
            control = original_run(**control_kwargs)
            if not control.passed:
                raise RuntimeError(f"baseline control failed its original comparators: {control.error}")
            if Path(control.work_dir) != baseline_work_dir:
                raise ValueError("the baseline control did not reuse the original runtime directory")
            results["baseline_control"] = {"passed": True, "work_dir": str(control.work_dir),
                                           "runtime_dir": str(baseline_work_dir)}
            if not args.sim:
                benchmark = summarize(control.bench, args.rounds, args.warmup, devices)
                results["baseline_control"]["benchmark"] = benchmark
                before = results["baseline"]["benchmark"]["fastest_rank_median_us"]
                results["baseline_control_drift_percent"] = (
                    benchmark["fastest_rank_median_us"] / before - 1.0
                ) * 100.0
            print(f"[PAIR] baseline_control PASS; reused build={control.work_dir}", flush=True)
        if not args.sim:
            before = results["baseline"]["benchmark"]["fastest_rank_median_us"]
            after = results["candidate"]["benchmark"]["fastest_rank_median_us"]
            results["change_percent"] = (after / before - 1.0) * 100.0
        results["passed"] = True
    finally:
        golden.run = original_run
        (output / "result.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in results.items()
                      if key not in ("helper_hashes", "snapshot_sha256")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
