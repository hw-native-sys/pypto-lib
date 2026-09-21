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
case captures one model's benchmark inside its task-submit allocation.
"""

from __future__ import annotations

import argparse
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
QUEUE_TIMEOUT = 3600  # task-submit --timeout: queue wait plus execution
# Ring sizes pinned for every case, independent of the host's PTO2_RING_* exports.
RING_CONFIG = {"ring_task_window": 16384, "ring_dep_pool": 16384, "ring_heap": 1 << 30}


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_manifest():
    manifest = read_json(MANIFEST)
    ids = [case["case_id"] for case in manifest["cases"]]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case_id in the manifest")
    return manifest


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


def make_capture(original_run, output, selected):
    """Wrap golden.run: pin the ring settings and save the benchmark."""
    def capture(**kwargs):
        if output.exists():
            raise ValueError("a case must call golden.run exactly once")
        cfg = kwargs["config"]
        distributed = cfg.get("distributed_config")
        actual = list(distributed.device_ids) if distributed else [cfg["device_id"]]
        if actual != selected:
            raise ValueError(f"model runs on devices {actual}, allocation is {selected}")
        kwargs["config"] = {**cfg, **RING_CONFIG}
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
    import golden

    golden.run = make_capture(golden.run, args.output, selected)
    entry = ROOT / case["entrypoint"]
    sys.path.insert(0, str(entry.parent))
    sys.argv = [str(entry), "-p", "a2a3", "-d", ",".join(map(str, selected)), *case["arguments"],
                "--enable-chip-swimlane", "0"]
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


def run_ci(output):
    manifest = load_manifest()
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
        with (case_dir / "run.log").open("w") as log:
            rc = subprocess.run(submit_command(case, case_dir, sampling), env=env,
                                stdout=log, stderr=subprocess.STDOUT, check=False).returncode
        try:
            if rc != 0:
                raise ValueError(f"task-submit exited {rc}")
            results[case_id] = effective_us(read_json(case_dir / "raw-result.json"), case, sampling)
            print(f"[perf] {case_id}: {results[case_id]:.2f} us", flush=True)
        except (OSError, ValueError, KeyError, TypeError) as error:
            results[case_id] = "FAIL"
            print(f"[perf] {case_id}: FAIL ({error}); see {case_id}/run.log in the artifact", flush=True)
        # Publish logs and raw results only, never build trees.
        (published / case_id).mkdir()
        for name in ("run.log", "raw-result.json"):
            if (case_dir / name).is_file():
                shutil.copyfile(case_dir / name, published / case_id / name)
        (published / "report.md").write_text(render_report(manifest, results))
    return 0 if all(isinstance(value, float) for value in results.values()) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    ci = commands.add_parser("ci", help="Run every case through task-submit and write report.md")
    ci.add_argument("--output-dir", required=True, type=Path)
    capture = commands.add_parser("case", help="Capture one model inside its task-submit allocation")
    capture.add_argument("--case", required=True)
    capture.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.mode == "ci":
        return run_ci(args.output_dir.resolve())
    return run_case(args)


if __name__ == "__main__":
    raise SystemExit(main())
