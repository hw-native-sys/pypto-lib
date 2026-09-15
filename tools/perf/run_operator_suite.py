# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run and report the ten fixed MTP/DSpark workloads in CI's environment."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import stat
import statistics
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).parent / "suites/dsv4_operators.json"
CASE_IDS = {
    f"{model}-{op}" for model in ("mtp", "dspark")
    for op in ("attention-csa", "attention-hca", "attention-swa", "moe-ep8", "lm-head")
}
SAMPLING = {"seed": 1807, "rounds": 100, "warmup": 5, "raw": True}


class ContractError(ValueError):
    """Evidence is missing or incompatible with the selected workload."""


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def devices(value):
    try:
        parsed = [int(item) for item in value.split(",")] if isinstance(value, str) else value
        if not isinstance(parsed, list) or not parsed:
            raise ValueError
        if any(type(item) is not int or item < 0 or item % 2 for item in parsed):
            raise ValueError
        if len(set(parsed)) != len(parsed) or parsed != sorted(parsed):
            raise ValueError
    except (TypeError, ValueError):
        raise ContractError("devices must be distinct, ascending, non-negative even host IDs") from None
    return parsed


def load_profile(path=None):
    profile = read_json(path) if path else read_json(MANIFEST)["device_profile"]
    if profile.get("schema_version") != 1 or profile.get("device_id_domain") != "host":
        raise ContractError("only schema 1 host device IDs are supported; remapping is not supported")
    epoch = profile.get("device_epoch")
    if not isinstance(epoch, str) or not epoch.strip():
        raise ContractError("a device epoch is required")
    for count in (1, 4, 8):
        selected = devices(profile.get("sequences", {}).get(str(count)))
        if len(selected) != count:
            raise ContractError(f"the {count}-device sequence has the wrong size")
    if not set(profile["sequences"]["1"] + profile["sequences"]["4"]) <= set(profile["sequences"]["8"]):
        raise ContractError("single-card and TP4 devices must be subsets of the EP8 allocation")
    return profile


def validate_allocation(profile, allocated, selected, environ=None):
    environ = os.environ if environ is None else environ
    allocation = devices(allocated)
    if allocation != profile["sequences"]["8"]:
        raise ContractError("the suite requires the exact eight-device allocation from its profile")
    if devices(environ.get("TASK_DEVICE", "")) != allocation:
        raise ContractError("TASK_DEVICE differs from the requested allocation")
    if environ.get("TASKQUEUE_INSIDE") != "1":
        raise ContractError("run inside a task-submit allocation")
    for key in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES"):
        if environ.get(key, "").strip():
            raise ContractError(f"{key} remaps host IDs; this runner requires an unmasked host")
    if selected != profile["sequences"].get(str(len(selected))) or not set(selected) <= set(allocation):
        raise ContractError("case devices differ from the fixed profile or escape the allocation")


def load_manifest(path=MANIFEST):
    manifest = read_json(path)
    cases = manifest.get("cases", [])
    if manifest.get("schema_version") != 1 or manifest.get("sampling") != SAMPLING:
        raise ContractError("unsupported manifest or sampling contract")
    if len(cases) != 10 or {case.get("case_id") for case in cases} != CASE_IDS:
        raise ContractError("the suite must contain exactly the ten MTP/DSpark cases")
    if manifest.get("platform") != "a2a3":
        raise ContractError("official performance requires a2a3")
    for case in cases:
        entry = (ROOT / case["entrypoint"]).resolve()
        if not entry.is_relative_to(ROOT / "models") or not entry.is_file():
            raise ContractError("entry point must be an existing model in this checkout")
        if case["device_count"] not in (1, 4, 8) or not case.get("required_outputs"):
            raise ContractError("invalid case devices or output contract")
    return manifest


def case_arguments(case, selected):
    return ["-p", "a2a3", "-d", ",".join(map(str, selected)),
            *case["arguments"], "--enable-chip-swimlane", "0"]


def positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ContractError(f"{label} must be a finite positive number")
    return float(value)


def check_specs(case, specs, *, require_outputs=False):
    by_name = {spec["name"]: spec for spec in specs}
    if len(by_name) != len(specs):
        raise ContractError("duplicate tensor specifications")
    for name, expected in case["required_specs"].items():
        actual = by_name.get(name, {})
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ContractError(f"{name} shape/dtype differs from the workload contract")
    if require_outputs:
        outputs = {spec["name"] for spec in specs if spec.get("direction") in ("out", "inout")}
        if not set(case["required_outputs"]) <= outputs:
            raise ContractError("required results are not validated outputs")


def validate_result(payload, case, *, process_rc=0):
    if process_rc != 0 or payload.get("passed") is not True:
        raise ContractError(f"operator did not pass correctness (exit {process_rc}): {payload.get('error')}")
    if payload.get("case_id") != case["case_id"] or payload.get("sampling") != SAMPLING:
        raise ContractError("case identity or sampling differs from the request")
    if payload.get("run_config") != case["run_config"]:
        raise ContractError("runtime settings differ from the case contract")
    if not payload.get("fixture_sha256") or not payload.get("golden_sha256"):
        raise ContractError("missing actual input or reference fingerprint")
    check_specs(case, payload["specs"], require_outputs=True)
    bench = payload["benchmark"]
    if (bench["rounds"], bench["warmup"]) != (100, 5):
        raise ContractError("expected 100 measured rounds after 5 warmup rounds")
    if any(bench.get(key) is not False for key in (
        "fallback_flattened", "unstable_dispatch_slots", "all_zero_device",
    )):
        raise ContractError("benchmark has unavailable, flattened or unstable timing")
    count = case["device_count"]
    if count > 1 and bench.get("distributed_grid") is not True:
        raise ContractError("distributed round boundaries are missing")
    rows = bench["samples"]
    if len(rows) != 100:
        raise ContractError("incomplete benchmark rounds")
    pids = set(rows[0])
    if len(pids) != count:
        raise ContractError("benchmark rank count differs from allocated devices")
    samples = {pid: [] for pid in pids}
    slots = {}
    identities = {}
    seen = set()
    for row in rows:
        if set(row) != pids:
            raise ContractError("missing or unexpected rank in a measured round")
        for pid, dispatches in row.items():
            if not dispatches or len(dispatches) != len(rows[0][pid]):
                raise ContractError("dispatch count changed between rounds")
            total = 0.0
            for slot, dispatch in enumerate(dispatches):
                key = (pid, slot)
                identity = dispatch["task"]
                if not identity or identities.setdefault(key, identity) != identity:
                    raise ContractError("dispatch slot changed callable between rounds")
                inv = dispatch["inv"]
                if type(inv) is not int or inv < 0 or (pid, inv) in seen:
                    raise ContractError("duplicate or invalid rank invocation")
                seen.add((pid, inv))
                value = positive(dispatch["effective_us"], "effective_us")
                slots.setdefault(key, []).append(value)
                total += value
            samples[pid].append(total)
    per_rank = [
        {"trace_pid": pid, "median_us": statistics.median(values), "samples_us": values,
         "dispatches": [{"slot": slot, "task": identities[(pid, slot)],
                         "samples_us": values, "median_us": statistics.median(values)}
                        for (rank, slot), values in sorted(slots.items()) if rank == pid]}
        for pid, values in sorted(samples.items())
    ]
    medians = [rank["median_us"] for rank in per_rank]
    return {"metric_us": min(medians), "max_rank_median_us": max(medians),
            "rank_spread_us": max(medians) - min(medians), "ranks": per_rank}


def comparison(current, baseline):
    """A positive percentage means slower; unavailable evidence is never zero."""
    if not baseline or current.get("status") != "pass" or baseline.get("status") != "pass":
        return {"delta_pct": None, "reason": "no passing baseline/current result"}
    keys = ["case_contract", "device_identity", "fixture_sha256", "golden_sha256"]
    # CI selects the toolchain revision. System changes start a new series;
    # compiler/runtime changes remain visible as complete-stack deltas.
    for key in ("python", "torch", "numpy", "cann_sha256", "driver_sha256", "bundle"):
        if (not current.get("toolchain", {}).get(key) or
                current["toolchain"][key] != baseline.get("toolchain", {}).get(key)):
            return {"delta_pct": None, "reason": f"incomparable system component: {key}"}
    for key in keys:
        if not current.get(key) or current[key] != baseline.get(key):
            return {"delta_pct": None, "reason": f"incomparable {key}"}
    now = positive(current["metric_us"], "current metric")
    before = positive(baseline["metric_us"], "baseline metric")
    changes = [key for key in sorted(set(current.get("toolchain", {})) | set(baseline.get("toolchain", {})))
               if current.get("toolchain", {}).get(key) != baseline.get("toolchain", {}).get(key)]
    if current.get("source_sha") != baseline.get("source_sha"):
        changes.insert(0, "pypto-lib")
    return {"scope": "ci_stack", "changed_components": changes,
            "delta_pct": (now / before - 1) * 100, "baseline_run_id": baseline.get("run_id"),
            "baseline_date": baseline.get("logical_date")}


def command(argv, cwd=None):
    return subprocess.check_output(argv, cwd=cwd, text=True, stderr=subprocess.PIPE).strip()


def capture_toolchain(driver_info=Path("/usr/local/Ascend/driver/version.info")):
    """Record versions from setup-ci-job; installation validation belongs to CI."""
    pypto_root = Path(os.environ["PYPTO_SRC"])
    cann_info = sorted(Path(os.environ["ASCEND_HOME_PATH"]).glob("*/ascend_toolkit_install.info"))
    if not cann_info or not driver_info.is_file():
        raise ContractError("CANN or driver version metadata is missing")
    return {
        "pypto": command(["git", "rev-parse", "HEAD"], pypto_root),
        "runtime": command(["git", "rev-parse", "HEAD"], pypto_root / "runtime"),
        "pto_isa": os.environ["PTO_ISA_COMMIT"],
        "ptoas_version": command([str(Path(os.environ["PTOAS_ROOT"]) / "bin/ptoas"), "--version"]),
        "bundle": Path(os.environ["PYPTO_TOOLCHAIN"]).name,
        "python": platform.python_version(),
        "torch": metadata.version("torch"), "numpy": metadata.version("numpy"),
        "cann_sha256": hashlib.sha256(cann_info[0].read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(driver_info.read_bytes()).hexdigest(),
    }


def source_identity():
    return command(["git", "rev-parse", "HEAD"], ROOT)


def capture_host(profile, output_dir):
    inventory = command(["npu-smi", "info"])
    (Path(output_dir) / "npu-smi-info.txt").write_text(inventory + "\n")
    mapping = []
    for device in profile["sequences"]["8"]:
        node = Path(f"/dev/davinci{device}").stat()
        if not stat.S_ISCHR(node.st_mode):
            raise ContractError("allocated device is not a host character device")
        mapping.append({"host_device_id": device, "major": os.major(node.st_rdev),
                        "minor": os.minor(node.st_rdev)})
    return {"hostname": socket.gethostname(), "device_epoch": profile["device_epoch"],
            "device_id_domain": "host", "devices": mapping, "kernel": platform.release()}


DEVICE_FAULT = re.compile(
    r"50703[45]|SCHEDULER_TIMEOUT|AICORE_EXCEPTION|device hang|"
    r"HDC[^\n]*(?:disconnect|error)|SDMA[^\n]*(?:error|fail)", re.I,
)


def execution_plan(manifest, profile, python):
    allocation = profile["sequences"]["8"]
    return {
        "suite_id": manifest["suite_id"], "sampling": SAMPLING,
        "allocation": allocation,
        "queue_prefix": ["task-submit", "--device", ",".join(map(str, allocation))],
        "cases": [{"case_id": case["case_id"],
                   "devices": profile["sequences"][str(case["device_count"])],
                   "model_command": [python, case["entrypoint"],
                                     *case_arguments(case, profile["sequences"][str(case["device_count"])])]}
                  for case in manifest["cases"]],
    }


def run_process(command, log, cwd, environ, timeout):
    """Keep model subprocesses isolated; a timeout stops the entire allocation."""
    with log.open("w") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=environ, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return 124, True


def render_report(suite):
    lines = [f"Operator performance: {suite['logical_date']} (CI)",
             f"Run: `{suite['run_id']}`; status: **{suite['status']}**.", "",
             "Metric: minimum rank median of summed operator dispatch Effective times (us).",
             "Communication and signal cleanup are included; this is not model step latency.", "",
             "| Case | Devices | Status | us | Previous change | Week change |",
             "| --- | --- | --- | --- | --- | --- |"]
    for case in suite["cases"]:
        def delta(key):
            value = case.get(key, {}).get("delta_pct")
            return "N/A" if value is None else f"{value:+.2f}%"
        metric = f"{case['metric_us']:.2f}" if case.get("status") == "pass" else "N/A"
        selected = ",".join(map(str, case["devices"]))
        lines.append(f"| {case['case_id']} | {selected} | {case['status']} | {metric} | "
                     f"{delta('previous')} | {delta('week')} |")
    lines.extend(["", "Changes describe the complete CI software stack, not an isolated kernel edit.",
                  "Positive changes mean slower. N/A means missing or incompatible evidence.", ""])
    for case in suite["cases"]:
        if case.get("error"):
            error = str(case["error"]).replace("\n", " ")
            lines.append(f"- {case['case_id']}: {error}")
        for key in ("previous", "week"):
            info = case.get(key)
            if info:
                components = ", ".join(info.get("changed_components", [])) or "none"
                detail = info.get("reason") or (
                    f"baseline run {info.get('baseline_run_id')}; changed components: {components}")
                lines.append(f"- {case['case_id']} {key}: {detail}")
    if suite.get("error"):
        lines.extend(["", str(suite["error"])])
    return "\n".join(lines) + "\n"


def load_baseline(path, logical_date, *, weekly=False):
    if path is None:
        return {}
    baseline = read_json(path)
    previous = datetime.fromisoformat(baseline["logical_date"]).date()
    today = datetime.fromisoformat(logical_date).date()
    age = (today - previous).days
    if age <= 0 or (weekly and age != 7):
        raise ContractError("baseline must be earlier; the weekly baseline must be exactly seven days earlier")
    records = baseline["cases"]
    if len({case["case_id"] for case in records}) != len(records):
        raise ContractError("baseline contains duplicate cases")
    return {case["case_id"]: case for case in records}


def execute(args, manifest, profile):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    suite = {"schema_version": 1, "suite_id": manifest["suite_id"], "run_id": run_id,
             "logical_date": args.date, "variant": "ci", "status": "running",
             "coverage_complete": False, "allocation": profile["sequences"]["8"],
             "sampling": SAMPLING, "cases": []}
    for case in manifest["cases"]:
        suite["cases"].append({"case_id": case["case_id"], "run_id": run_id, "status": "not_run",
                               "logical_date": args.date, "variant": "ci",
                               "devices": profile["sequences"][str(case["device_count"])],
                               "case_contract": digest([case, manifest["metric"], SAMPLING,
                                                        manifest["contract_id"]])})

    def checkpoint():
        write_json(output / "suite-result.json", suite)
        (output / "report.md").write_text(render_report(suite))

    checkpoint()
    try:
        for case in suite["cases"]:
            validate_allocation(profile, suite["allocation"], case["devices"])
        if profile.get("hostname") and profile["hostname"] != socket.gethostname():
            raise ContractError("this runner is not the configured performance host")
        source_sha = source_identity()
        toolchain = capture_toolchain()
        host = capture_host(profile, output)
        suite.update(source_sha=source_sha, toolchain=toolchain, host=host,
                     task_id=os.environ.get("TASKQUEUE_TASK_ID"))
        prior = load_baseline(args.baseline, args.date)
        week = load_baseline(args.week_baseline, args.date, weekly=True)
    except (OSError, ValueError, KeyError, ImportError, subprocess.SubprocessError) as error:
        suite.update(status="preflight_failed", error=str(error))
        checkpoint()
        return 1

    env = os.environ.copy()
    env.update(PYTHONHASHSEED="1807", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               PYPTO_BENCH="1", PYPTO_BENCH_RAW="1", PYPTO_BENCH_ROUNDS="100", PYPTO_BENCH_WARMUP="5",
               PYPTO_LOG_LEVEL="error", PYPTO_RUNTIME_LOG="error", SIMPLER_DEVICE_STRACE_ENABLE="1")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    stop = False
    build_dirs = []
    for case, record in zip(manifest["cases"], suite["cases"], strict=True):
        if stop:
            record["error"] = "allocation stopped after device fault or timeout"
            continue
        case_dir = output / case["case_id"]
        case_dir.mkdir()
        raw_path = case_dir / "raw-result.json"
        log = case_dir / "run.log"
        command = [sys.executable, str(ROOT / "tools/perf/deterministic_run.py"),
                   "--case", case["case_id"], "--devices", ",".join(map(str, record["devices"])),
                   "--output", str(raw_path)]
        record.update(status="running", source_sha=source_sha, toolchain=toolchain,
                      device_identity={**host, "selected": record["devices"]},
                      command=command, log=str(log.relative_to(output)))
        checkpoint()
        try:
            rc, timed_out = run_process(command, log, case_dir, env, args.case_timeout)
            record["process_rc"] = rc
            if timed_out:
                stop = True
                raise ContractError("case timeout; the allocation is stopped")
            raw = read_json(raw_path)
            record.update(validate_result(raw, case, process_rc=rc))
            record.update(status="pass", fixture_sha256=raw["fixture_sha256"],
                          golden_sha256=raw["golden_sha256"])
            record["previous"] = comparison(record, prior.get(case["case_id"]))
            record["week"] = comparison(record, week.get(case["case_id"]))
            if raw.get("work_dir"):
                build = Path(raw["work_dir"])
                if not build.is_absolute():
                    build = case_dir / build
                build_dirs.append((case_dir, build.resolve()))
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
            record.update(status="fail", error=str(error))
            if log.exists():
                with log.open(errors="replace") as stream:
                    stop = stop or any(DEVICE_FAULT.search(line) for line in stream)
        checkpoint()
    suite["coverage_complete"] = all(case["status"] == "pass" for case in suite["cases"])
    suite["allocation_stopped"] = stop
    suite["status"] = "pass" if suite["coverage_complete"] else "incomplete"
    checkpoint()
    for case_dir, build in build_dirs:
        # Delete only successful builds inside this run's private case directory.
        if build.is_relative_to(case_dir / "build_output") and build != case_dir / "build_output":
            shutil.rmtree(build)
    return 0 if suite["coverage_complete"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-profile", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--date", default=datetime.now(timezone(timedelta(hours=8))).date().isoformat())
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--week-baseline", type=Path)
    parser.add_argument("--case-timeout", type=int, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    manifest = load_manifest()
    profile = load_profile(args.device_profile)
    datetime.fromisoformat(args.date)
    if args.case_timeout <= 0:
        parser.error("--case-timeout must be positive")
    if args.dry_run:
        print(json.dumps(execution_plan(manifest, profile, sys.executable), indent=2))
        return 0
    if not args.output_dir:
        parser.error("--output-dir is required for execution")
    return execute(args, manifest, profile)


if __name__ == "__main__":
    raise SystemExit(main())
