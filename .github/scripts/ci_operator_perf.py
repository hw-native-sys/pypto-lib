# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Daily operator performance: CI history, grouped allocation and model capture.

Commands: ci runs the workflow; suite submits the fixed workloads; case captures
one model within an allocation; finalize publishes checkpoints after cleanup.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
from http.client import IncompleteRead
from importlib import metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import runpy
import shlex
import shutil
import signal
import socket
from ssl import SSLCertVerificationError, SSLEOFError
import stat
import statistics
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[2]
# Model capture must also work when launched from a private build directory.
sys.path.insert(0, str(ROOT))
MANIFEST = Path(__file__).with_name("operator_perf_cases.json")
CASE_IDS = {
    f"{model}-{op}-{context}" for model in ("mtp", "dspark")
    for context in ("8k", "128k")
    for op in ("attention-csa", "attention-hca", "attention-swa", "moe-ep8", "lm-head")
}
SAMPLING = {"seed": 1807, "rounds": 100, "warmup": 5, "raw": True}


class ContractError(ValueError):
    """Evidence is missing or incompatible with the selected workload."""


class TimeoutCleanupError(ContractError):
    """A timed-out process group could not be confirmed terminated."""


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
    for count in (1, 8):
        selected = devices(profile.get("sequences", {}).get(str(count)))
        if len(selected) != count:
            raise ContractError(f"the {count}-device sequence has the wrong size")
    if not set(profile["sequences"]["1"]) <= set(profile["sequences"]["8"]):
        raise ContractError("single-card devices must be a subset of the EP8 allocation")
    return profile


def validate_allocation(profile, allocated, selected, environ=None):
    environ = os.environ if environ is None else environ
    allocation = devices(allocated)
    if allocation != selected or allocation != profile["sequences"].get(str(len(selected))):
        raise ContractError("each case requires exactly its one- or eight-device profile allocation")
    if devices(environ.get("TASK_DEVICE", "")) != allocation:
        raise ContractError("TASK_DEVICE differs from the requested allocation")
    if environ.get("TASKQUEUE_INSIDE") != "1":
        raise ContractError("run inside a task-submit allocation")
    for key in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES"):
        if environ.get(key, "").strip():
            raise ContractError(f"{key} remaps host IDs; this runner requires an unmasked host")


def load_manifest(path=MANIFEST):
    manifest = read_json(path)
    cases = manifest.get("cases", [])
    if manifest.get("schema_version") != 1 or manifest.get("sampling") != SAMPLING:
        raise ContractError("unsupported manifest or sampling contract")
    if len(cases) != len(CASE_IDS) or {case.get("case_id") for case in cases} != CASE_IDS:
        raise ContractError("the suite must contain exactly the twenty MTP/DSpark cases")
    if manifest.get("platform") != "a2a3":
        raise ContractError("official performance requires a2a3")
    for case in cases:
        entry = (ROOT / case["entrypoint"]).resolve()
        if not entry.is_relative_to(ROOT / "models") or not entry.is_file():
            raise ContractError("entry point must be an existing model in this checkout")
        if case["device_count"] not in (1, 8) or not case.get("required_outputs"):
            raise ContractError("invalid case devices or output contract")
    measurement_groups(manifest)
    return manifest


def measurement_groups(manifest):
    """Only explicitly shared, context-independent workloads may reuse evidence."""
    cases = {case["case_id"]: case for case in manifest["cases"]}
    groups = {key: [] for key, case in cases.items() if not case.get("measurement_case")}
    for case in cases.values():
        source = case.get("measurement_case", case["case_id"])
        if source not in groups:
            raise ContractError("shared measurement must refer directly to an executed case")
        if source != case["case_id"]:
            if not any(f"-{op}-" in case["case_id"] for op in ("moe-ep8", "lm-head")):
                raise ContractError("only context-independent MoE and LM-head may share measurements")
            def contract(value):
                value = copy.deepcopy(value)
                value.pop("case_id")
                value.pop("measurement_case", None)
                value["workload"].pop("scenario_context")
                return value
            if contract(case) != contract(cases[source]):
                raise ContractError("shared measurement has a different execution contract")
        groups[source].append(case)
    return [(cases[source], rows) for source, rows in groups.items()]


def allocation_groups(manifest):
    """Run both models under one lease per device count, single-card first."""
    measurements = measurement_groups(manifest)
    return [(count, [(case, rows) for case, rows in measurements if case["device_count"] == count])
            for count in (1, 8)]


def validate_host(profile):
    if not profile.get("hostname"):
        raise ContractError("OPERATOR_PERF_HOST is required to identify this run's performance host")
    if profile["hostname"] != socket.gethostname():
        raise ContractError("this runner is not the configured performance host")


def case_arguments(case, selected):
    return ["-p", "a2a3", "-d", ",".join(map(str, selected)),
            *case["arguments"], "--enable-chip-swimlane", "0"]


def capture_command(case, selected, output, python):
    return [python, str(ROOT / ".github/scripts/ci_operator_perf.py"), "case",
            "--case", case["case_id"], "--devices", ",".join(map(str, selected)),
            "--output", str(output)]


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
    return {
        "suite_id": manifest["suite_id"], "sampling": SAMPLING,
        "report_rows": len(manifest["cases"]),
        "allocations": [{"group_id": f"group-{count}", "devices": profile["sequences"][str(count)],
                         "case_ids": [case["case_id"] for case, _ in measurements],
                         "queue_prefix": ["task-submit", "--device",
                                          ",".join(map(str, profile["sequences"][str(count)]))]}
                        for count, measurements in allocation_groups(manifest)],
        "cases": [{"case_id": case["case_id"], "report_cases": [row["case_id"] for row in rows],
                   "devices": profile["sequences"][str(case["device_count"])],
                   "allocation_group": f"group-{case['device_count']}",
                   "entrypoint": case["entrypoint"], "arguments": case["arguments"],
                   "capture_command": capture_command(
                       case, profile["sequences"][str(case["device_count"])],
                       Path(case["case_id"]) / "raw-result.json", python)}
                  for case, rows in measurement_groups(manifest)],
    }


def group_running(group):
    # A killed grandchild can remain a zombie until its new parent reaps it.
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group and fields[0] != "Z":
                return True
        except (FileNotFoundError, ProcessLookupError):
            continue
    return False


def run_process(command, log, cwd, environ, timeout):
    """Bound one model process and clean its entire process group on timeout."""
    with log.open("w") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=environ, stdout=stream,
                                   # Isolate the process group but retain the queue's
                                   # session so daemon cancellation can sweep children.
                                   stderr=subprocess.STDOUT, preexec_fn=os.setpgrp)
        try:
            return process.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            # The leader can exit before its children. Always kill the group.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
            deadline = time.monotonic() + 5
            while group_running(process.pid):
                if time.monotonic() >= deadline:
                    raise TimeoutCleanupError("timed-out process group could not be cleaned")
                time.sleep(0.1)
            return 124, True


def healthy_idle_inventory(text, selected):
    """Match unmasked physical IDs in npu-smi's two-line device table."""
    seen, health = {}, None
    for line in text.splitlines():
        fields = [field.strip() for field in line.split("|")]
        if len(fields) < 5:
            continue
        if re.fullmatch(r"\d+\s+Ascend\S+", fields[1]):
            health = fields[2]
        elif health is not None and re.fullmatch(r"\d+\s+\d+", fields[1]):
            device = int(fields[1].split()[1])
            busy = fields[3].split()[0] if fields[3].split() else "unknown"
            if device in seen:
                return False
            seen[device] = health == "OK" and busy == "0"
            health = None
    return all(seen.get(device) is True for device in selected)


def check_recovery(selected, output):
    """Require two healthy, idle observations while still holding the allocation."""
    try:
        for attempt in range(2):
            result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True,
                                    timeout=15, check=True)
            (output / f"recovery-{attempt}.txt").write_text(result.stdout)
            if not healthy_idle_inventory(result.stdout, selected):
                return False
            if attempt == 0:
                time.sleep(2)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def run_worker(args, manifest, profile):
    """Supervise exactly one measurement inside its queue allocation."""
    case = next(case for case, _ in measurement_groups(manifest) if case["case_id"] == args.worker_case)
    selected = profile["sequences"][str(case["device_count"])]
    output = args.output_dir.resolve()
    evidence = {"case_id": case["case_id"], "status": "running", "safe_to_continue": False,
                "golden_replayed": False, "task_id": os.environ.get("TASKQUEUE_TASK_ID")}
    write_json(output / "measurement.json", evidence)
    started = time.monotonic()
    log = output / "run.log"
    try:
        validate_host(profile)
        validate_allocation(profile, selected, selected)
        env = os.environ.copy()
        env.update(PYTHONHASHSEED="1807", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                   PYPTO_BENCH="1", PYPTO_BENCH_RAW="1", PYPTO_BENCH_ROUNDS="100", PYPTO_BENCH_WARMUP="5",
                   PYPTO_LOG_LEVEL="error", PYPTO_RUNTIME_LOG="error", SIMPLER_DEVICE_STRACE_ENABLE="1")
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        command = capture_command(case, selected, output / "raw-result.json", sys.executable)
        rc, timed_out = run_process(command, log, output, env, args.case_timeout)
        evidence.update(process_rc=rc, timed_out=timed_out, wall_seconds=time.monotonic() - started)
        with log.open(errors="replace") as stream:
            fault = any(DEVICE_FAULT.search(line) for line in stream)
        if fault:
            evidence.update(status="device_fault", error="device fault signature in model log")
        elif timed_out:
            recovered = check_recovery(selected, output)
            evidence.update(status="timeout", recovery_verified=recovered, safe_to_continue=recovered,
                            error="case wall-clock limit exceeded" if recovered else
                                  "case timeout; device recovery could not be verified")
        else:
            evidence["safe_to_continue"] = True
            raw = read_json(output / "raw-result.json")
            evidence.update(validate_result(raw, case, process_rc=rc))
            evidence.update(status="pass", fixture_sha256=raw["fixture_sha256"],
                            golden_sha256=raw["golden_sha256"])
    except TimeoutCleanupError as error:
        evidence.update(status="timeout", timed_out=True, recovery_verified=False, error=str(error))
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        evidence.update(status="fail", error=str(error))
    evidence.setdefault("wall_seconds", time.monotonic() - started)
    write_json(output / "measurement.json", evidence)
    return 0 if evidence["status"] == "pass" else 1


def run_group(args, manifest, profile):
    """Keep one queue allocation until all its measurements finish or it is unsafe."""
    groups = dict(allocation_groups(manifest))[args.worker_group]
    output = args.output_dir.resolve()
    state_path = output / f"group-{args.worker_group}" / "group-result.json"
    started = time.monotonic()
    state = {"device_count": args.worker_group, "case_ids": [case["case_id"] for case, _ in groups],
             "status": "running", "safe_to_continue": False, "completed_cases": [],
             "task_id": os.environ.get("TASKQUEUE_TASK_ID")}
    write_json(state_path, state)
    try:
        selected = profile["sequences"][str(args.worker_group)]
        validate_host(profile)
        validate_allocation(profile, selected, selected)
        state["safe_to_continue"] = True
        for case, _ in groups:
            remaining = args.group_timeout - (time.monotonic() - started)
            # Preserve the full per-case limit and recovery allowance. Do not
            # launch a case which the group watchdog could interrupt early.
            if remaining < args.case_timeout + 90:
                state["error"] = "group budget exhausted before another full case could start"
                break
            worker = copy.copy(args)
            worker.worker_case = case["case_id"]
            worker.output_dir = output / case["case_id"]
            worker.output_dir.mkdir()
            run_worker(worker, manifest, profile)
            evidence = read_json(worker.output_dir / "measurement.json")
            state["completed_cases"].append(case["case_id"])
            state["safe_to_continue"] = evidence.get("safe_to_continue") is True
            write_json(state_path, state)
            if not state["safe_to_continue"]:
                state["error"] = evidence.get("error", "case recovery could not be verified")
                break
        passed = len(state["completed_cases"]) == len(groups) and all(
            read_json(output / case_id / "measurement.json")["status"] == "pass"
            for case_id in state["completed_cases"])
        state["status"] = "pass" if passed else "incomplete"
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        state.update(status="incomplete", safe_to_continue=False, error=str(error))
    state["wall_seconds"] = time.monotonic() - started
    write_json(state_path, state)
    return 0 if state["status"] == "pass" else 1


def queue_command(count, profile, profile_path, output, args, env, remaining):
    # These are provisional safety limits, not measured execution estimates.
    task_limit = args.group_timeout + 30  # Final checkpoint and worker exit.
    # task-submit --timeout counts both queue wait and execution, not just
    # acquisition. Reserve client-exit overhead within the suite budget.
    wait_limit = min(args.queue_timeout, int(remaining) - 30)
    if wait_limit <= task_limit:
        raise ContractError("suite budget exhausted before another full group could start")
    command = [sys.executable, str(ROOT / ".github/scripts/ci_operator_perf.py"), "suite",
               "--worker-group", str(count), "--device-profile", str(profile_path),
               "--output-dir", str(output), "--case-timeout", str(args.case_timeout),
               "--group-timeout", str(args.group_timeout)]
    forwarded = {key: env[key] for key in ("PYPTO_SRC", "PTO_ISA_COMMIT", "PTOAS_ROOT")}
    forwarded["PYTHONPATH"] = str(ROOT)
    payload = (f"cd {shlex.quote(str(ROOT))} && source activate.sh && "
               + shlex.join(["env", *(f"{key}={value}" for key, value in forwarded.items()), *command]))
    selected = profile["sequences"][str(count)]
    return ["task-submit", "--device", ",".join(map(str, selected)),
            "--timeout", str(wait_limit), "--max-time", str(task_limit), "--run", payload]


def wait_for_queue(command, env, log, refresh):
    """Checkpoint case progress while one group holds its allocation."""
    timeout = int(command[command.index("--timeout") + 1]) + 30
    deadline = time.monotonic() + timeout
    with log.open("w") as stream:
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        try:
            while True:
                refresh()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    return process.wait(timeout=min(2, remaining))
                except subprocess.TimeoutExpired:
                    continue
        finally:
            if process.poll() is None:
                # Give task-submit's signal handler time to cancel its own
                # pending/running task before resorting to killing the client.
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def collect_measurements(suite, output):
    """Also used after queue cleanup to salvage the last worker checkpoints."""
    records = {case["case_id"]: case for case in suite["cases"]}
    for record in records.values():
        if record.get("shared_measurement"):
            continue
        path = output / record["case_id"] / "measurement.json"
        if not path.is_file():
            continue
        evidence = read_json(path)
        if evidence.pop("case_id", None) != record["case_id"]:
            raise ContractError("measurement belongs to another case")
        if evidence.get("status") == "running" and record["status"] not in ("not_run", "running"):
            continue
        if evidence.get("status") == "pass":
            record.pop("error", None)
        record.update(evidence)
    sync_shared_measurements(suite)


def sync_shared_measurements(suite):
    records = {case["case_id"]: case for case in suite["cases"]}
    for record in records.values():
        if record.get("shared_measurement"):
            identity = {key: record[key] for key in ("case_id", "case_contract", "shared_measurement", "previous", "week")
                        if key in record}
            source = records[record["measurement_case"]]
            record.clear()
            record.update(copy.deepcopy(source), **identity)


def render_report(suite):
    lines = [f"Operator performance: {suite['logical_date']} (CI)",
             f"Run: `{suite['run_id']}`; status: **{suite['status']}**.", "",
             "Metric: minimum rank median of summed operator dispatch Effective times (us).",
             "Communication and signal cleanup are included; this is not model step latency.", "",
             f"Sampling: warmup={suite.get('sampling', {}).get('warmup', 'unknown')}, "
             f"rounds={suite.get('sampling', {}).get('rounds', 'unknown')}; "
             f"golden replayed={suite.get('golden_replayed', 'unknown')}. "
             f"{suite.get('execution_count', 'unknown')} measurements, {len(suite['cases'])} report rows.",
             "Shared rows reference one measurement, not independent samples. Wall time excludes queue wait.", "",
             "| Case | Devices | Status | us | Rank spread us | Wall s | Measurement | Previous change | Week change |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for case in suite["cases"]:
        def delta(key):
            value = case.get(key, {}).get("delta_pct")
            return "N/A" if value is None else f"{value:+.2f}%"
        metric = f"{case['metric_us']:.2f}" if case.get("status") == "pass" else "N/A"
        spread = f"{case['rank_spread_us']:.2f}" if "rank_spread_us" in case else "N/A"
        wall = f"{case['wall_seconds']:.1f}" if "wall_seconds" in case else "N/A"
        measurement = (f"shared: {case['measurement_case']}" if case.get("shared_measurement")
                       else case.get("measurement_case", case["case_id"]))
        selected = ",".join(map(str, case["devices"]))
        lines.append(f"| {case['case_id']} | {selected} | {case['status']} | {metric} | "
                     f"{spread} | {wall} | {measurement} | {delta('previous')} | {delta('week')} |")
    lines.extend(["", "Changes describe the complete CI software stack, not an isolated kernel edit.",
                  "Positive changes mean slower. N/A means missing or incompatible evidence.", ""])
    for case in suite["cases"]:
        if case.get("error"):
            error = str(case["error"]).replace("\n", " ")
            lines.append(f"- {case['case_id']}: {error}")
        if case.get("queue_error"):
            lines.append(f"- {case['case_id']} queue: {case['queue_error']}")
        for key in ("previous", "week"):
            info = case.get(key)
            if info:
                components = ", ".join(info.get("changed_components", [])) or "none"
                detail = info.get("reason") or (
                    f"baseline run {info.get('baseline_run_id')}; changed components: {components}")
                lines.append(f"- {case['case_id']} {key}: {detail}")
    if suite.get("error"):
        lines.extend(["", str(suite["error"])])
    if suite.get("limits_seconds"):
        lines.append(f"Provisional limits (seconds): {suite['limits_seconds']}. Calibrate using measured wall times.")
    for allocation in suite.get("allocations", []):
        lines.append(f"- Allocation {allocation['group_id']}: {allocation['status']}; "
                     f"devices={allocation['devices']}; measurements={len(allocation['case_ids'])}; "
                     f"submission wall seconds={allocation.get('submission_wall_seconds', 'N/A')}")
        if allocation.get("error"):
            lines.append(f"  {allocation['error']}")
    state = suite.get("execution", {})
    if state:
        lines.append(f"CI execution: suite exit={state.get('suite_exit_code', 'unknown')}; "
                     f"wrapper step={state.get('step_outcome', 'unknown')}.")
        for error in state.get("errors", []):
            lines.append(f"- {error['phase']}: {error['error']}")
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
    started = time.monotonic()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    groups = allocation_groups(manifest)
    suite = {"schema_version": 1, "suite_id": manifest["suite_id"], "run_id": run_id,
             "logical_date": args.date, "variant": "ci", "status": "running",
             "coverage_complete": False, "sampling": SAMPLING, "golden_replayed": False,
             "execution_count": len(measurement_groups(manifest)), "limits_provisional": True,
             "limits_seconds": {"case": args.case_timeout, "queue": args.queue_timeout,
                                "group": args.group_timeout, "suite": args.suite_timeout},
             "allocations": [{"group_id": f"group-{count}", "status": "not_run",
                              "devices": profile["sequences"][str(count)],
                              "case_ids": [case["case_id"] for case, _ in measurements]}
                             for count, measurements in groups], "cases": []}
    for case in manifest["cases"]:
        source = case.get("measurement_case", case["case_id"])
        suite["cases"].append({"case_id": case["case_id"], "run_id": run_id, "status": "not_run",
                               "logical_date": args.date, "variant": "ci",
                               "measurement_case": source, "measurement_id": f"{run_id}/{source}",
                               "shared_measurement": source != case["case_id"],
                               "devices": profile["sequences"][str(case["device_count"])],
                               "case_contract": digest([case, manifest["metric"], SAMPLING,
                                                        manifest["contract_id"]])})
    records = {record["case_id"]: record for record in suite["cases"]}

    def checkpoint():
        suite["wall_seconds"] = time.monotonic() - started
        write_json(output / "suite-result.json", suite)
        (output / "report.md").write_text(render_report(suite))

    checkpoint()
    try:
        validate_host(profile)
        if os.environ.get("TASKQUEUE_INSIDE") == "1":
            raise ContractError("the suite submits grouped allocations; do not wrap it in task-submit")
        source_sha = source_identity()
        toolchain = capture_toolchain()
        host = capture_host(profile, output)
        suite.update(source_sha=source_sha, toolchain=toolchain, host=host)
        prior = load_baseline(args.baseline, args.date)
        week = load_baseline(args.week_baseline, args.date, weekly=True)
        profile_path = output / "device-profile.json"
        write_json(profile_path, profile)
    except (OSError, ValueError, KeyError, ImportError, subprocess.SubprocessError) as error:
        suite.update(status="preflight_failed", error=str(error))
        checkpoint()
        return 1

    env = os.environ.copy()
    for key in ("GH_TOKEN", "GITHUB_TOKEN"):
        env.pop(key, None)
    stopped = None
    for (count, measurements), allocation in zip(groups, suite["allocations"]):
        rows = [row for _, aliases in measurements for row in aliases]
        if stopped:
            for row in rows:
                records[row["case_id"]]["error"] = stopped
            continue
        group_dir = output / allocation["group_id"]
        group_dir.mkdir()
        try:
            remaining = args.suite_timeout - (time.monotonic() - started)
            command = queue_command(count, profile, profile_path, output, args, env, remaining)
        except ContractError as error:
            stopped = str(error)
            for row in rows:
                records[row["case_id"]]["error"] = stopped
            continue
        allocation.update(status="queued", command=command, log=f"{allocation['group_id']}/queue.log")
        for row in rows:
            record = records[row["case_id"]]
            record.update(source_sha=source_sha, toolchain=toolchain,
                          device_identity={**host, "selected": record["devices"]},
                          allocation_group=allocation["group_id"], log=f"{record['measurement_case']}/run.log")
        checkpoint()
        submitted = time.monotonic()
        outcome = {"status": "queue_failed", "safe_to_continue": False}

        def refresh():
            collect_measurements(suite, output)
            state_path = group_dir / "group-result.json"
            if state_path.is_file() and allocation["status"] == "queued":
                allocation["status"] = "running"
            for row in rows:
                record = records[row["case_id"]]
                record["previous"] = comparison(record, prior.get(row["case_id"]))
                record["week"] = comparison(record, week.get(row["case_id"]))
            checkpoint()

        try:
            rc = wait_for_queue(command, env, group_dir / "queue.log", refresh)
            allocation["queue_rc"] = rc
            refresh()
            evidence = read_json(group_dir / "group-result.json")
            if evidence.get("device_count") != count or evidence.get("case_ids") != allocation["case_ids"]:
                raise ContractError("group result belongs to another allocation")
            if evidence.get("status") not in ("pass", "incomplete"):
                raise ContractError("queue returned without a completed group result")
            if evidence["status"] == "pass" and any(
                records[case_id]["status"] != "pass" for case_id in allocation["case_ids"]
            ):
                raise ContractError("passing group has missing or unsuccessful measurements")
            outcome = evidence
            # A transport/reporting failure does not erase a validated measurement.
            if rc != (0 if evidence["status"] == "pass" else 1):
                outcome["error"] = f"queue exit {rc} disagrees with completed {evidence['status']} group"
                outcome["safe_to_continue"] = False
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
            outcome.update(error=f"queue completion could not be verified: {error}")
        # A worker may have finished another case just before a client error.
        try:
            refresh()
        except (OSError, ValueError, KeyError, TypeError) as error:
            outcome.update(status="queue_failed", safe_to_continue=False,
                           error=f"group checkpoints could not be verified: {error}")
        allocation.update(outcome, submission_wall_seconds=time.monotonic() - submitted)
        if not outcome.get("safe_to_continue"):
            stopped = "remaining measurements stopped: " + outcome.get("error", "unknown queue state")
        if outcome["status"] == "queue_failed":
            first = records[measurements[0][0]["case_id"]]
            if first["status"] == "not_run":
                first.update(status="queue_failed", error=outcome["error"])
        for row in rows:
            record = records[row["case_id"]]
            if record["status"] in ("not_run", "running"):
                record["error"] = outcome.get("error", "group ended without completing this measurement")
                if record["status"] == "running":
                    record["status"] = "interrupted"
        sync_shared_measurements(suite)
        for row in rows:
            record = records[row["case_id"]]
            record["previous"] = comparison(record, prior.get(row["case_id"]))
            record["week"] = comparison(record, week.get(row["case_id"]))
        checkpoint()
    suite["coverage_complete"] = all(case["status"] == "pass" for case in suite["cases"])
    suite["stop_reason"] = stopped
    suite["status"] = "pass" if suite["coverage_complete"] else "incomplete"
    checkpoint()
    return 0 if suite["coverage_complete"] and not stopped else 1



# Model capture runs in a fresh process for each distinct workload.

def benchmark_payload(stats):
    """Read the public RunResult.bench data without guessing a log layout."""
    if stats is None:
        raise ContractError("benchmark unavailable or skipped")
    def invocation(item):
        return {"inv": item.inv, "task": item.task, "effective_us": item.effective_us,
                "task_name": getattr(item, "task_name", item.task)}

    grid = stats.rounds_dispatches
    if grid:
        rounds = [{str(pid): [invocation(item) for item in items] for pid, items in row.items()}
                  for row in grid]
    else:
        # Only the single-card consumer may accept this representation.
        rounds = [{str(item.pid): [invocation(item)]} for item in stats.invocations]
    return {"rounds": stats.rounds, "warmup": stats.warmup,
            "fallback_flattened": stats.fallback_flattened,
            "unstable_dispatch_slots": getattr(stats, "unstable_dispatch_slots", False),
            "all_zero_device": stats.all_zero_device, "distributed_grid": bool(grid),
            "samples": rounds}


def tensor_fingerprint(values, specs, direction):
    """Hash actual contents without serializing multi-GB weight snapshots."""
    import torch

    result = hashlib.sha256()
    for spec in sorted(specs, key=lambda item: item.name):
        is_tensor = hasattr(spec, "shape")
        include = spec.is_input if is_tensor and direction == "input" else (
            spec.is_output if is_tensor else direction == "input")
        if not include:
            continue
        value = values[spec.name]
        result.update(spec.name.encode() + b"\0")
        if isinstance(value, torch.Tensor):
            result.update(json.dumps([list(value.shape), str(value.dtype)]).encode())
            raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            result.update(memoryview(raw))
        else:
            result.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
    return result.hexdigest()


def describe_specs(specs):
    return [{"name": spec.name, "shape": list(spec.shape), "dtype": str(spec.dtype),
             "direction": spec.direction, "resident": spec.resident}
            for spec in specs if hasattr(spec, "shape")]


def make_capture(original_run, case, output, selected):
    """Install a process-local observer; model code keeps its own golden/tolerances."""
    calls = 0

    def capture(**kwargs):
        nonlocal calls
        calls += 1
        if calls != 1:
            output.unlink(missing_ok=True)
            raise ContractError("a case must call golden.run exactly once")
        specs = kwargs["specs"]
        check_specs(case, describe_specs(specs))
        cfg = kwargs["config"]
        distributed = cfg.get("distributed_config")
        actual_devices = list(distributed.device_ids) if distributed else [cfg["device_id"]]
        if actual_devices != selected or cfg.get("platform") != "a2a3":
            raise ContractError("the model's RunConfig differs from its allocated devices/platform")
        if cfg.get("enable_chip_swimlane", 0) or kwargs.get("compile_only") or kwargs.get("runtime_dir"):
            raise ContractError("official measurement requires a fresh compile with swimlane disabled")
        golden_fn = kwargs.get("golden_fn")
        if golden_fn is None or kwargs.get("golden_data"):
            raise ContractError("this capture requires the model's in-memory golden")
        payload = {"case_id": case["case_id"], "sampling": SAMPLING, "passed": False}
        overrides = case.get("run_config", {})
        kwargs["config"] = {**cfg, **overrides}
        payload["run_config"] = overrides

        def golden(values):
            payload["fixture_sha256"] = tensor_fingerprint(values, specs, "input")
            golden_fn(values)
            payload["golden_sha256"] = tensor_fingerprint(values, specs, "output")

        kwargs["golden_fn"] = golden
        kwargs["save_data"] = False
        result = original_run(**kwargs)
        payload.update(passed=result.passed, error=result.error, specs=describe_specs(specs),
                       work_dir=str(result.work_dir) if result.work_dir else None)
        if result.passed:
            payload["benchmark"] = benchmark_payload(result.bench)
        write_json(output, payload)
        return result

    return capture


def model_arguments(case, selected):
    arguments = case_arguments(case, selected)
    if case["entrypoint"] == "models/deepseek_v4_flash_mtp/decode_moe.py":
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        parser.add_argument("--experts-per-rank", type=int, choices=(16,), required=True)
        override, arguments = parser.parse_known_args(arguments)
        import config

        # The existing MoE entry preserves this local expert count when applying --ep.
        # Override only this fresh CI process, before importing any model kernels.
        config.FLASH = replace(config.FLASH, n_routed_experts=override.experts_per_rank * config.EP_WORLD_SIZE)
    return arguments


def run_case(args):
    manifest = load_manifest()
    case = next(case for case in manifest["cases"] if case["case_id"] == args.case)
    selected = devices(args.devices)
    if len(selected) != case["device_count"]:
        raise ContractError("device count differs from the case")
    if os.environ.get("PYTHONHASHSEED") != str(SAMPLING["seed"]):
        raise ContractError("PYTHONHASHSEED must be set before Python starts")
    import numpy as np
    import torch
    import golden

    random.seed(SAMPLING["seed"])
    np.random.seed(SAMPLING["seed"])
    torch.manual_seed(SAMPLING["seed"])
    golden.run = make_capture(golden.run, case, args.output, selected)
    entry = ROOT / case["entrypoint"]
    sys.path.insert(0, str(entry.parent))
    sys.argv = [str(entry), *model_arguments(case, selected)]
    runpy.run_path(str(entry), run_name="__main__")
    if not args.output.is_file():
        raise ContractError("model produced no golden result")


# GitHub history and result publication stay outside the allocation.

ZONE = timezone(timedelta(hours=8))
WORKFLOW = ".github/workflows/daily_ci.yml"
MAX_ARCHIVE = 64 * 1024 * 1024
MAX_RESULT = 16 * 1024 * 1024


def logical_date(timestamp):
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(ZONE).date()


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def github_read(open_response, read_response):
    """Retry a bounded GET including body reads, without logging signed URLs."""
    for attempt in range(4):
        try:
            with open_response() as response:
                return read_response(response)
        except (URLError, TimeoutError, ConnectionError, SSLEOFError, IncompleteRead) as error:
            if isinstance(error, HTTPError):
                retryable = error.code in (408, 500, 502, 503, 504)
                label = f"HTTP {error.code}"
                error.close()
            else:
                retryable = not isinstance(getattr(error, "reason", error), SSLCertVerificationError)
                label = type(error).__name__
            if not retryable or attempt == 3:
                if isinstance(error, IncompleteRead):
                    raise URLError(error) from error
                raise
            delay = 2 ** (attempt + 1)
            print(f"[perf] GitHub GET failed ({label}); retry {attempt + 2}/4 in {delay}s", flush=True)
            time.sleep(delay)


class GitHub:
    def __init__(self, repository, token):
        self.base = f"https://api.github.com/repos/{repository}"
        self.headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                        "X-GitHub-Api-Version": "2022-11-28"}

    def request(self, path):
        return build_opener(NoRedirect).open(Request(self.base + path, headers=self.headers), timeout=45)

    def json(self, path):
        return github_read(lambda: self.request(path), json.load)

    def archive(self, artifact_id):
        def open_archive():
            try:
                return self.request(f"/actions/artifacts/{int(artifact_id)}/zip")
            except HTTPError as error:
                if error.code not in (301, 302, 303, 307, 308):
                    raise
                location = error.headers["Location"]
                error.close()
                if urlparse(location).scheme != "https":
                    raise ContractError("artifact download must use HTTPS") from error
                # The signed blob URL needs no GitHub token. Never forward it there.
                return urlopen(location, timeout=45)

        data = github_read(open_archive, lambda response: response.read(MAX_ARCHIVE + 1))
        if len(data) > MAX_ARCHIVE:
            raise ContractError("historical artifact exceeds the size limit")
        return data

    def pages(self, path, key):
        items = []
        for page in range(1, 5):
            separator = "&" if "?" in path else "?"
            batch = self.json(f"{path}{separator}per_page=100&page={page}")[key]
            items.extend(batch)
            if len(batch) < 100:
                return items
        raise ContractError("history pagination limit reached; no partial history will be selected")


def select_runs(runs, today):
    by_date = {}
    for run in runs:
        if (run.get("head_branch") != "main" or run.get("event") not in ("schedule", "workflow_dispatch")
                or run.get("path") != WORKFLOW):
            continue
        day = logical_date(run["created_at"])
        if day >= today:
            continue
        prior = by_date.get(day)
        if prior is None or (run["id"], run["run_attempt"]) > (prior["id"], prior["run_attempt"]):
            by_date[day] = run
    return {"previous": by_date[max(by_date)] if by_date else None,
            "week": by_date.get(today - timedelta(days=7))}


def suite_from_archive(data, run, repository):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        candidates = [item for item in archive.infolist() if item.filename == "suite-result.json"]
        if len(candidates) != 1 or candidates[0].file_size > MAX_RESULT:
            raise ContractError("artifact must contain one bounded suite-result.json")
        suite = json.loads(archive.read(candidates[0]))
    if not isinstance(suite, dict) or not isinstance(suite.get("cases"), list):
        raise ContractError("artifact has no suite/case structure")
    cases = suite["cases"]
    if (len(cases) != len(CASE_IDS) or any(not isinstance(case, dict) for case in cases) or
            {case.get("case_id") for case in cases} != CASE_IDS):
        raise ContractError("artifact does not describe the twenty official cases")
    expected = {"repository": repository, "workflow_run_id": str(run["id"]),
                "run_attempt": str(run["run_attempt"]), "head_sha": run["head_sha"],
                "head_branch": "main", "workflow_path": WORKFLOW}
    if (suite.get("ci") != expected or suite.get("logical_date") != logical_date(run["created_at"]).isoformat()
            or suite.get("suite_id") != "dsv4-operators" or suite.get("variant") != "ci"):
        raise ContractError("artifact identity does not match its workflow run")
    return suite


def retrieve_history(client, repository, today, destination):
    query = urlencode({"branch": "main", "created": ">=" + (today - timedelta(days=14)).isoformat()})
    runs = client.pages(f"/actions/workflows/daily_ci.yml/runs?{query}", "workflow_runs")
    selected = select_runs(runs, today)
    history = {}
    for name, run in selected.items():
        info = {"run_id": run["id"] if run else None, "path": None}
        history[name] = info
        if run is None:
            info["reason"] = "no workflow run for the selected date"
            continue
        info.update(date=logical_date(run["created_at"]).isoformat(), attempt=run["run_attempt"])
        try:
            if run["status"] != "completed":
                raise ContractError("latest attempt is still running; an older attempt is not substituted")
            artifacts = client.pages(f"/actions/runs/{run['id']}/artifacts", "artifacts")
            matching = [item for item in artifacts if item["name"] == f"operator-perf-{run['run_attempt']}"
                        and not item["expired"]]
            if len(matching) != 1:
                raise ContractError("latest attempt has no unique retained performance artifact")
            suite = suite_from_archive(client.archive(matching[0]["id"]), run, repository)
            path = destination / f"{name}.json"
            write_json(path, suite)
            info["path"] = str(path)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
            info["reason"] = str(error)
    return history


def suite_command(profile_path, output, history, env):
    command = [sys.executable, str(ROOT / ".github/scripts/ci_operator_perf.py"), "suite",
               "--device-profile", str(profile_path), "--output-dir", str(output),
               "--date", env["PERF_LOGICAL_DATE"]]
    for name, default in (("CASE", 1800), ("GROUP", 3600), ("QUEUE", 7200), ("SUITE", 10800)):
        value = int(env.get(f"OPERATOR_PERF_{name}_TIMEOUT") or default)
        if value <= 0:
            raise ContractError("performance timeout settings must be positive")
        command += [f"--{name.lower()}-timeout", str(value)]
    for name, option in (("previous", "--baseline"), ("week", "--week-baseline")):
        if history.get(name, {}).get("path"):
            command += [option, history[name]["path"]]
    return command


def publish(output, published, context, history, execution):
    source = output / "suite-result.json"
    if source.is_file():
        suite = read_json(source)
        if suite.get("allocations"):
            # Queue cleanup may finish after the suite's last polling update.
            collect_measurements(suite, output)
            for name in ("previous", "week"):
                path = history.get(name, {}).get("path")
                baseline = load_baseline(path, suite["logical_date"], weekly=name == "week")
                for case in suite["cases"]:
                    case[name] = comparison(case, baseline.get(case["case_id"]))
        suite["ci"] = context
        suite["execution"] = execution
        if suite.get("status") == "running":
            suite.update(status="interrupted", coverage_complete=False)
        for case in suite["cases"]:
            if case.get("status") == "running":
                case.update(status="interrupted", error="wrapper ended before measurement completion")
        for allocation in suite.get("allocations", []):
            if allocation["status"] in ("queued", "running"):
                allocation.update(status="interrupted", error="wrapper ended before allocation completion")
        write_json(published / "suite-result.json", suite)
        report = render_report(suite)
        # Publish only the small evidence files, never builds or tensor dumps.
        for case in suite["cases"]:
            for name in ("raw-result.json", "measurement.json", "run.log", "queue.log",
                         "recovery-0.txt", "recovery-1.txt"):
                path = output / case["case_id"] / name
                if path.is_file() and not path.is_symlink():
                    target = published / case["case_id"] / name
                    target.parent.mkdir(exist_ok=True)
                    if path.stat().st_size <= MAX_RESULT:
                        shutil.copyfile(path, target)
                    else:
                        report += f"\n- {case['case_id']}/{name}: too large to upload.\n"
        inventory = output / "npu-smi-info.txt"
        if inventory.is_file():
            shutil.copyfile(inventory, published / inventory.name)
        for allocation in suite.get("allocations", []):
            for name in ("queue.log", "group-result.json"):
                path = output / allocation["group_id"] / name
                if path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_RESULT:
                    target = published / allocation["group_id"] / name
                    target.parent.mkdir(exist_ok=True)
                    shutil.copyfile(path, target)
        complete = suite.get("status") == "pass" and suite.get("coverage_complete") is True
    else:
        report = "Operator performance failed before producing results.\n"
        for error in execution.get("errors", []):
            report += f"- {error['phase']}: {error['error']}\n"
        complete = False
    report += "\nHistory selection:\n"
    for name, info in history.items():
        report += f"- {name}: run {info.get('run_id')}, {info.get('date')}; {info.get('reason', 'artifact loaded')}\n"
    (published / "report.md").write_text(report)
    return complete


def finalize(env):
    """Salvage checkpoints after a failed/cancelled wrapper, after queue cleanup."""
    root = Path(env["RUNNER_TEMP"]) / f"operator-perf-{env['GITHUB_RUN_ID']}-{env['GITHUB_RUN_ATTEMPT']}"
    published = root / "published"
    if (root / "work/suite-result.json").is_file():
        submission = read_json(published / "submission.json")
        history = read_json(published / "history-selection.json")
        state_path = published / "execution.json"
        execution = read_json(state_path) if state_path.is_file() else {}
        execution["step_outcome"] = env.get("PERF_STEP_OUTCOME", "unknown")
        write_json(state_path, execution)
        publish(root / "work", published, submission["ci"], history, execution)
    if (published / "report.md").is_file() and env.get("GITHUB_STEP_SUMMARY"):
        with Path(env["GITHUB_STEP_SUMMARY"]).open("a") as summary:
            summary.write((published / "report.md").read_text())
    return 0


def progress_snapshot(work):
    """Read bounded, existing checkpoints without touching the running workload."""
    try:
        suite = read_json(work / "suite-result.json")
        unique = [case for case in suite["cases"] if not case.get("shared_measurement")]
        completed = sum(case["status"] not in ("not_run", "running") for case in unique)
        message = f"{completed}/{len(unique)} measurements completed; suite={suite['status']}"
        allocation = next((group for group in suite.get("allocations", [])
                           if group["status"] in ("queued", "running")), None)
        if allocation:
            message += f"; allocation={allocation['group_id']} devices={allocation['devices']}"
        active = next((case for case in unique if case["status"] == "running"), None)
        if active is None:
            if allocation:
                message += ("; waiting for group allocation (queue/worker startup)" if allocation["status"] == "queued"
                            else "; group worker startup/between measurements")
            return message
        case_dir = work / active["case_id"]
        message += f"; case={active['case_id']} devices={active['devices']}"
        measurement = case_dir / "measurement.json"
        if not measurement.is_file():
            return message + "; waiting for worker checkpoint (queue/worker startup)"
        message += f"; worker={read_json(measurement)['status']}"
        log = case_dir / "run.log"
        if log.is_file():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 8192))
                lines = stream.read(8192).decode(errors="replace").splitlines()
            stage = next((line for line in reversed(lines) if line.startswith("[RUN] ")), None)
            if stage:
                message += f"; last stage: {stage[:240]}"
        return message
    except FileNotFoundError:
        return "waiting for suite checkpoint (initialization/preflight)"
    except (OSError, ValueError, KeyError, TypeError) as error:
        return f"progress checkpoint unavailable: {type(error).__name__}"


def report_progress(work, stopped):
    while not stopped.is_set():
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"[perf] {timestamp} {progress_snapshot(work)}", flush=True)
        stopped.wait(30)


def run_ci():
    env = os.environ.copy()
    run_id, attempt = env["GITHUB_RUN_ID"], env["GITHUB_RUN_ATTEMPT"]
    root = Path(env["RUNNER_TEMP"]) / f"operator-perf-{run_id}-{attempt}"
    published = root / "published"
    published.mkdir(parents=True, exist_ok=False)
    repository = env["GITHUB_REPOSITORY"]
    context = {"repository": repository, "workflow_run_id": run_id, "run_attempt": attempt,
               "head_sha": env["GITHUB_SHA"], "head_branch": env["GITHUB_REF_NAME"], "workflow_path": WORKFLOW}
    history = {}
    execution = {"phase": "setup", "errors": []}
    write_json(published / "execution.json", execution)
    try:
        print(f"[perf] Resolving run {run_id} and historical baselines", flush=True)
        client = GitHub(repository, env["GH_TOKEN"])
        # A logical date belongs to the workflow, even when the queue runs past midnight.
        current = client.json(f"/actions/runs/{run_id}")
        today = logical_date(current["created_at"])
        env["PERF_LOGICAL_DATE"] = today.isoformat()
        (root / "history").mkdir()
        try:
            history = retrieve_history(client, repository, today, root / "history")
        except (OSError, ValueError, KeyError) as error:
            history = {"lookup": {"reason": str(error)}}
        write_json(published / "history-selection.json", history)
        print("[perf] History lookup finished; validating the host and preparing the suite", flush=True)
        profile = load_profile()
        profile["device_epoch"] = env.get("OPERATOR_PERF_DEVICE_EPOCH") or profile["device_epoch"]
        profile["hostname"] = env.get("OPERATOR_PERF_HOST", "")
        validate_host(profile)
        profile_path = root / "device-profile.json"
        write_json(profile_path, profile)
        command = suite_command(profile_path, root / "work", history, env)
        write_json(published / "submission.json", {"command": command, "ci": context, "logical_date": today.isoformat()})
        # Tokens are for artifact retrieval only, not for the queued model processes.
        for key in ("GH_TOKEN", "GITHUB_TOKEN"):
            env.pop(key, None)
        execution["phase"] = "suite"
        write_json(published / "execution.json", execution)
        print(f"[perf] Starting suite on {profile['hostname']}; progress every 30 seconds", flush=True)
        stopped = threading.Event()
        monitor = threading.Thread(target=report_progress, args=(root / "work", stopped), daemon=True)
        monitor.start()
        try:
            with (published / "suite.log").open("w") as log:
                rc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
        finally:
            stopped.set()
            monitor.join(timeout=2)
        print(f"[perf] Suite exited {rc}; {progress_snapshot(root / 'work')}; publishing results", flush=True)
        execution.update(suite_exit_code=rc, phase="publish")
        write_json(published / "execution.json", execution)
        complete = publish(root / "work", published, context, history, execution)
        execution["phase"] = "complete"
        write_json(published / "execution.json", execution)
        print(f"[perf] Results published; complete={complete}; suite_exit_code={rc}", flush=True)
        return 0 if rc == 0 and complete else 1
    except (OSError, ValueError, KeyError) as error:
        print(f"[perf] {execution['phase']} failed: {error}", flush=True)
        execution["errors"].append({"phase": execution["phase"], "error": str(error)})
        write_json(published / "execution.json", execution)
        (published / "report.md").write_text(f"Operator performance {execution['phase']} failed: {error}\n")
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    commands.add_parser("ci", help="Run Daily CI with GitHub history and publication")
    commands.add_parser("finalize", help="Publish final checkpoints after queue cleanup")
    capture = commands.add_parser("case", help="Capture one model inside its allocation")
    capture.add_argument("--case", required=True, choices=sorted(CASE_IDS))
    capture.add_argument("--devices", required=True)
    capture.add_argument("--output", required=True, type=Path)
    suite = commands.add_parser("suite", help="Submit the fixed suite or print its execution plan")
    suite.add_argument("--device-profile", type=Path)
    suite.add_argument("--output-dir", type=Path)
    suite.add_argument("--date", default=datetime.now(ZONE).date().isoformat())
    suite.add_argument("--baseline", type=Path)
    suite.add_argument("--week-baseline", type=Path)
    suite.add_argument("--case-timeout", type=int, default=1800)
    suite.add_argument("--group-timeout", type=int, default=3600)
    suite.add_argument("--queue-timeout", type=int, default=7200)
    suite.add_argument("--suite-timeout", type=int, default=10800)
    suite.add_argument("--worker-group", type=int, choices=(1, 8), help=argparse.SUPPRESS)
    suite.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "ci":
        return run_ci()
    if args.mode == "finalize":
        return finalize(os.environ)
    if args.mode == "case":
        return run_case(args)
    manifest = load_manifest()
    profile = load_profile(args.device_profile)
    datetime.fromisoformat(args.date)
    if min(args.case_timeout, args.group_timeout, args.queue_timeout, args.suite_timeout) <= 0:
        suite.error("timeouts must be positive")
    if args.group_timeout <= args.case_timeout + 90:
        suite.error("group timeout must exceed the case timeout plus 90 seconds")
    if not args.worker_group and args.queue_timeout <= args.group_timeout + 30:
        suite.error("queue completion timeout must exceed the group timeout plus 30 seconds")
    if args.dry_run:
        print(json.dumps(execution_plan(manifest, profile, sys.executable), indent=2))
        return 0
    if not args.output_dir:
        suite.error("--output-dir is required for execution")
    if args.worker_group:
        return run_group(args, manifest, profile)
    return execute(args, manifest, profile)


if __name__ == "__main__":
    raise SystemExit(main())
