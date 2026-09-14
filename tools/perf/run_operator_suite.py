# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run ten fixed operator workloads inside an exact even-device allocation."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.perf.operator_contract import (
    CASE_IDS, ContractError, ROOT, SAMPLING, case_arguments, digest, load_manifest,
    load_profile, read_json, validate_allocation, write_json,
)
from tools.perf.operator_provenance import capture_host, capture_toolchain, source_identity
from tools.perf.operator_results import comparison, validate_result


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
        if profile["device_epoch"].startswith("set-to-"):
            raise ContractError("replace the example device epoch with the deployment's actual epoch")
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
        if args.model and case["model"] != args.model or args.case and case["case_id"] != args.case:
            record["status"] = "not_selected"
            continue
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
        if args.save_data:
            command.append("--save-data")
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
            record["previous"] = comparison(record, prior.get(case["case_id"]), mode="ci_history")
            record["week"] = comparison(record, week.get(case["case_id"]), mode="ci_history")
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
    if not args.keep_builds and not args.save_data:
        for case_dir, build in build_dirs:
            # Delete only successful builds inside this run's private case directory.
            if build.is_relative_to(case_dir / "build_output") and build != case_dir / "build_output":
                shutil.rmtree(build)
    return 0 if suite["coverage_complete"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-profile", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", choices=("mtp", "dspark"))
    parser.add_argument("--case", choices=sorted(CASE_IDS))
    parser.add_argument("--date", default=datetime.now(timezone(timedelta(hours=8))).date().isoformat())
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--week-baseline", type=Path)
    parser.add_argument("--case-timeout", type=int, default=3600)
    parser.add_argument("--keep-builds", action="store_true")
    parser.add_argument("--save-data", action="store_true")
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
