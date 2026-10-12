# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run directory-selected A5 script entries and precision tests in the device queue."""

import argparse
import json
import re
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
V41_ROOT = ROOT / "models/deepseek_v4_1_flash"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_changes import all_entries, has_pytest_entry  # noqa: E402


def collect(entry, output):
    output.unlink(missing_ok=True)
    command = [sys.executable, "-m", "pytest", entry, "--collect-only", "-q",
               "--rootdir", str(ROOT), "--import-mode=importlib", "--a5-case-list", str(output)]
    result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, check=False)
    if result.returncode:
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
        raise subprocess.CalledProcessError(result.returncode, command)
    cases = json.loads(output.read_text())
    if not cases:
        raise ValueError(f"{entry}: no pytest precision cases collected")
    return cases


def queue_command(case):
    command = ["task-submit", "--device", "auto"]
    if case["cards"] > 1:
        command += ["--device-num", str(case["cards"]), "--ignore-whitelist"]
    args = ["python", "-m", "pytest", case["nodeid"], "--rootdir", str(ROOT),
            "--import-mode=importlib", "-v", "-s"]
    for name, value in case["axes"].items():
        args += [f"--{name}", str(value)]
    run = (f"cd {shlex.quote(str(ROOT))} && source activate.sh && "
           f"PYTHONPATH={shlex.quote(str(ROOT))} {shlex.join(args)} --device \"$TASK_DEVICE\"")
    return command + ["--timeout", "3600", "--max-time", "900", "--run", run]


def script_cases(entry):
    """Expand V4.1 CLI combinations; run other scripts once with their device hint."""
    matrix = [{"cards": 1, "arguments": []}]
    if (ROOT / entry).resolve().is_relative_to(V41_ROOT):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "matrix.json"
            subprocess.run(
                [sys.executable, str(ROOT / ".github/scripts/a5_parallel_cases.py"), entry,
                 "--output", str(output)],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True,
            )
            matrix = json.loads(output.read_text())
    cases = []
    marker = re.search(r"^#\s*ci:\s*devices=(\d+)", (ROOT / entry).read_text(), re.MULTILINE)
    for configuration in matrix:
        arguments = configuration["arguments"]
        cards = max(configuration["cards"], int(marker.group(1)) if marker else 1)
        label = " ".join([entry, *arguments])
        args = ["python", entry, "-p", "a5", *arguments]
        run = (f"cd {shlex.quote(str(ROOT))} && source activate.sh && "
               f"PYTHONPATH={shlex.quote(str(ROOT))} {shlex.join(args)} -d \"$TASK_DEVICE\"")
        command = ["task-submit", "--device", "auto"]
        if cards > 1:
            command += ["--device-num", str(cards), "--ignore-whitelist"]
        command += ["--timeout", "3600", "--max-time", "900", "--run", run]
        cases.append({"nodeid": label, "command": command})
    if not cases:
        raise ValueError(f"{entry}: no script cases declared")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entries", nargs="*", help="A5 entries; defaults to all configured directories")
    parser.add_argument("--results", type=Path, help="optional TSV result report")
    parser.add_argument("--mode", choices=("auto", "cli"), default="auto",
                        help="auto uses precision pytest nodes; cli expands declared script combinations")
    args = parser.parse_args()
    entries = args.entries or all_entries("a5")
    if args.results:
        args.results.write_text("")
    failed_cases = []
    statuses = {}

    def write_results():
        if args.results:
            args.results.write_text("".join(f"{label}\t{status}\n" for label, status in statuses.items()))

    def report(label, passed):
        if not passed:
            failed_cases.append(label)
        statuses[label] = "pass" if passed else "fail"
        write_results()

    plan = []
    with tempfile.TemporaryDirectory() as directory:
        for entry in entries:
            output = Path(directory) / "cases.json"
            try:
                cases = (collect(entry, output)
                         if args.mode == "auto" and has_pytest_entry(entry)
                         else script_cases(entry))
            except (subprocess.CalledProcessError, ValueError, OSError) as error:
                if isinstance(error, subprocess.CalledProcessError) and error.stderr:
                    print(error.stderr, flush=True)
                print(f"::error file={entry}::case collection failed: {error}", flush=True)
                report(entry, False)
                continue
            for case in cases:
                statuses[case["nodeid"]] = "missing"
                plan.append((entry, case))
        write_results()
        for entry, case in plan:
            print(f"::group::{case['nodeid']}", flush=True)
            try:
                result = subprocess.run(case.get("command") or queue_command(case), stdin=subprocess.DEVNULL, check=False)
                passed = result.returncode == 0
            except OSError as error:
                print(error, flush=True)
                passed = False
            report(case["nodeid"], passed)
            if not passed:
                print(f"::error file={entry}::FAIL {case['nodeid']}", flush=True)
            print("::endgroup::", flush=True)
    if failed_cases:
        print("FAILED CASES:", flush=True)
        for label in failed_cases:
            print(f"  {label}", flush=True)
        return 1
    print("All selected cases passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
