# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Collect implementation-local pytest cases and run each in the A5 queue."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/deepseek_v4_1_flash"


def collect(entry, output):
    output.unlink(missing_ok=True)
    command = [sys.executable, "-m", "pytest", entry, "--collect-only", "-q",
               "--rootdir", str(ROOT), "--import-mode=importlib", "--a5-case-list", str(output)]
    subprocess.run(command, cwd=ROOT, check=True)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entries", nargs="*", help="marked A5 implementations; defaults to all")
    parser.add_argument("--results", type=Path, help="optional TSV result report")
    args = parser.parse_args()
    entries = args.entries or [str(path.relative_to(ROOT)) for path in sorted(MODEL.glob("*.py"))
                               if "# ci: a5" in path.read_text().splitlines()]
    if args.results:
        args.results.write_text("")
    failed = False

    def report(label, passed):
        nonlocal failed
        failed |= not passed
        if args.results:
            with args.results.open("a") as stream:
                stream.write(f"{label}\t{'pass' if passed else 'fail'}\n")

    with tempfile.TemporaryDirectory() as directory:
        for entry in entries:
            output = Path(directory) / "cases.json"
            try:
                cases = collect(entry, output)
            except (subprocess.CalledProcessError, ValueError, OSError) as error:
                print(f"::error file={entry}::pytest collection failed: {error}", flush=True)
                report(entry, False)
                continue
            for case in cases:
                print(f"::group::{case['nodeid']}", flush=True)
                try:
                    result = subprocess.run(queue_command(case), stdin=subprocess.DEVNULL, check=False)
                    passed = result.returncode == 0
                except OSError as error:
                    print(error, flush=True)
                    passed = False
                report(case["nodeid"], passed)
                if not passed:
                    print(f"::error file={entry}::FAIL {case['nodeid']}", flush=True)
                print("::endgroup::", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
